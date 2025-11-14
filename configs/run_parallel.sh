#!/usr/bin/env bash
set -euo pipefail

# Run multiple W&B sweeps in parallel with optional YAML overrides.
# Each task selects a YAML config and optionally overrides fields.
#
# Usage examples:
#   bash configs/run_parallel.sh \
#     --task "/home/wanzl/project/FM-IRL/configs/ant/airl.yaml|name=Ant-airl-test,parameters.seed.value=1" \
#     --task "/home/wanzl/project/FM-IRL/configs/ant/wail.yaml|name=Ant-wail-A,parameters.seed.value=2" \
#     --max-procs 2
#
# Override syntax:
#   - Comma-separated key=value pairs after a '|'
#   - Dot notation supported for deep keys (e.g., parameters.seed.value=3)
#   - If key is a plain word and exists under .parameters, we set its .value
#   - If top-level key exists, we set it; otherwise we create parameters.<key>.value
#   - Values are type-cast via yaml.safe_load (supports numbers, booleans, lists)
#
# Notes:
#   - Uses utils/wandb.sh to create and run W&B sweeps for each (possibly temp) config
#   - Temporary configs are written under: /tmp/fmirl_sweeps/<timestamp>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WANDB_SWEEP_SH="${REPO_ROOT}/utils/wandb.sh"

if [[ ! -f "${WANDB_SWEEP_SH}" ]]; then
  echo "Cannot find utils/wandb.sh at ${WANDB_SWEEP_SH}" >&2
  exit 1
fi

TASKS=()   # Each item: "<abs_path_yaml>|key1=val1,key2=val2"
MAX_PROCS=0
TMP_BASE="/tmp/fmirl_sweeps/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TMP_BASE}"

print_help() {
  cat <<EOF
Usage:
  bash configs/run_parallel.sh --task "<yaml>|<overrides>" [--task "<yaml>|<overrides>"]... [--max-procs N]

Options:
  --task         One task spec per flag. Example:
                 "/home/.../configs/ant/airl.yaml|name=Ant-airl,parameters.seed.value=1"
                 Overrides are optional: "/home/.../configs/ant/airl.yaml"
  --max-procs    Max concurrent processes. Default: number of tasks
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      shift
      [[ $# -gt 0 ]] || { echo "--task requires an argument"; exit 1; }
      TASKS+=("$1")
      shift
      ;;
    --max-procs)
      shift
      [[ $# -gt 0 ]] || { echo "--max-procs requires an integer"; exit 1; }
      MAX_PROCS="$1"
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      print_help
      exit 1
      ;;
  esac
done

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "No tasks provided. See --help."
  exit 1
fi

if [[ "${MAX_PROCS}" -le 0 ]]; then
  MAX_PROCS="${#TASKS[@]}"
fi

run_with_limit() {
  # Wait until number of running jobs < MAX_PROCS
  while [[ "$(jobs -rp | wc -l)" -ge "${MAX_PROCS}" ]]; do
    sleep 1
  done
  "$@" &
}

apply_overrides_and_run() {
  local src_yaml="$1"
  local overrides="$2"  # e.g., "name=Ant-airl,parameters.seed.value=1"
  local idx="$3"

  if [[ ! -f "${src_yaml}" ]]; then
    echo "YAML not found: ${src_yaml}" >&2
    return 1
  fi

  local tmp_yaml="${TMP_BASE}/$(basename "${src_yaml%.*}")_${idx}.yaml"
  cp "${src_yaml}" "${tmp_yaml}"

  if [[ -n "${overrides}" ]]; then
    python3 - "$tmp_yaml" "$overrides" <<'PY'
import sys, os
import yaml

cfg_path = sys.argv[1]
overrides = sys.argv[2]

with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}

def set_by_path(d, path, value):
    keys = path.split('.')
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def smart_set(cfg, key, value):
    # Try deep path if key contains dot or starts with parameters.
    if '.' in key or key.startswith('parameters.'):
        set_by_path(cfg, key, value)
        return
    # Prefer parameters[key].value when key exists there
    params = cfg.get('parameters', {})
    if isinstance(params, dict) and key in params:
        node = params[key] if isinstance(params[key], dict) else {}
        node['value'] = value
        params[key] = node
        cfg['parameters'] = params
        return
    # If top-level key exists, set it
    if key in cfg:
        cfg[key] = value
        return
    # Otherwise create parameters.<key>.value
    if 'parameters' not in cfg or not isinstance(cfg['parameters'], dict):
        cfg['parameters'] = {}
    cfg['parameters'][key] = {'value': value}

pairs = [p for p in overrides.split(',') if p.strip()]
for p in pairs:
    if '=' not in p:
        continue
    k, v = p.split('=', 1)
    k = k.strip()
    v_raw = v.strip()
    try:
        # Try YAML-based casting: numbers, bools, lists, etc.
        v_cast = yaml.safe_load(v_raw)
    except Exception:
        v_cast = v_raw
    smart_set(cfg, k, v_cast)

with open(cfg_path, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  fi

  echo "[RUN] ${src_yaml} (overrides: ${overrides:-none})"
  bash "${WANDB_SWEEP_SH}" "${tmp_yaml}"
}

idx=0
for spec in "${TASKS[@]}"; do
  ((idx+=1))
  yaml_path="${spec}"
  override_str=""
  if [[ "${spec}" == *"|"* ]]; then
    yaml_path="${spec%%|*}"
    override_str="${spec#*|}"
  fi
  # Normalize to absolute path if not already
  if [[ "${yaml_path}" != /* ]]; then
    yaml_path="$(cd "${REPO_ROOT}" && realpath -m "${yaml_path}")"
  fi
  run_with_limit apply_overrides_and_run "${yaml_path}" "${override_str}" "${idx}"
done

wait
echo "All tasks completed."


