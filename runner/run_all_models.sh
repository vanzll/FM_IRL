#!/bin/bash

# DRAIL 多模型并行训练脚本
# 使用tmux为每个模型创建独立会话

# set -e  # 暂时注释，避免tmux命令错误导致脚本退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="Genbot"
WANDB_SCRIPT="$PROJECT_ROOT/utils/wandb.sh"
GPU_IDS=""

# 模型列表
# 新增 FM-BC 与 FMAIL
MODELS=("drail" "drail-un" "fm-drail" "fmail" "decoupled-fmail" "fmail_reg1" "gail" "vail" "gailGP" "wail" "bc" "diffusion-policy" "fm-bc" "airl" "giril" "pwil")

# 显示使用方法
show_usage() {
    echo -e "${CYAN}DRAIL 多模型并行训练脚本${NC}"
    echo -e "${YELLOW}用法:${NC}"
    echo "  $0 <环境名> [参数值] [子参数值] [--models <模型1> <模型2> ...] [--gpus <ids>]"
    echo ""
    echo -e "${YELLOW}参数说明:${NC}"
    echo "  --models <模型列表>    - 指定要运行的模型，空格分隔 (默认: 所有模型)"
    echo "  --gpus <ids>            - 指定可见GPU，如: 0 或 0,1,2,3 (默认: 全部)"
    echo ""
    echo -e "${YELLOW}支持的环境和参数含义:${NC}"
    echo "  pick <noise_level>     - FetchPick环境，噪声级别 (1.00, 1.25, 1.50, 1.75, 2.00)"
    echo "  push <noise_level> [expert_transitions] - FetchPush环境，噪声级别和专家转换数"
    echo "  hand <noise_level>     - HandRotate环境，噪声级别 (1.00, 1.25, 1.50, 1.75, 2.00)"
    echo "  ant <noise_level>      - AntGoal环境，噪声级别 (从目录动态检测)"
    echo "  maze <coverage>        - Maze环境，专家覆盖率 (25, 50, 75, 100)"
    echo "  walker <trajectories>  - Walker环境，轨迹数量 (1traj, 2traj, 3traj, 5traj)"
    echo "  halfcheetah <trajectories>  - HalfCheetah环境，轨迹数量 (1traj, 2traj, 3traj, 5traj)"
    echo "  hopper <trajectories>  - Hopper环境，轨迹数量 (1traj, 2traj, 3traj, 5traj)"
    echo "  navigation <noise_level> - Navigation环境，噪声级别 (0.00)"
    echo "  sine                     - Sine环境，无参数（根目录下直接是各模型yaml）"
    echo ""
    echo -e "${YELLOW}基本示例:${NC}"
    echo "  $0 pick 1.25              # 运行pick环境，噪声级别1.25的所有模型"
    echo "  $0 maze 75                # 运行maze环境，专家覆盖率75%的所有模型"
    echo "  $0 pick                   # 运行pick环境的所有参数配置"
    echo ""
    echo -e "${YELLOW}模型选择示例:${NC}"
    echo "  $0 ant 0.00 --models drail bc         # 只运行drail和bc模型"
    echo "  $0 pick 1.25 --models gail            # 只运行gail模型"
    echo "  $0 maze 75 --models drail gail wail   # 运行指定的3个模型"
    echo ""
    echo -e "${YELLOW}支持的模型:${NC}"
    printf "  %s\n" "${MODELS[@]}"
}

# 解析参数
SELECTED_MODELS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            # 读取模型列表直到下一个参数或结束
            while [[ $# -gt 0 ]] && [[ $1 != --* ]]; do
                SELECTED_MODELS+=("$1")
                shift
            done
            ;;
        --gpus)
            shift
            if [[ $# -gt 0 ]]; then
                GPU_IDS=$1
                shift
            else
                echo -e "${RED}错误: --gpus 需要提供形如 0 或 0,1,2,3 的参数${NC}"
                exit 1
            fi
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            # 环境名和参数值
            if [ -z "$ENV_NAME" ]; then
                ENV_NAME=$1
            elif [ -z "$PARAM_VALUE" ]; then
                PARAM_VALUE=$1
            elif [ -z "$SUB_PARAM_VALUE" ]; then
                SUB_PARAM_VALUE=$1
            fi
            shift
            ;;
    esac
done

# 检查必需参数
if [ -z "$ENV_NAME" ]; then
    echo -e "${RED}错误: 缺少环境名参数${NC}"
    show_usage
    exit 1
fi

# 如果没有指定模型，使用所有模型
if [ ${#SELECTED_MODELS[@]} -eq 0 ]; then
    SELECTED_MODELS=("${MODELS[@]}")
fi

# 验证选择的模型是否有效
for selected_model in "${SELECTED_MODELS[@]}"; do
    if [[ ! " ${MODELS[@]} " =~ " ${selected_model} " ]]; then
        echo -e "${RED}错误: 不支持的模型名 '${selected_model}'${NC}"
        echo -e "${YELLOW}支持的模型: ${MODELS[*]}${NC}"
        exit 1
    fi
done

# 验证环境名
VALID_ENVS=("pick" "push" "hand" "ant" "maze" "walker" "halfcheetah" "hopper" "navigation" "sine")
if [[ ! " ${VALID_ENVS[@]} " =~ " ${ENV_NAME} " ]]; then
    echo -e "${RED}错误: 不支持的环境名 '${ENV_NAME}'${NC}"
    show_usage
    exit 1
fi

# 检查configs目录
CONFIGS_DIR="$PROJECT_ROOT/configs/${ENV_NAME}"
if [ ! -d "$CONFIGS_DIR" ]; then
    echo -e "${RED}错误: 配置目录不存在: $CONFIGS_DIR${NC}"
    exit 1
fi

# 根据环境获取参数类型描述
get_param_description() {
    case "$ENV_NAME" in
        "pick"|"hand") echo "噪声级别" ;;
        "push") echo "噪声级别和专家转换数" ;;
        "ant") echo "噪声级别" ;;
        "maze") echo "专家覆盖率" ;;
        "walker"|"halfcheetah"|"hopper") echo "轨迹数量" ;;
        "navigation") echo "噪声级别" ;;
        "sine") echo "无参数" ;;
    esac
}

# 获取参数值的可用选项
get_available_params() {
    case "$ENV_NAME" in
        "pick"|"hand")
            if [ -d "$CONFIGS_DIR" ]; then
                local opts=()
                for d in $(ls "$CONFIGS_DIR" | sort -V); do
                    if [ -d "$CONFIGS_DIR/$d" ] && [ "$d" != "expert" ]; then
                        opts+=("$d")
                    fi
                done
                printf '%s ' "${opts[@]}"
                echo ""
            fi
            ;;
        "push")
            if [ -n "$PARAM_VALUE" ] && [ -d "$CONFIGS_DIR/$PARAM_VALUE" ]; then
                # 检查是否有子目录
                local subdirs=($(ls "$CONFIGS_DIR/$PARAM_VALUE" 2>/dev/null))
                if [ -f "$CONFIGS_DIR/$PARAM_VALUE/${subdirs[0]}" ] 2>/dev/null; then
                    echo "$PARAM_VALUE"
                else
                    for subdir in "${subdirs[@]}"; do
                        if [ -d "$CONFIGS_DIR/$PARAM_VALUE/$subdir" ]; then
                            echo "$PARAM_VALUE/$subdir"
                        fi
                    done
                fi
            else
                echo "1.00 1.25 1.50 1.75 2.00"
            fi
            ;;
        "ant")
            if [ -d "$CONFIGS_DIR" ]; then
                local opts=()
                for d in $(ls "$CONFIGS_DIR" | sort -V); do
                    if [ -d "$CONFIGS_DIR/$d" ] && [ "$d" != "expert" ]; then
                        opts+=("$d")
                    fi
                done
                printf '%s ' "${opts[@]}"
                echo ""
            fi
            ;;
        "maze")
            echo "25 50 75 100"
            ;;
        "walker"|"halfcheetah"|"hopper")
            echo "1traj 2traj 3traj 5traj"
            ;;
        "navigation")
            echo "0.00"
            ;;
        "sine")
            echo "-"
            ;;
    esac
}

# 获取实际的配置路径列表
get_config_paths() {
    local configs=()
    
    if [ -n "$PARAM_VALUE" ]; then
        if [ "$ENV_NAME" = "push" ] && [ -n "$SUB_PARAM_VALUE" ]; then
            # push环境的特殊二层结构
            configs+=("${PARAM_VALUE}/${SUB_PARAM_VALUE}")
        else
            configs+=("$PARAM_VALUE")
        fi
    else
        # 自动扫描所有可用的配置
        if [ "$ENV_NAME" = "push" ]; then
            # push环境需要特殊处理二层结构
            for noise_level in $(ls "$CONFIGS_DIR" | sort -V); do
                if [ -d "$CONFIGS_DIR/$noise_level" ] && [ "$noise_level" != "expert" ]; then
                    # 检查是否有子目录（expert transitions）
                    local subdirs=($(ls "$CONFIGS_DIR/$noise_level" 2>/dev/null))
                    if [ ${#subdirs[@]} -gt 0 ] && [ -f "$CONFIGS_DIR/$noise_level/${subdirs[0]}" ] 2>/dev/null; then
                        # 直接包含.yaml文件，没有子目录
                        configs+=("$noise_level")
                    else
                        # 有子目录，遍历expert transitions
                        for expert_trans in "${subdirs[@]}"; do
                            if [ -d "$CONFIGS_DIR/$noise_level/$expert_trans" ]; then
                                configs+=("${noise_level}/${expert_trans}")
                            fi
                        done
                    fi
                fi
            done
        else
            # 其他环境直接扫描一层目录
            for config in $(ls "$CONFIGS_DIR" | sort -V); do
                if [ -d "$CONFIGS_DIR/$config" ] && [ "$config" != "expert" ]; then
                    configs+=("$config")
                fi
            done
            # 若未发现任何子目录但根目录存在模型yaml，则使用根目录（适配如 sine）
            if [ ${#configs[@]} -eq 0 ]; then
                if ls "$CONFIGS_DIR"/*.yaml >/dev/null 2>&1; then
                    configs+=(".")
                fi
            fi
        fi
    fi
    
    printf '%s\n' "${configs[@]}"
}

# 清理参数值用于tmux会话名
clean_param_for_session() {
    local param=$1
    # 根目录或空参数作为 root
    if [ -z "$param" ] || [ "$param" = "." ]; then
        echo "root"
        return
    fi
    # 移除小数点和斜杠，例如 1.25 -> 125, 1.50/2000 -> 150-2000, 5traj -> 5traj
    echo "${param//./}" | sed 's/\//-/g'
}

# 启动单个模型
start_model() {
    local env_name=$1
    local param_config=$2
    local model=$3
    
    # 允许模型名与实际yaml文件名不同的映射（并支持候选名依次尝试）
    local yaml_candidates=("$model")
    case "$model" in
        "fmail")
            yaml_candidates=("fmail" "fmirl")
            ;;
        "fmirl")
            yaml_candidates=("fmirl" "fmail")
            ;;
        "fm-bc")
            yaml_candidates=("fm_policy" "fm-bc")
            ;;
    esac
    
    local config_path
    # 依次尝试候选文件名，找到第一个存在的
    for cand in "${yaml_candidates[@]}"; do
        if [ -z "$param_config" ] || [ "$param_config" = "." ]; then
            config_path="$CONFIGS_DIR/${cand}.yaml"
        else
            config_path="$CONFIGS_DIR/${param_config}/${cand}.yaml"
        fi
        if [ -f "$config_path" ]; then
            break
        fi
    done
    
    # 检查配置文件是否存在
    if [ ! -f "$config_path" ]; then
        echo -e "${YELLOW}⚠️  跳过: 配置文件不存在 $config_path${NC}"
        return 1
    fi
    
    # 清理参数值用于会话名
    local clean_param=$(clean_param_for_session "$param_config")
    
    # 创建tmux会话名
    local session_name="drail-${env_name}-${clean_param}"
    local window_name="${model}"
    
    echo -e "${BLUE}🚀 启动模型: ${model} (环境: ${env_name}, 配置: ${param_config})${NC}"
    
    # 检查会话是否已存在
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo -e "${YELLOW}📱 会话已存在，添加新窗口: $session_name${NC}"
        tmux new-window -t "$session_name" -n "$window_name" -c "$PROJECT_ROOT"
    else
        echo -e "${GREEN}🆕 创建新会话: $session_name${NC}"
        tmux new-session -d -s "$session_name" -n "$window_name" -c "$PROJECT_ROOT"
    fi
    
    # 等待窗口创建完成
    sleep 0.2
    
    # 在窗口中运行命令
    tmux send-keys -t "$session_name:$window_name" "cd $PROJECT_ROOT" Enter
    tmux send-keys -t "$session_name:$window_name" "source /mnt/data/wanzl/conda/bin/activate $CONDA_ENV" Enter
    if [ -n "$GPU_IDS" ]; then
        tmux send-keys -t "$session_name:$window_name" "export CUDA_VISIBLE_DEVICES=$GPU_IDS" Enter
    fi
    tmux send-keys -t "$session_name:$window_name" "echo '🚀 启动模型: $model'" Enter
    tmux send-keys -t "$session_name:$window_name" "echo '📝 配置文件: $config_path'" Enter
    tmux send-keys -t "$session_name:$window_name" "echo '⏰ 开始时间: \$(date)'" Enter
    if [ -n "$GPU_IDS" ]; then
        tmux send-keys -t "$session_name:$window_name" "echo '🎛  使用GPU: $GPU_IDS'" Enter
    fi
    tmux send-keys -t "$session_name:$window_name" "$WANDB_SCRIPT $config_path" Enter
    
    echo -e "${GREEN}✅ 已启动: $session_name:$window_name${NC}"
    return 0
}

# 主函数
main() {
    echo -e "${CYAN}🚀 DRAIL 多模型并行训练启动器${NC}"
    echo -e "${BLUE}环境: ${ENV_NAME} ($(get_param_description))${NC}"
    echo -e "${BLUE}选择的模型: ${SELECTED_MODELS[*]}${NC}"
    if [ -n "$GPU_IDS" ]; then
        echo -e "${BLUE}GPU: CUDA_VISIBLE_DEVICES=${GPU_IDS}${NC}"
    else
        echo -e "${BLUE}GPU: 默认(全部可见)${NC}"
    fi
    
    # 获取配置列表
    CONFIG_PATHS=($(get_config_paths))
    
    if [ ${#CONFIG_PATHS[@]} -eq 0 ]; then
        echo -e "${RED}错误: 未找到可用的配置${NC}"
        echo -e "${YELLOW}可用的参数选项: $(get_available_params)${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}配置列表:${NC}"
    printf "  %s\n" "${CONFIG_PATHS[@]}"
    echo ""
    
    # 检查必要工具
    if ! command -v tmux &> /dev/null; then
        echo -e "${RED}错误: 未安装tmux${NC}"
        exit 1
    fi
    
    if [ ! -f "$WANDB_SCRIPT" ]; then
        echo -e "${RED}错误: wandb脚本不存在: $WANDB_SCRIPT${NC}"
        exit 1
    fi
    
    # 总计数器
    local total_started=0
    local total_skipped=0
    
    # 遍历每个配置
    for param_config in "${CONFIG_PATHS[@]}"; do
        echo -e "${YELLOW}📂 处理配置: ${param_config}${NC}"
        
        # 遍历选择的模型
        for model in "${SELECTED_MODELS[@]}"; do
            if start_model "$ENV_NAME" "$param_config" "$model"; then
                ((total_started++))
            else
                ((total_skipped++))
            fi
        done
        
        echo ""
    done
    
    echo -e "${GREEN}🎉 启动完成!${NC}"
    echo -e "${GREEN}✅ 已启动: ${total_started} 个模型${NC}"
    echo -e "${YELLOW}⚠️  已跳过: ${total_skipped} 个模型${NC}"
    echo ""
    echo -e "${BLUE}查看运行状态:${NC}"
    echo "  tmux ls"
    echo ""
    echo -e "${BLUE}连接到特定会话:${NC}"
    echo "  tmux attach-session -t <会话名>"
    echo ""
    echo -e "${BLUE}停止所有会话:${NC}"
    echo "  ./runner/stop_all_models.sh $ENV_NAME"
}

# 执行主函数
main