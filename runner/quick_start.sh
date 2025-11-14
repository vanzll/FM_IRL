#!/bin/bash

# DRAIL 快速启动器
# 交互式界面，方便用户快速启动和管理训练

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 支持的环境配置
declare -A ENV_CONFIGS=(
    ["pick"]="FetchPick 噪声级别:自动检测"
    ["push"]="FetchPush 噪声级别和专家转换数:1.00,1.25,1.50,1.75,2.00"
    ["hand"]="HandRotate 噪声级别:1.00,1.25,1.50,1.75,2.00"
    ["ant"]="AntGoal 噪声级别:自动检测"
    ["maze"]="Maze 专家覆盖率:25,50,75,100"
    ["walker"]="Walker 轨迹数量:1traj,2traj,3traj,5traj"
    ["halfcheetah"]="HalfCheetah 轨迹数量:1traj,2traj,3traj,5traj"
    ["hopper"]="Hopper 轨迹数量:1traj,2traj,3traj,5traj"
    ["navigation"]="Navigation 噪声级别:0.00"
    ["sine"]="Sine 无参数"
)

# 支持的模型列表
AVAILABLE_MODELS=("drail" "drail-un" "fm-drail" "fmail" "decoupled-fmail" "fmail_reg1" "gail" "vail" "gailGP" "wail" "bc" "diffusion-policy" "fm-bc" "airl" "giril" "pwil")

# 显示标题
show_header() {
    clear
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    🚀 DRAIL 快速启动器 🚀                    ║"
    echo "║                  多模型并行训练管理工具                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 显示主菜单
show_main_menu() {
    echo -e "${BLUE}📋 选择操作:${NC}"
    echo "  1) 启动新的训练"
    echo "  2) 查看训练状态"
    echo "  3) 停止训练"
    echo "  4) 查看使用帮助"
    echo "  0) 退出"
}

# 显示环境选择菜单
show_env_menu() {
    echo -e "${BLUE}🎯 选择训练环境:${NC}"
    local i=1
    for env in $(echo "${!ENV_CONFIGS[@]}" | tr ' ' '\n' | sort); do
        local desc=$(echo "${ENV_CONFIGS[$env]}" | cut -d' ' -f1)
        local params=$(echo "${ENV_CONFIGS[$env]}" | cut -d':' -f2)
        echo "  $i) $env - $desc"
        ((i++))
    done
    echo "  0) 返回主菜单"
}

# 获取环境的可用参数
get_env_params() {
    local env=$1
    local config_dir="$PROJECT_ROOT/configs/$env"
    
    if [ ! -d "$config_dir" ]; then
        echo ""
        return
    fi
    
    case "$env" in
        "push")
            # push环境特殊处理二层结构
            local params=()
            for noise_level in $(ls "$config_dir" | sort -V); do
                if [ -d "$config_dir/$noise_level" ] && [ "$noise_level" != "expert" ]; then
                    local subdirs=($(ls "$config_dir/$noise_level" 2>/dev/null))
                    if [ ${#subdirs[@]} -gt 0 ] && [ -f "$config_dir/$noise_level/${subdirs[0]}" ] 2>/dev/null; then
                        # 直接包含.yaml文件
                        params+=("$noise_level")
                    else
                        # 有子目录
                        for expert_trans in "${subdirs[@]}"; do
                            if [ -d "$config_dir/$noise_level/$expert_trans" ]; then
                                params+=("${noise_level}/${expert_trans}")
                            fi
                        done
                    fi
                fi
            done
            printf '%s\n' "${params[@]}"
            ;;
        "sine")
            # sine 环境根目录直接是模型yaml，无参数
            if ls "$config_dir"/*.yaml >/dev/null 2>&1; then
                echo "."
            fi
            ;;
        *)
            # 其他环境直接扫描一层
            for param in $(ls "$config_dir" | sort -V); do
                if [ -d "$config_dir/$param" ] && [ "$param" != "expert" ]; then
                    echo "$param"
                fi
            done
            ;;
    esac
}

# 显示参数选择菜单
show_param_menu() {
    local env=$1
    local desc=$(echo "${ENV_CONFIGS[$env]}" | cut -d' ' -f2-)
    
    echo -e "${BLUE}⚙️  选择 $env 环境的参数 ($desc):${NC}"
    
    local params=($(get_env_params "$env"))
    if [ ${#params[@]} -eq 0 ]; then
        echo -e "${RED}❌ 未找到可用的参数配置${NC}"
        return 1
    fi
    
    local i=1
    for param in "${params[@]}"; do
        # 格式化显示参数
        local display_param="$param"
        case "$env" in
            "pick"|"hand")
                display_param="噪声级别: $param"
                ;;
            "push")
                if [[ $param =~ ^([^/]+)/(.+)$ ]]; then
                    display_param="噪声级别: ${BASH_REMATCH[1]}, 专家转换: ${BASH_REMATCH[2]}"
                else
                    display_param="噪声级别: $param"
                fi
                ;;
            "ant")
                display_param="噪声级别: $param"
                ;;
            "maze")
                display_param="专家覆盖率: ${param}%"
                ;;
            "walker"|"halfcheetah"|"hopper")
                display_param="轨迹数量: $param"
                ;;
            "navigation")
                display_param="噪声级别: $param"
                ;;
            "sine")
                display_param="默认配置"
                ;;
        esac
        echo "  $i) $display_param"
        ((i++))
    done
    echo "  a) 运行所有配置"
    echo "  0) 返回环境选择"
    
    # 将参数数组设为全局变量供后续使用
    AVAILABLE_PARAMS=("${params[@]}")
}

# 显示模型选择菜单
show_model_menu() {
    echo -e "${BLUE}🤖 选择要训练的模型:${NC}"
    
    local i=1
    for model in "${AVAILABLE_MODELS[@]}"; do
        # 格式化模型名显示
        local display_name="$model"
        case "$model" in
            "drail")
                display_name="DRAIL (Diffusion Rewards Adversarial Imitation Learning)"
                ;;
            "fmail")
                display_name="FMAIL (Flow Matching AIL)"
                ;;
            "vail")
                display_name="VAIL (Variational Adversarial Imitation Learning)"
                ;;
            "decoupled-fmail")
                display_name="Decoupled FMAIL"
                ;;
            "fmail_reg1")
                display_name="FMAIL (reg=1 variant)"
                ;;
            "drail-un")
                display_name="DRAIL-UN (DRAIL Unnormalized)"
                ;;
            "gail")
                display_name="GAIL (Generative Adversarial Imitation Learning)"
                ;;
            "gailGP")
                display_name="GAIL-GP (GAIL with Gradient Penalty)"
                ;;
            "wail")
                display_name="WAIL (Wasserstein Adversarial Imitation Learning)"
                ;;
            "bc")
                display_name="BC (Behavioral Cloning)"
                ;;
            "diffusion-policy")
                display_name="Diffusion Policy"
                ;;
            "fm-bc")
                display_name="FM-BC (Flow Matching Behavioral Cloning)"
                ;;
            "airl")
                display_name="AIRL (Adversarial IRL)"
                ;;
            "giril")
                display_name="GIRIL (Generative IRL)"
                ;;
            "pwil")
                display_name="PWIL (Preference-based WIL)"
                ;;
            "navigation")
                display_name="Navigation"
                ;;
        esac
        echo "  $i) $display_name"
        ((i++))
    done
    echo "  a) 运行所有模型"
    echo "  m) 多选模型"
    echo "  0) 返回参数选择"
}

# 多选模型功能
select_multiple_models() {
    local selected_models=()
    
    echo -e "${BLUE}🔽 多选模型模式 (输入数字，多个用空格分隔，如: 1 3 5):${NC}"
    echo ""
    
    # 显示模型列表（简化版）
    local i=1
    for model in "${AVAILABLE_MODELS[@]}"; do
        echo "  $i) $model"
        ((i++))
    done
    echo ""
    
    read -p "请输入要选择的模型编号 (空格分隔): " model_numbers
    
    # 解析输入的数字
    for num in $model_numbers; do
        if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#AVAILABLE_MODELS[@]}" ]; then
            selected_models+=("${AVAILABLE_MODELS[$((num-1))]}")
        else
            echo -e "${YELLOW}⚠️  忽略无效输入: $num${NC}"
        fi
    done
    
    if [ ${#selected_models[@]} -eq 0 ]; then
        echo -e "${RED}❌ 未选择任何有效模型${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ 选择的模型: ${selected_models[*]}${NC}"
    
    # 将选择的模型设为全局变量
    SELECTED_MODELS=("${selected_models[@]}")
    return 0
}

# 启动训练
start_training() {
    show_header
    show_env_menu
    
    echo ""
    read -p "请选择环境 (0-$(echo "${!ENV_CONFIGS[@]}" | wc -w)): " env_choice
    
    if [ "$env_choice" = "0" ]; then
        return
    fi
    
    # 获取环境名
    local envs=($(echo "${!ENV_CONFIGS[@]}" | tr ' ' '\n' | sort))
    if [ "$env_choice" -ge 1 ] && [ "$env_choice" -le "${#envs[@]}" ]; then
        local selected_env="${envs[$((env_choice-1))]}"
    else
        echo -e "${RED}❌ 无效选择${NC}"
        read -p "按Enter继续..."
        return
    fi
    
    # 显示参数选择
    show_header
    if ! show_param_menu "$selected_env"; then
        read -p "按Enter继续..."
        return
    fi
    
    echo ""
    read -p "请选择参数 (0-${#AVAILABLE_PARAMS[@]}, a): " param_choice
    
    if [ "$param_choice" = "0" ]; then
        start_training  # 递归回到环境选择
        return
    fi
    
    # 确定选择的参数
    local selected_params=()
    if [ "$param_choice" = "a" ]; then
        # 运行所有配置
        selected_params=("${AVAILABLE_PARAMS[@]}")
    elif [ "$param_choice" -ge 1 ] && [ "$param_choice" -le "${#AVAILABLE_PARAMS[@]}" ]; then
        # 运行特定配置
        selected_params=("${AVAILABLE_PARAMS[$((param_choice-1))]}")
    else
        echo -e "${RED}❌ 无效选择${NC}"
        read -p "按Enter继续..."
        return
    fi
    
    # 显示模型选择
    show_header
    show_model_menu
    
    echo ""
    read -p "请选择模型 (0-${#AVAILABLE_MODELS[@]}, a, m，或用空格分隔的编号如: 1 4 7): " model_choice
    
    if [ "$model_choice" = "0" ]; then
        start_training  # 递归回到环境选择
        return
    fi
    
    # 确定选择的模型
    local selected_models=()
    case "$model_choice" in
        "a")
            # 运行所有模型
            selected_models=("${AVAILABLE_MODELS[@]}")
            ;;
        "m")
            # 多选模型
            if ! select_multiple_models; then
                read -p "按Enter继续..."
                return
            fi
            selected_models=("${SELECTED_MODELS[@]}")
            ;;
        *)
            # 支持空格分隔的多个编号或单个编号
            # 先按空白拆分
            IFS=' ' read -r -a model_nums <<< "$model_choice"
            if [ "${#model_nums[@]}" -gt 1 ]; then
                # 多个编号
                for num in "${model_nums[@]}"; do
                    if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#AVAILABLE_MODELS[@]}" ]; then
                        selected_models+=("${AVAILABLE_MODELS[$((num-1))]}")
                    else
                        echo -e "${YELLOW}⚠️  忽略无效输入: $num${NC}"
                    fi
                done
                if [ ${#selected_models[@]} -eq 0 ]; then
                    echo -e "${RED}❌ 未选择任何有效模型${NC}"
                    read -p "按Enter继续..."
                    return
                fi
            else
                # 单个编号
                if [[ "$model_choice" =~ ^[0-9]+$ ]] && [ "$model_choice" -ge 1 ] && [ "$model_choice" -le "${#AVAILABLE_MODELS[@]}" ]; then
                    selected_models=("${AVAILABLE_MODELS[$((model_choice-1))]}")
                else
                    echo -e "${RED}❌ 无效选择${NC}"
                    read -p "按Enter继续..."
                    return
                fi
            fi
            ;;
    esac
    
    # 显示选择摘要并确认
    echo ""
    echo -e "${CYAN}📋 训练配置摘要:${NC}"
    echo -e "  环境: ${GREEN}$selected_env${NC}"
    echo -e "  参数: ${GREEN}${selected_params[*]}${NC}"
    echo -e "  模型: ${GREEN}${selected_models[*]}${NC}"

    # 选择GPU
    echo ""
    echo -e "${BLUE}🖥️  可用GPU: 0 1 2 3${NC}"
    read -p "可见GPU (例如: 0 或 0,1,2,3; 留空为全部): " selected_gpus
    if [ -n "$selected_gpus" ]; then
        echo -e "  GPU: ${GREEN}${selected_gpus}${NC}"
    else
        echo -e "  GPU: ${GREEN}默认(全部可见)${NC}"
    fi
    echo ""
    
    read -p "确认开始训练? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}⚠️  已取消${NC}"
        read -p "按Enter继续..."
        return
    fi
    
    # 执行训练启动
    echo -e "${BLUE}🚀 开始启动训练...${NC}"
    echo ""
    
    local all_success=true
    
    # 为每个参数配置启动选择的模型
    for param in "${selected_params[@]}"; do
        echo -e "${CYAN}📍 启动配置: $selected_env/$param${NC}"
        
        # 构建启动命令
        local start_cmd="$SCRIPT_DIR/run_all_models.sh $selected_env"
        
        # 处理push环境的特殊格式
        if [ "$selected_env" = "push" ] && [[ $param =~ ^([^/]+)/(.+)$ ]]; then
            start_cmd="$start_cmd ${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
        else
            start_cmd="$start_cmd $param"
        fi
        
        # 添加模型选择参数
        if [ ${#selected_models[@]} -lt ${#AVAILABLE_MODELS[@]} ]; then
            # 只运行选择的模型
            start_cmd="$start_cmd --models ${selected_models[*]}"
        fi

        # 添加GPU参数
        if [ -n "$selected_gpus" ]; then
            start_cmd="$start_cmd --gpus $selected_gpus"
        fi
        
        echo -e "${YELLOW}执行: $start_cmd${NC}"
        
        # 执行启动命令
        if eval "$start_cmd"; then
            echo -e "${GREEN}✅ 配置 $param 启动成功${NC}"
        else
            echo -e "${RED}❌ 配置 $param 启动失败${NC}"
            all_success=false
        fi
        echo ""
    done
    
    # 显示总体结果
    if $all_success; then
        echo -e "${GREEN}🎉 所有配置启动成功!${NC}"
        echo -e "${CYAN}💡 提示: 使用 '2) 查看训练状态' 来监控训练进度${NC}"
    else
        echo -e "${YELLOW}⚠️  部分配置启动失败，请检查错误信息${NC}"
    fi
    
    echo ""
    read -p "按Enter继续..."
}

# 查看状态
view_status() {
    show_header
    echo -e "${BLUE}📊 查看训练状态${NC}"
    echo ""
    
    if ! "$SCRIPT_DIR/status_all_models.sh"; then
        echo -e "${RED}❌ 查看状态失败${NC}"
    fi
    
    echo ""
    read -p "按Enter继续..."
}

# 停止训练
stop_training() {
    show_header
    echo -e "${BLUE}🛑 停止训练${NC}"
    echo ""
    
    # 显示当前运行的会话
    echo -e "${YELLOW}当前运行的DRAIL会话:${NC}"
    local sessions=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -E "^drail-" || true)
    
    if [ -z "$sessions" ]; then
        echo "  无DRAIL会话运行"
        echo ""
        read -p "按Enter继续..."
        return
    fi
    
    echo "$sessions" | sed 's/^/  /'
    echo ""
    
    echo -e "${BLUE}选择停止方式:${NC}"
    echo "  1) 停止所有DRAIL会话"
    echo "  2) 停止特定环境的会话"
    echo "  0) 返回主菜单"
    
    read -p "请选择 (0-2): " stop_choice
    
    case "$stop_choice" in
        1)
            echo ""
            if "$SCRIPT_DIR/stop_all_models.sh"; then
                echo -e "${GREEN}✅ 停止完成${NC}"
            else
                echo -e "${RED}❌ 停止失败${NC}"
            fi
            ;;
        2)
            show_env_menu
            echo ""
            read -p "请选择要停止的环境 (0-$(echo "${!ENV_CONFIGS[@]}" | wc -w)): " env_choice
            
            if [ "$env_choice" = "0" ]; then
                return
            fi
            
            local envs=($(echo "${!ENV_CONFIGS[@]}" | tr ' ' '\n' | sort))
            if [ "$env_choice" -ge 1 ] && [ "$env_choice" -le "${#envs[@]}" ]; then
                local selected_env="${envs[$((env_choice-1))]}"
                echo ""
                if "$SCRIPT_DIR/stop_all_models.sh" "$selected_env"; then
                    echo -e "${GREEN}✅ 停止 $selected_env 环境完成${NC}"
                else
                    echo -e "${RED}❌ 停止失败${NC}"
                fi
            else
                echo -e "${RED}❌ 无效选择${NC}"
            fi
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}❌ 无效选择${NC}"
            ;;
    esac
    
    echo ""
    read -p "按Enter继续..."
}

# 显示帮助
show_help() {
    show_header
    echo -e "${BLUE}📖 使用帮助${NC}"
    echo ""
    
    echo -e "${YELLOW}支持的环境和参数:${NC}"
    for env in $(echo "${!ENV_CONFIGS[@]}" | tr ' ' '\n' | sort); do
        local info="${ENV_CONFIGS[$env]}"
        local desc=$(echo "$info" | cut -d' ' -f1)
        local params=$(echo "$info" | cut -d':' -f2)
        echo "  $env - $desc"
        echo "    参数: $(echo "$params" | tr ',' ' ')"
        echo ""
    done
    
    echo -e "${YELLOW}手动命令示例:${NC}"
    echo "  ./runner/run_all_models.sh pick 1.25"
    echo "  ./runner/run_all_models.sh pick 1.25 --models drail bc"
    echo "  ./runner/run_all_models.sh push 1.50 2000"
    echo "  ./runner/run_all_models.sh walker 5traj --models gail"
    echo "  ./runner/status_all_models.sh"
    echo "  ./runner/stop_all_models.sh pick"
    echo ""
    
    echo -e "${YELLOW}Tmux 快捷键:${NC}"
    echo "  Ctrl+b d     - 从会话中分离"
    echo "  Ctrl+b n     - 下一个窗口"
    echo "  Ctrl+b p     - 上一个窗口"
    echo "  Ctrl+b c     - 创建新窗口"
    echo "  Ctrl+b &     - 关闭当前窗口"
    echo ""
    
    read -p "按Enter继续..."
}

# 主循环
main() {
    # 检查必要工具
    if ! command -v tmux &> /dev/null; then
        echo -e "${RED}❌ 错误: 未安装tmux${NC}"
        echo "请先安装tmux: sudo apt-get install tmux"
        exit 1
    fi
    
    # 检查项目根目录
    if [ ! -d "$PROJECT_ROOT/configs" ]; then
        echo -e "${RED}❌ 错误: 未找到configs目录${NC}"
        echo "请确保在DRAIL项目根目录下运行此脚本"
        exit 1
    fi
    
    while true; do
        show_header
        show_main_menu
        echo ""
        read -p "请选择操作 (0-4): " choice
        
        case "$choice" in
            1)
                start_training
                ;;
            2)
                view_status
                ;;
            3)
                stop_training
                ;;
            4)
                show_help
                ;;
            0)
                echo -e "${CYAN}👋 再见！${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}❌ 无效选择，请重新输入${NC}"
                sleep 1
                ;;
        esac
    done
}

# 启动主程序
main