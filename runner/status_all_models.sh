#!/bin/bash

# DRAIL 查看所有模型训练状态脚本

# set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 显示使用方法
show_usage() {
    echo -e "${CYAN}DRAIL 训练状态查看脚本${NC}"
    echo -e "${YELLOW}用法:${NC}"
    echo "  $0 [环境名]"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  $0                        # 查看所有DRAIL会话状态"
    echo "  $0 pick                   # 查看pick环境的会话状态"
    echo "  $0 maze                   # 查看maze环境的会话状态"
    echo "  $0 sine                   # 查看sine环境的会话状态"
    echo ""
}

ENV_NAME=$1

# 构建会话模式
if [ -n "$ENV_NAME" ]; then
    SESSION_PATTERN="drail-${ENV_NAME}-*"
    echo -e "${BLUE}查看环境 ${ENV_NAME} 的训练状态${NC}"
else
    SESSION_PATTERN="drail-*"
    echo -e "${BLUE}查看所有DRAIL训练状态${NC}"
fi

echo ""

# 获取所有会话
ALL_SESSIONS=$(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)

if [ -z "$ALL_SESSIONS" ]; then
    echo -e "${YELLOW}未找到任何tmux会话${NC}"
    exit 0
fi

# 过滤DRAIL相关会话
DRAIL_SESSIONS=$(echo "$ALL_SESSIONS" | grep -E "^drail-" | grep -E "${SESSION_PATTERN//\*/.*}" || true)

if [ -z "$DRAIL_SESSIONS" ]; then
    echo -e "${YELLOW}未找到匹配的DRAIL会话${NC}"
    echo -e "${BLUE}所有现有会话:${NC}"
    echo "$ALL_SESSIONS" | sed 's/^/  /'
    exit 0
fi

echo -e "${GREEN}找到 $(echo "$DRAIL_SESSIONS" | wc -l) 个DRAIL会话:${NC}"
echo ""

# 详细显示每个会话
SESSION_COUNT=0
while IFS= read -r session; do
    if [ -n "$session" ]; then
        ((SESSION_COUNT++))
        
        # 解析会话名称信息
        if [[ $session =~ ^drail-([^-]+)-(.+)$ ]]; then
            env_name="${BASH_REMATCH[1]}"
            param_info="${BASH_REMATCH[2]}"
            
            # 根据环境类型解释参数
            case "$env_name" in
                "pick"|"hand")
                    # 噪声级别，需要重新加小数点，格式如125 -> 1.25
                    if [[ $param_info =~ ^([0-9]{3})$ ]]; then
                        noise_level="${param_info:0:1}.${param_info:1:2}"
                        param_desc="噪声级别: $noise_level"
                    else
                        param_desc="参数: $param_info"
                    fi
                    ;;
                "push")
                    # 可能是噪声级别或噪声级别-专家转换数
                    if [[ $param_info =~ ^([0-9]{3})-([0-9]+)$ ]]; then
                        noise_part="${BASH_REMATCH[1]}"
                        expert_part="${BASH_REMATCH[2]}"
                        noise_level="${noise_part:0:1}.${noise_part:1:2}"
                        param_desc="噪声级别: $noise_level, 专家转换: $expert_part"
                    elif [[ $param_info =~ ^([0-9]{3})$ ]]; then
                        noise_level="${param_info:0:1}.${param_info:1:2}"
                        param_desc="噪声级别: $noise_level"
                    else
                        param_desc="参数: $param_info"
                    fi
                    ;;
                "ant")
                    # 噪声级别，格式如 000, 001, 003, 005
                    case "$param_info" in
                        "000") param_desc="噪声级别: 0.00" ;;
                        "001") param_desc="噪声级别: 0.01" ;;
                        "003") param_desc="噪声级别: 0.03" ;;
                        "005") param_desc="噪声级别: 0.05" ;;
                        *) param_desc="参数: $param_info" ;;
                    esac
                    ;;
                "maze")
                    param_desc="专家覆盖率: ${param_info}%"
                    ;;
                "walker"|"halfcheetah"|"hopper")
                    param_desc="轨迹数量: $param_info"
                    ;;
                "navigation")
                    param_desc="噪声级别: $param_info"
                    ;;
                *)
                    param_desc="参数: $param_info"
                    ;;
            esac
        else
            env_name="未知"
            param_desc="参数: 未解析"
        fi
        
        echo -e "${PURPLE}[$SESSION_COUNT] 会话: ${session}${NC}"
        echo -e "    ${BLUE}环境: ${env_name}${NC}"
        echo -e "    ${BLUE}${param_desc}${NC}"
        
        # 获取会话窗口信息
        WINDOWS=$(tmux list-windows -t "$session" -F '#{window_name}' 2>/dev/null || echo "")
        if [ -n "$WINDOWS" ]; then
            WINDOW_COUNT=$(echo "$WINDOWS" | wc -l)
            echo -e "    ${CYAN}窗口数量: ${WINDOW_COUNT}${NC}"
            echo -e "    ${CYAN}运行模型: $(echo "$WINDOWS" | tr '\n' ' ')${NC}"
        else
            echo -e "    ${YELLOW}无法获取窗口信息${NC}"
        fi
        
        # 获取会话创建时间等信息  
        SESSION_INFO=$(tmux list-sessions -F '#{session_name} #{session_created} #{session_windows}' 2>/dev/null | grep "^$session " || echo "")
        if [ -n "$SESSION_INFO" ]; then
            CREATED_TIME=$(echo "$SESSION_INFO" | awk '{print $2}')
            WINDOW_COUNT=$(echo "$SESSION_INFO" | awk '{print $3}')
            if [ -n "$CREATED_TIME" ] && [ "$CREATED_TIME" != "0" ]; then
                CREATED_DATE=$(date -d "@$CREATED_TIME" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "未知")
                echo -e "    ${CYAN}创建时间: ${CREATED_DATE}${NC}"
            fi
        fi
        
        echo ""
    fi
done <<< "$DRAIL_SESSIONS"

echo -e "${GREEN}🎉 状态查看完成!${NC}"
echo ""
echo -e "${YELLOW}有用的命令:${NC}"
echo "  tmux attach -t <会话名>           # 连接到特定会话"
echo "  tmux list-windows -t <会话名>     # 查看会话的所有窗口"
echo "  ./runner/stop_all_models.sh       # 停止所有会话"
echo "  ./runner/stop_all_models.sh <env> # 停止特定环境的会话"