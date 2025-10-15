#!/bin/bash

# DRAIL 停止所有模型训练脚本
# 使用tmux杀死所有相关会话

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 显示使用方法
show_usage() {
    echo -e "${BLUE}DRAIL 停止训练脚本${NC}"
    echo -e "${YELLOW}用法:${NC}"
    echo "  $0 [环境名] [参数值]"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  $0                        # 停止所有drail相关的tmux会话"
    echo "  $0 pick                   # 停止pick环境的所有会话"
    echo "  $0 pick 125               # 停止pick环境噪声级别1.25的会话"
    echo "  $0 push 150-2000          # 停止push环境特定配置的会话"
    echo "  $0 navigation 0.00        # 停止navigation环境噪声级别0.00的会话"
    echo "  $0 sine                    # 停止sine环境的所有会话"
    echo ""
}

ENV_NAME=$1
PARAM_VALUE=$2

# 构建会话模式
if [ -n "$ENV_NAME" ] && [ -n "$PARAM_VALUE" ]; then
    SESSION_PATTERN="drail-${ENV_NAME}-${PARAM_VALUE}"
    echo -e "${BLUE}停止特定会话: ${SESSION_PATTERN}${NC}"
elif [ -n "$ENV_NAME" ]; then
    SESSION_PATTERN="drail-${ENV_NAME}-*"
    echo -e "${BLUE}停止环境 ${ENV_NAME} 的所有会话${NC}"
else
    SESSION_PATTERN="drail-*"
    echo -e "${BLUE}停止所有DRAIL会话${NC}"
fi

# 获取匹配的会话列表
if [ -n "$ENV_NAME" ] && [ -n "$PARAM_VALUE" ]; then
    # 精确匹配特定会话
    MATCHING_SESSIONS=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -E "^${SESSION_PATTERN}$" || true)
elif [ -n "$ENV_NAME" ]; then
    # 匹配特定环境的所有会话
    MATCHING_SESSIONS=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -E "^drail-${ENV_NAME}-" || true)
else
    # 匹配所有DRAIL会话
    MATCHING_SESSIONS=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -E "^drail-" || true)
fi

if [ -z "$MATCHING_SESSIONS" ]; then
    echo -e "${YELLOW}未找到匹配的tmux会话${NC}"
    echo -e "${BLUE}当前所有会话:${NC}"
    tmux list-sessions 2>/dev/null || echo "  无会话运行"
    exit 0
fi

echo -e "${YELLOW}找到以下匹配的会话:${NC}"
echo "$MATCHING_SESSIONS" | sed 's/^/  /'
echo ""

# 确认停止
read -p "确定要停止这些会话吗? (y/N): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}取消操作${NC}"
    exit 0
fi

# 停止会话
STOPPED_COUNT=0
FAILED_COUNT=0

while IFS= read -r session; do
    if [ -n "$session" ]; then
        echo -e "${BLUE}正在停止会话: $session${NC}"
        if tmux kill-session -t "$session" 2>/dev/null; then
            echo -e "${GREEN}✅ 已停止: $session${NC}"
            ((STOPPED_COUNT++))
        else
            echo -e "${RED}❌ 停止失败: $session${NC}"
            ((FAILED_COUNT++))
        fi
    fi
done <<< "$MATCHING_SESSIONS"

echo ""
echo -e "${GREEN}🎉 操作完成!${NC}"
echo -e "${GREEN}✅ 已停止: ${STOPPED_COUNT} 个会话${NC}"
if [ $FAILED_COUNT -gt 0 ]; then
    echo -e "${RED}❌ 停止失败: ${FAILED_COUNT} 个会话${NC}"
fi

# 显示剩余会话
echo ""
echo -e "${BLUE}剩余的tmux会话:${NC}"
tmux list-sessions 2>/dev/null || echo "  无会话运行"