#!/bin/bash

# DRAIL 使用示例脚本
# 演示如何使用多模型并行训练工具

echo "🎯 DRAIL 多模型并行训练工具 - 使用示例"
echo ""

echo "📁 当前目录结构:"
ls -la

echo ""
echo "🚀 示例1: 查看帮助信息"
echo "命令: ./run_all_models.sh"
./run_all_models.sh
echo ""

echo "🚀 示例2: 查看当前训练状态"
echo "命令: ./status_all_models.sh"
./status_all_models.sh
echo ""

echo "🚀 示例3: 检查配置文件"
echo "检查 pick 环境 1.25 噪声比例的配置文件:"
CONFIG_DIR="../configs/pick/1.25"
if [ -d "$CONFIG_DIR" ]; then
    echo "✓ 配置目录存在: $CONFIG_DIR"
    echo "可用的模型配置:"
    ls -1 "$CONFIG_DIR"/*.yaml | xargs -I {} basename {} .yaml
else
    echo "❌ 配置目录不存在: $CONFIG_DIR"
fi
echo ""

echo "🚀 示例4: 启动命令示例"
echo "以下是一些启动命令示例 (仅显示，不实际执行):"
echo ""
echo "# 启动 pick 环境 1.25 噪声比例下的所有模型:"
echo "./run_all_models.sh pick 1.25"
echo ""
echo "# 只启动 drail 和 gail 模型:"
echo "./run_all_models.sh pick 1.25 drail gail"
echo ""
echo "# 启动 sine 环境的单个模型:"
echo "./run_all_models.sh sine 1.00 drail"
echo ""

echo "💡 提示:"
echo "- 使用 ./quick_start.sh 获得交互式体验"
echo "- 实际启动训练前，请确保有足够的GPU和内存资源"
echo "- 每个模型会创建单独的tmux窗口进行训练"