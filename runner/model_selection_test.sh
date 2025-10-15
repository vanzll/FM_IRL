#!/bin/bash

# 测试模型选择功能的脚本

echo "🧪 测试模型选择功能"
echo "========================="

# 测试1: 单个模型
echo "📝 测试1: 选择单个模型 (drail)"
./run_all_models.sh ant 0.00 --models drail
echo ""

# 测试2: 多个模型
echo "📝 测试2: 选择多个模型 (drail bc)"
./run_all_models.sh ant 0.00 --models drail bc  
echo ""

# 测试3: 无效模型
echo "📝 测试3: 无效模型名 (应该报错)"
./run_all_models.sh ant 0.00 --models invalid_model
echo ""

# 测试4: 所有模型
echo "📝 测试4: 所有模型 (默认行为)"
./run_all_models.sh ant 0.00
echo ""

echo "✅ 测试完成"