#!/bin/bash
# CosyVoice Kaggle 快速启动脚本
# 使用方法：source kaggle_quick_start.sh

# 配置路径
export WORK_DIR='/kaggle/working/cosyvoice_env'
export CONDA_DIR="$WORK_DIR/miniconda3"
export REPO_DIR="$WORK_DIR/CosyVoice"
export MODEL_DIR="$WORK_DIR/models"

# 默认模型
export MODEL_NAME='Fun-CosyVoice3-0.5B-2512'
export MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

echo "=========================================="
echo "  CosyVoice Kaggle 快速启动脚本"
echo "=========================================="
echo "工作目录: $WORK_DIR"
echo "Conda: $CONDA_DIR"
echo "模型: $MODEL_PATH"
echo ""

# 检查环境是否存在
if [ ! -d "$CONDA_DIR" ]; then
    echo "❌ 错误: Miniconda 未安装"
    echo "请先运行 Notebook 的第 0-4 节进行首次部署"
    return 1
fi

if [ ! -d "$CONDA_DIR/envs/cosyvoice" ]; then
    echo "❌ 错误: Conda 环境未创建"
    echo "请先运行 Notebook 的第 3 节创建环境"
    return 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型未下载"
    echo "请先运行 Notebook 的第 4 节下载模型"
    return 1
fi

echo "✅ 环境检查通过！"
echo ""

# 函数：启动 Web UI
start_webui() {
    echo "🚀 启动 Web UI..."
    cd "$REPO_DIR"
    "$CONDA_DIR/envs/cosyvoice/bin/python" webui.py \
        --port 50000 \
        --model_dir "$MODEL_PATH" \
        --share
}

# 函数：运行测试
run_test() {
    echo "🧪 运行测试..."
    cat > /tmp/test_quick.py << 'EOF'
import sys
sys.path.insert(0, "$WORK_DIR/CosyVoice")
sys.path.insert(0, "$WORK_DIR/CosyVoice/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

print("加载模型...")
model = AutoModel(model_dir=os.environ['MODEL_PATH'])

text = "你好，这是快速测试。"
print("生成音频...")
for i, result in enumerate(model.inference_instruct2(text, "You are a helpful assistant.<|endofprompt|>", stream=False)):
    output_path = f"$WORK_DIR/quick_test_{i}.wav"
    torchaudio.save(output_path, result['tts_speech'], model.sample_rate)
    print(f"已保存: {output_path}")

print("测试完成！")
EOF
    "$CONDA_DIR/envs/cosyvoice/bin/python" /tmp/test_quick.py
}

# 函数：查看状态
status() {
    echo "📊 环境状态:"
    echo "  Conda: $(test -d $CONDA_DIR && echo '✅ 已安装' || echo '❌ 未安装')"
    echo "  环境: $(test -d $CONDA_DIR/envs/cosyvoice && echo '✅ 已创建' || echo '❌ 未创建')"
    echo "  代码: $(test -d $REPO_DIR && echo '✅ 已克隆' || echo '❌ 未克隆')"
    echo "  模型: $(test -d $MODEL_PATH && echo '✅ 已下载' || echo '❌ 未下载')"
    echo ""
    echo "💾 磁盘使用:"
    du -sh $WORK_DIR/* 2>/dev/null | head -10
}

# 函数：清理缓存
clean() {
    echo "🧹 清理缓存..."
    rm -rf "$CONDA_DIR/pkgs/*"
    rm -rf "$REPO_DIR/.git"
    echo "✅ 清理完成"
}

# 显示帮助
help() {
    echo "可用命令:"
    echo "  start_webui  - 启动 Web UI"
    echo "  run_test     - 运行快速测试"
    echo "  status       - 查看环境状态"
    echo "  clean        - 清理缓存文件"
    echo "  help         - 显示帮助"
    echo ""
    echo "示例:"
    echo "  source kaggle_quick_start.sh"
    echo "  start_webui"
}

echo "可用命令: start_webui | run_test | status | clean | help"
echo ""
help
