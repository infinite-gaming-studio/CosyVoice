#!/bin/bash
# CosyVoice Google Colab 快速启动脚本
# 使用方法：在 Colab 单元格中运行：!bash /content/drive/MyDrive/CosyVoice_Colab/colab_quick_start.sh [命令]

# 配置路径
export DRIVE_WORK_DIR='/content/drive/MyDrive/CosyVoice_Colab'
export CONDA_DIR="$DRIVE_WORK_DIR/miniconda3"
export REPO_DIR="$DRIVE_WORK_DIR/CosyVoice"
export MODEL_DIR="$DRIVE_WORK_DIR/models"

# 默认模型
export MODEL_NAME='Fun-CosyVoice3-0.5B-2512'
export MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

echo "=========================================="
echo "  🎙️ CosyVoice Colab 快速启动脚本"
echo "=========================================="
echo "📁 工作目录: $DRIVE_WORK_DIR"
echo "🐍 Conda: $CONDA_DIR"
echo "📦 模型: $MODEL_PATH"
echo ""

# 检查环境
if [ ! -d "$CONDA_DIR" ]; then
    echo "❌ 错误: Miniconda 未安装"
    echo "请先在 Colab 中运行 Notebook 的第 1 节"
    exit 1
fi

if [ ! -d "$CONDA_DIR/envs/cosyvoice" ]; then
    echo "❌ 错误: Conda 环境未创建"
    echo "请先在 Colab 中运行 Notebook 的第 3 节"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型未下载"
    echo "请先在 Colab 中运行 Notebook 的第 4 节"
    exit 1
fi

echo "✅ 所有组件检查通过！"
echo ""

# 获取命令
COMMAND=${1:-help}

case $COMMAND in
    webui|start)
        echo "🚀 启动 Web UI..."
        echo "⏳ 请稍等..."
        cd "$REPO_DIR"
        "$CONDA_DIR/envs/cosyvoice/bin/python" webui.py \
            --port 50000 \
            --model_dir "$MODEL_PATH" \
            --share
        ;;
    
    test)
        echo "🧪 运行快速测试..."
        cat > /tmp/test_colab_quick.py << 'EOF'
import sys
import os
sys.path.insert(0, os.environ['REPO_DIR'])
sys.path.insert(0, os.environ['REPO_DIR'] + '/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

print("⏳ 加载模型...")
model = AutoModel(model_dir=os.environ['MODEL_PATH'])
print(f"✅ 模型加载完成！")

text = "你好，这是 Colab 快速测试。"
instruct = "You are a helpful assistant.<|endofprompt|>"

print("🔊 生成音频...")
for i, result in enumerate(model.inference_instruct2(text, instruct, stream=False)):
    output_path = f"{os.environ['DRIVE_WORK_DIR']}/quick_test_{i}.wav"
    torchaudio.save(output_path, result['tts_speech'], model.sample_rate)
    print(f"💾 已保存: {output_path}")

print("\n✅ 测试完成！")
EOF
        export REPO_DIR MODEL_PATH DRIVE_WORK_DIR
        "$CONDA_DIR/envs/cosyvoice/bin/python" /tmp/test_colab_quick.py
        ;;
    
    status|check)
        echo "📊 环境状态检查"
        echo "=========================================="
        echo "组件状态:"
        [ -d "$CONDA_DIR" ] && echo "  ✅ Miniconda" || echo "  ❌ Miniconda"
        [ -d "$CONDA_DIR/envs/cosyvoice" ] && echo "  ✅ Conda 环境" || echo "  ❌ Conda 环境"
        [ -d "$REPO_DIR" ] && echo "  ✅ 代码仓库" || echo "  ❌ 代码仓库"
        [ -d "$MODEL_PATH" ] && echo "  ✅ 模型文件" || echo "  ❌ 模型文件"
        
        echo ""
        echo "💾 存储使用:"
        if [ -d "$DRIVE_WORK_DIR" ]; then
            du -sh "$DRIVE_WORK_DIR"/* 2>/dev/null | sort -hr | head -10
            echo ""
            echo "总计: $(du -sh "$DRIVE_WORK_DIR" 2>/dev/null | cut -f1)"
        fi
        
        echo ""
        echo "🐍 Python 版本:"
        "$CONDA_DIR/envs/cosyvoice/bin/python" --version
        
        echo ""
        echo "🔥 PyTorch/CUDA:"
        "$CONDA_DIR/envs/cosyvoice/bin/python" -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
        ;;
    
    clean)
        echo "🧹 清理缓存..."
        echo "这将删除:"
        echo "  - Conda 包缓存"
        echo "  - Git 历史记录"
        read -p "确认? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$CONDA_DIR/pkgs/*"
            rm -rf "$REPO_DIR/.git"
            echo "✅ 清理完成"
        else
            echo "❌ 取消"
        fi
        ;;
    
    export-env)
        echo "📤 导出环境配置..."
        "$CONDA_DIR/bin/conda" env export -n cosyvoice > "$DRIVE_WORK_DIR/cosyvoice_environment.yml"
        echo "✅ 已保存到: $DRIVE_WORK_DIR/cosyvoice_environment.yml"
        ;;
    
    help|*)
        echo "使用方法: !bash colab_quick_start.sh [命令]"
        echo ""
        echo "可用命令:"
        echo "  webui, start   - 启动 Web UI（默认）"
        echo "  test           - 运行快速测试"
        echo "  status, check  - 查看环境状态"
        echo "  clean          - 清理缓存文件"
        echo "  export-env     - 导出环境配置"
        echo "  help           - 显示帮助"
        echo ""
        echo "示例:"
        echo "  !bash /content/drive/MyDrive/CosyVoice_Colab/colab_quick_start.sh webui"
        echo "  !bash /content/drive/MyDrive/CosyVoice_Colab/colab_quick_start.sh status"
        ;;
esac
