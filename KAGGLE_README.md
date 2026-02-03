# CosyVoice Kaggle 部署指南

本指南帮助您在 Kaggle 上运行 CosyVoice (Fun-CosyVoice3-0.5B) TTS 模型。

## 🚀 快速开始

### 方法 1: 使用 Kaggle Notebook 文件 (推荐)

1. **上传 Notebook**
   - 打开 [Kaggle](https://www.kaggle.com)
   - 创建一个新的 Notebook
   - 点击 File → Import Notebook
   - 上传 `cosyvoice_kaggle.ipynb` 文件

2. **配置环境**
   - 在 Notebook 右侧设置面板中:
     - **Accelerator**: 选择 GPU (T4 x2 或 P100)
     - **Internet**: 开启

3. **运行 Notebook**
   - 依次运行每个代码单元格
   - 等待模型下载完成（约需 10-15 分钟）
   - 启动 Web UI 后会显示公共访问链接

### 方法 2: 手动创建 Notebook

1. 创建新的 Kaggle Notebook
2. 添加以下代码单元格:

```python
# 1. 克隆仓库
!git clone --recursive https://github.com/infinite-gaming-studio/CosyVoice.git
%cd CosyVoice
```

```python
# 2. 安装依赖
!apt-get update -qq && apt-get install -y -qq sox libsox-dev
!pip install -q -r requirements.txt --no-deps
```

```python
# 3. 下载模型
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

```python
# 4. 启动 Web UI
!python webui.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B --share
```

## 📋 功能特性

- **零样本语音克隆**: 只需 3-10 秒音频即可克隆声音
- **跨语种合成**: 支持中英文等不同语言的语音合成
- **指令控制**: 使用自然语言控制语速、情感、方言等
- **细粒度控制**: 支持笑声、呼吸声等特殊标记

## 🛠️ 支持的模型

| 模型 | 大小 | 功能 |
|------|------|------|
| Fun-CosyVoice3-0.5B | 0.5B | 推荐，支持多语种、细粒度控制 |
| CosyVoice2-0.5B | 0.5B | 支持跨语种、流式推理 |
| CosyVoice-300M | 300M | 基础零样本克隆 |
| CosyVoice-300M-SFT | 300M | 预训练音色 |
| CosyVoice-300M-Instruct | 300M | 指令控制模式 |

## 📝 使用示例

### 零样本克隆
```python
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

for i, j in enumerate(cosyvoice.inference_zero_shot(
    '你好，这是CosyVoice合成的语音。',
    '提示文本内容',
    'prompt_audio.wav'
)):
    torchaudio.save(f'output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### 指令控制模式
```python
# 使用方言
instruct = 'You are a helpful assistant. 请用四川话说这句话。<|endofprompt|>'
for i, j in enumerate(cosyvoice.inference_instruct2(text, instruct, prompt_wav)):
    torchaudio.save(f'output.wav', j['tts_speech'], cosyvoice.sample_rate)

# 控制语速
instruct = 'You are a helpful assistant. 请用尽可能快地语速说这句话。<|endofprompt|>'
```

### 细粒度控制
```python
# 添加笑声和呼吸声
text = '他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。'
text = '让我们[breath]深呼吸一下[breath]继续。'
```

## 🔧 故障排除

### 内存不足
- 使用较小的模型如 CosyVoice-300M
- 减少 batch size
- 启用流式推理 (stream=True)

### 依赖安装失败
```python
# 单独安装关键依赖
!pip install -q torch torchaudio transformers modelscope gradio
```

### 模型下载慢
```python
# 使用 HuggingFace 镜像
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B',
                  endpoint='https://hf-mirror.com')
```

## 🌐 Web UI 访问

启动 Web UI 后，Gradio 会生成一个公共 URL（类似 `https://xxxx.gradio.live`），您可以通过此链接在浏览器中访问界面。

## 📚 更多资源

- [项目主页](https://funaudiollm.github.io/cosyvoice3/)
- [GitHub 仓库](https://github.com/infinite-gaming-studio/CosyVoice)
- [模型文档](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)

## ⚠️ 注意事项

1. Kaggle 会话有 9 小时运行时间限制
2. GPU 使用时间每周有限额（约 30 小时）
3. 下载的模型和生成的文件在会话结束后会被清除，请及时下载
4. 建议使用 Kaggle 的 Output 目录保存重要文件

## 📄 许可

本项目遵循 Apache License 2.0 开源协议。
