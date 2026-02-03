# CosyVoice Google Colab 部署指南

使用 **Miniconda + Google Drive 持久化** 部署 CosyVoice，实现真正的断点续传。

## 🎯 核心特性

- ✅ **Conda 环境隔离** - Python 3.10，避免与 Colab 系统冲突
- ✅ **Google Drive 持久化** - 环境、模型、代码永久保存
- ✅ **断点续传** - 会话中断后可立即恢复，无需重复安装
- ✅ **免费 GPU** - T4 GPU 加速（12-15GB 显存）
- ✅ **Gradio 公网访问** - 自动生成可分享的 Web UI 链接

## 🚀 快速开始

### 方式 1：直接打开（推荐）

1. 上传 `cosyvoice_colab_conda.ipynb` 到你的 Google Drive
2. 右键 → 打开方式 → Google Colaboratory
3. 运行时 → 更改运行时类型 → **GPU T4**
4. 运行第 0 节挂载 Drive
5. 首次运行：依次执行 1-7 节（约 20-30 分钟）
6. 后续运行：只执行第 0 节和第 7 节（约 2 分钟）

### 方式 2：从 GitHub 导入

1. 打开 [Google Colab](https://colab.research.google.com)
2. 文件 → 上传笔记本 → GitHub
3. 粘贴仓库地址：`https://github.com/infinite-gaming-studio/CosyVoice`
4. 选择 `cosyvoice_colab_conda.ipynb`

## 📁 目录结构

```
Google Drive/MyDrive/CosyVoice_Colab/    # 工作目录（永久保存）
├── miniconda3/                           # Miniconda 安装
│   └── envs/cosyvoice/                   # Conda 环境
├── CosyVoice/                            # 代码仓库
│   ├── cosyvoice/                        # 核心代码
│   ├── webui.py                          # Web 界面
│   └── third_party/                      # 子模块
├── models/                               # 模型文件
│   └── Fun-CosyVoice3-0.5B-2512/         # 下载的模型
├── batch_outputs/                        # 批量生成结果
├── test_output_*.wav                     # 测试音频
└── cosyvoice_environment.yml             # 环境配置导出
```

## 🔄 会话恢复流程

Colab 免费版有 12 小时会话限制，但我们的数据都保存在 Drive：

### 首次部署（20-30 分钟）
```
第0节 → 第1节 → 第2节 → 第3节 → 第4节 → 第5节 → 第6节 → 第7节
(挂载)  (conda)  (代码)   (环境)   (模型)   (验证)   (测试)   (启动UI)
```

### 后续恢复（2 分钟）
```
第0节 → 第7节
(挂载)   (启动UI)
```

**无需重新安装任何东西！**

## 🎮 使用 Web UI

启动第 7 节后，会看到类似输出：

```
Running on local URL: http://0.0.0.0:50000
Running on public URL: https://xxxx.gradio.live  ← 使用这个链接
```

### 访问方式：
1. **直接点击链接**（在新标签页打开）
2. **分享链接**（链接 72 小时内有效）
3. **手机访问**（支持移动端浏览器）

### 功能模式：
- **预训练音色** - 使用内置说话人
- **3秒极速复刻** - 上传 3-10 秒音频克隆声音
- **跨语种复刻** - 中英文等不同语言合成
- **自然语言控制** - 用指令控制语速、情感、方言

## 🛠️ 支持的模型

修改 Notebook 第 4 节的 `MODEL_NAME` 变量：

| 模型 | 大小 | 特点 | 下载时间 |
|------|------|------|----------|
| **Fun-CosyVoice3-0.5B-2512** ⭐ | 0.5B | 最新，支持多语种、细粒度控制 | ~8 分钟 |
| CosyVoice2-0.5B | 0.5B | 流式推理优化 | ~8 分钟 |
| CosyVoice-300M | 300M | 轻量级，快速测试 | ~5 分钟 |
| CosyVoice-300M-Instruct | 300M | 指令控制模式 | ~5 分钟 |
| CosyVoice-300M-SFT | 300M | 预训练音色 | ~5 分钟 |

## 📝 使用示例

### 零样本语音克隆
```python
# 上传 prompt 音频到 Drive
prompt_wav = '/content/drive/MyDrive/CosyVoice_Colab/my_voice.wav'
prompt_text = '这段音频中的说话内容'

# 克隆
text = '你好，这是用我的声音合成的语音。'
for i, result in enumerate(model.inference_zero_shot(text, prompt_text, prompt_wav)):
    torchaudio.save(f'cloned_{i}.wav', result['tts_speech'], model.sample_rate)
```

### 指令控制（方言、语速、情感）
```python
# 四川话
text = '今天天气真不错。'
instruct = 'You are a helpful assistant. 请用四川话说这句话。<|endofprompt|>'

# 快速语速
instruct = 'You are a helpful assistant. 请用很快的语速说。<|endofprompt|>'

# 带情感
instruct = 'You are a helpful assistant. 请用兴奋的语气说。<|endofprompt|>'

for i, result in enumerate(model.inference_instruct2(text, instruct, prompt_wav)):
    torchaudio.save(f'output_{i}.wav', result['tts_speech'], model.sample_rate)
```

### 细粒度控制（笑声、呼吸）
```python
text = '他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。'
text = '让我们深呼吸一下[breath]继续。'
text = '这是一个<strong>重要</strong>的提醒。'

for i, result in enumerate(model.inference_cross_lingual(text, prompt_wav)):
    torchaudio.save(f'fine_grained_{i}.wav', result['tts_speech'], model.sample_rate)
```

### 跨语种合成
```python
# 用中文声音说英文
text = '<|en|>Hello, this is cross-lingual voice synthesis.'
for i, result in enumerate(model.inference_cross_lingual(text, chinese_prompt_wav)):
    torchaudio.save(f'cross_lingual_{i}.wav', result['tts_speech'], model.sample_rate)
```

## 📊 批量生成

使用 Notebook 第 8 节批量生成多个音频：

```python
texts = [
    "第一句要合成的内容",
    "第二句要合成的内容",
    "第三句要合成的内容",
]

# 自动保存到 Drive/CosyVoice_Colab/batch_outputs/
```

## 🐛 故障排除

### 1. 挂载 Drive 失败
```python
# 重新挂载
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 2. CUDA Out of Memory
```python
# 清理显存
!{CONDA_DIR}/envs/cosyvoice/bin/python -c "import torch; torch.cuda.empty_cache()"

# 或使用更小模型
MODEL_NAME = 'CosyVoice-300M'  # 代替 Fun-CosyVoice3-0.5B-2512
```

### 3. 模型下载太慢/失败
```python
# 使用 HuggingFace 镜像
!{CONDA_DIR}/envs/cosyvoice/bin/pip install -q huggingface_hub

from huggingface_hub import snapshot_download
snapshot_download(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir=MODEL_PATH,
    endpoint='https://hf-mirror.com'  # 国内镜像
)
```

### 4. 环境损坏
```python
# 删除环境重新创建
!rm -rf /content/drive/MyDrive/CosyVoice_Colab/miniconda3/envs/cosyvoice
# 然后重新运行第 3 节
```

### 5. Drive 空间不足
检查空间使用：
```python
!du -sh /content/drive/MyDrive/CosyVoice_Colab/*
```

清理不必要的文件：
```python
# 清理 conda 缓存
!rm -rf /content/drive/MyDrive/CosyVoice_Colab/miniconda3/pkgs/*

# 删除旧模型（如果下载了多个）
!rm -rf /content/drive/MyDrive/CosyVoice_Colab/models/CosyVoice-300M
```

**存储需求：**
- Miniconda: ~3GB
- 环境: ~5GB
- 模型: ~2-4GB
- **总计: 10-15GB**

### 6. Web UI 无法访问
- 检查是否正确运行第 7 节
- 确保显示了 `https://xxxx.gradio.live` 链接
- 链接有效期 72 小时，超时后需要重新启动

## 🎓 进阶技巧

### 保存多个声音配置
```python
# 保存不同的说话人
model.add_zero_shot_spk('我的声音描述', './my_voice.wav', 'my_voice_id')
model.add_zero_shot_spk('朋友声音描述', './friend_voice.wav', 'friend_voice_id')

# 保存配置
model.save_spkinfo()

# 以后直接使用 ID
text = '你好'
for i, result in enumerate(model.inference_zero_shot(text, '', '', zero_shot_spk_id='my_voice_id')):
    torchaudio.save(f'output.wav', result['tts_speech'], model.sample_rate)
```

### 流式推理（低延迟）
```python
# 开启流式模式（注意：Web UI 也支持选择流式/非流式）
def text_generator():
    yield '第一句话'
    yield '第二句话'
    yield '第三句话'

for i, result in enumerate(model.inference_zero_shot(text_generator(), prompt_text, prompt_wav, stream=True)):
    # 实时播放或保存
    pass
```

### 自定义推理脚本
创建 `my_script.py` 保存在 Drive，然后：
```bash
!cd /content/drive/MyDrive/CosyVoice_Colab && \
  /content/drive/MyDrive/CosyVoice_Colab/miniconda3/envs/cosyvoice/bin/python my_script.py
```

## ⚡ 性能优化

### 使用 V100/A100（Colab Pro）
运行时 → 更改运行时类型 → GPU → 选择 V100 或 A100

### 启用混合精度
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 批量推理
单次加载模型，批量处理多段文本（见第 8 节）

## 🔗 相关资源

- **原始仓库**: https://github.com/infinite-gaming-studio/CosyVoice
- **官方文档**: https://funaudiollm.github.io/cosyvoice3/
- **模型下载**: https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- **Colab 帮助**: https://colab.research.google.com/notebooks/

## ⚠️ 注意事项

1. **免费版限制**：
   - GPU 会话最长 12 小时
   - 空闲 90 分钟会自动断开
   - T4 GPU 显存 12-15GB

2. **存储**：
   - Google Drive 提供 15GB 免费空间
   - 环境 + 模型约占用 10-15GB
   - 定期清理不需要的模型

3. **网络**：
   - 需要稳定的网络连接
   - 模型下载约需 5-10 分钟

4. **分享**：
   - Gradio 链接 72 小时有效
   - 可截图分享生成的音频

## 🆘 需要帮助？

1. 查看原始仓库 [Issues](https://github.com/infinite-gaming-studio/CosyVoice/issues)
2. 参考官方 [CosyVoice 文档](https://github.com/FunAudioLLM/CosyVoice)
3. 在 GitHub 提交新的 Issue

## 🎉 成功标志

看到以下输出表示部署成功：
```
✅ 工作目录: /content/drive/MyDrive/CosyVoice_Colab
✅ Conda: /content/drive/MyDrive/CosyVoice_Colab/miniconda3
✅ 代码: /content/drive/MyDrive/CosyVoice_Colab/CosyVoice
✅ 模型: /content/drive/MyDrive/CosyVoice_Colab/models

✅ 模型加载完成！采样率: 24000 Hz
Running on public URL: https://xxxx.gradio.live
```

**现在可以点击链接开始使用 CosyVoice！** 🎊
