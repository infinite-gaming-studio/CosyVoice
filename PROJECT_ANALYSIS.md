# CosyVoice 项目详细分析文档

**项目路径**: `/Users/nvozi/Coding/ai-based-projects/CosyVoice`  
**分析日期**: 2026-02-03  
**版本**: Fun-CosyVoice 3.0 / CosyVoice 2.0 / CosyVoice 1.0

---

## 一、项目概述

### 1.1 项目简介

CosyVoice 是阿里巴巴通义实验室开源的基于大语言模型（LLM）的文本转语音（TTS）系统。该项目提供了先进的零样本语音合成能力，支持多种语言、方言、情感和风格控制。

### 1.2 主要版本

| 版本 | 模型大小 | 主要特点 |
|------|----------|----------|
| **Fun-CosyVoice 3.0** | 0.5B | 最新版本，9种语言+18种方言，DiT架构，发音修复 |
| **CosyVoice 2.0** | 0.5B | 流式推理，25Hz采样率，Qwen2基础模型 |
| **CosyVoice 1.0** | 300M | 基础版本，支持SFT/Zero-shot/Cross-lingual/Instruct |

### 1.3 核心能力

- **多语言支持**: 中文、英语、日语、韩语、德语、西班牙语、法语、意大利语、俄语
- **方言支持**: 广东话、闽南语、四川话、东北话、山西话、上海话、天津话、山东话、宁夏话、甘肃话等
- **零样本克隆**: 仅需3秒音频即可克隆音色
- **跨语言合成**: 支持不同语言之间的音色迁移
- **细粒度控制**: 支持情感、语速、音量、呼吸、笑声等控制
- **流式推理**: 支持双向流式（文本输入流+音频输出流），延迟低至150ms

---

## 二、项目结构

```
CosyVoice/
├── cosyvoice/                    # 核心代码库
│   ├── __init__.py
│   ├── bin/                      # 训练与导出脚本
│   │   ├── train.py              # 训练脚本
│   │   ├── export_jit.py         # TorchScript导出
│   │   ├── export_onnx.py        # ONNX导出
│   │   └── average_model.py      # 模型平均
│   ├── cli/                      # 命令行接口
│   │   ├── cosyvoice.py          # AutoModel与推理类
│   │   ├── model.py              # 模型推理实现
│   │   └── frontend.py           # 前端处理
│   ├── llm/                      # 大语言模型
│   │   └── llm.py                # TransformerLM/Qwen2LM/CosyVoice3LM
│   ├── flow/                     # 流匹配模型
│   │   ├── flow.py               # MaskedDiff/CausalDiff
│   │   ├── flow_matching.py      # 条件流匹配
│   │   ├── decoder.py            # 解码器
│   │   ├── length_regulator.py   # 长度调节器
│   │   └── DiT/                  # Diffusion Transformer
│   │       ├── dit.py
│   │       └── modules.py
│   ├── hifigan/                  # 声码器
│   │   ├── generator.py          # HiFTGenerator
│   │   ├── hifigan.py            # HiFiGAN
│   │   ├── discriminator.py      # 判别器
│   │   └── f0_predictor.py       # 基频预测
│   ├── transformer/              # Transformer组件
│   │   ├── encoder.py            # 编码器
│   │   ├── decoder.py            # 解码器
│   │   ├── attention.py          # 注意力机制
│   │   ├── encoder_layer.py      # 编码层
│   │   ├── decoder_layer.py      # 解码层
│   │   ├── embedding.py          # 嵌入层
│   │   └── ...
│   ├── tokenizer/                # 分词器
│   │   └── tokenizer.py          # CosyVoice2/3 Tokenizer
│   ├── dataset/                  # 数据处理
│   │   ├── dataset.py
│   │   └── processor.py
│   └── utils/                    # 工具函数
│       ├── common.py             # 通用函数
│       ├── mask.py               # 掩码工具
│       ├── class_utils.py        # 类注册器
│       ├── losses.py             # 损失函数
│       ├── file_utils.py         # 文件工具
│       └── ...
├── runtime/                      # 部署运行时
│   ├── python/                   # Python服务
│   │   ├── grpc/                 # gRPC服务
│   │   │   ├── server.py
│   │   │   └── client.py
│   │   └── fastapi/              # FastAPI服务
│   │       ├── server.py
│   │       └── client.py
│   └── triton_trtllm/            # Triton推理服务
│       ├── model_repo/           # 模型仓库
│       ├── scripts/              # 工具脚本
│       ├── client_grpc.py
│       ├── client_http.py
│       └── offline_inference.py
├── tools/                        # 数据处理工具
│   ├── extract_speech_token.py
│   ├── extract_embedding.py
│   └── make_parquet_list.py
├── examples/                     # 示例与配置
│   ├── libritts/                 # LibriTTS示例
│   ├── magicdata-read/           # MagicData示例
│   └── grpo/                     # GRPO训练示例
├── third_party/                  # 第三方依赖
│   └── Matcha-TTS/               # Matcha-TTS
├── docker/                       # Docker配置
├── asset/                        # 静态资源
├── webui.py                      # Gradio Web界面
├── example.py                    # 使用示例
├── vllm_example.py               # vLLM使用示例
├── requirements.txt              # 依赖列表
├── README.md                     # 项目文档
├── LICENSE                       # Apache-2.0许可证
├── CODE_OF_CONDUCT.md
└── FAQ.md
```

---

## 三、系统架构

### 3.1 三阶段级联架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CosyVoice Pipeline                              │
└─────────────────────────────────────────────────────────────────────────────┘

阶段1: 大语言模型 (LLM)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Input: 文本 + 说话人嵌入 + 提示音频token                                    │
│         ↓                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ TransformerLM / Qwen2LM / CosyVoice3LM                                │  │
│  │                                                                        │  │
│  │  • 文本编码器 → 文本嵌入                                               │  │
│  │  • 自回归生成 → 语音token序列                                          │  │
│  │  • 双流训练支持 (文本+语音混合)                                        │  │
│  │  • vLLM加速支持                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│         ↓                                                                   │
│  Output: 离散语音token序列                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
阶段2: 流匹配模型 (Flow Matching)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Input: 离散语音token + 条件特征                                           │
│         ↓                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ MaskedDiffWithXvec / CausalMaskedDiffWithXvec / CausalMaskedDiffWithDiT│  │
│  │                                                                        │  │
│  │  • Token编码器                                                         │  │
│  │  • 长度调节器 (Length Regulator)                                       │  │
│  │  • 条件流匹配 (Conditional CFM)                                        │  │
│  │  • 欧拉求解器 (Euler Solver)                                           │  │
│  │  • 分类器自由引导 (CFG)                                                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│         ↓                                                                   │
│  Output: 梅尔谱 (Mel-spectrogram)                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
阶段3: 神经声码器 (Neural Vocoder)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Input: 梅尔谱                                                              │
│         ↓                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ HiFTGenerator / CausalHiFTGenerator                                   │  │
│  │                                                                        │  │
│  │  • F0预测器                                                            │  │
│  │  • 神经源滤波 (NSF)                                                    │  │
│  │  • ISTFTNet                                                            │  │
│  │  • 因果卷积支持流式                                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│         ↓                                                                   │
│  Output: 音频波形 (Waveform)                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 版本演进对比

| 组件 | CosyVoice 1.0 | CosyVoice 2.0 | Fun-CosyVoice 3.0 |
|------|---------------|---------------|-------------------|
| **LLM** | TransformerLM | Qwen2LM (0.5B) | CosyVoice3LM (0.5B) |
| **Flow** | MaskedDiffWithXvec | CausalMaskedDiffWithXvec | CausalMaskedDiffWithDiT |
| **Vocoder** | HiFTGenerator | HiFTGenerator | CausalHiFTGenerator |
| **架构** | 非因果 | 因果+流式 | 因果+DiT+流式 |
| **采样率** | 50Hz | 25Hz | 25Hz |
| **vLLM** | ❌ | ✅ | ✅ |
| **TensorRT** | ✅ | ✅ | ✅ |

---

## 四、核心模块详解

### 4.1 LLM 模块 (`cosyvoice/llm/`)

#### 4.1.1 架构设计

**CosyVoice 1.0 - TransformerLM**
- 基于传统Transformer架构
- 自回归语言模型
- 支持文本和语音token的混合训练

**CosyVoice 2.0 - Qwen2LM**
- 基于Qwen2-0.5B基础模型
- 引入双流训练机制
- 支持文本流式输入
- 集成vLLM加速

**Fun-CosyVoice 3.0 - CosyVoice3LM**
- 扩展Tokenizer支持更多控制token
- 支持DPO（直接偏好优化）训练
- 增强的跨语言/方言能力

#### 4.1.2 关键特性

```python
# 1. 双流训练 (CosyVoice2/3)
mix_ratio = [5, 15]  # 文本和语音token的混合比例

# 2. 重复感知采样 (RAS)
# 防止LLM生成重复token，提升稳定性

# 3. vLLM支持
# 通过vLLM的LLMEngine实现高效推理
# 支持 continuous batching

# 4. 流式文本输入
# 支持生成器(generator)作为文本输入
# 适用于LLM实时生成文本场景
```

#### 4.1.3 文件位置
- 核心代码: `cosyvoice/llm/llm.py`

### 4.2 Flow 模块 (`cosyvoice/flow/`)

#### 4.2.1 架构设计

**基础组件**
1. **条件流匹配** (`ConditionalCFM`): 基于流的生成模型
2. **掩码扩散**: 支持部分观测数据的生成
3. **长度调节器**: 调整token序列长度

**版本演进**
- **CosyVoice 1.0**: 非因果扩散模型
- **CosyVoice 2.0**: 因果扩散模型，支持流式
- **CosyVoice 3.0**: DiT (Diffusion Transformer) 架构

#### 4.2.2 关键技术

```python
# 1. 条件流匹配
# 从噪声分布到目标分布的连续变换
mu = encoder(token)
x = mu + sigma * noise  # 加噪
v = estimator(x, t, cond)  # 预测速度
x_next = x + v * dt  # 欧拉步进

# 2. 分类器自由引导 (CFG)
# 提升生成质量，cfg_rate=0.7
uncond = estimator(x, t, null_cond)
cond = estimator(x, t, real_cond)
v = uncond + cfg_rate * (cond - uncond)

# 3. DiT (Diffusion Transformer)
# 使用Transformer替代UNet进行扩散建模
```

#### 4.2.3 文件位置
- Flow模型: `cosyvoice/flow/flow.py`
- 流匹配: `cosyvoice/flow/flow_matching.py`
- 解码器: `cosyvoice/flow/decoder.py`
- DiT: `cosyvoice/flow/DiT/dit.py`

### 4.3 HiFT/HiFi-GAN 模块 (`cosyvoice/hifigan/`)

#### 4.3.1 架构设计

**HiFTGenerator (CosyVoice 1.0/2.0)**
```
梅尔谱 → F0预测器 → SourceModuleHnNSF → ResBlock → ISTFT → 波形
```

**CausalHiFTGenerator (CosyVoice 3.0)**
```
梅尔谱 → F0预测器 → SourceModuleHnNSF → 因果ResBlock → ISTFT → 波形
```

#### 4.3.2 关键技术

1. **神经源滤波 (NSF)**: 生成谐波源信号
2. **ISTFTNet**: 使用逆短时傅里叶变换，高效合成
3. **F0预测器**: 从梅尔谱预测基频，辅助语音生成
4. **因果卷积**: `CausalConv1d`，支持实时流式生成

#### 4.3.3 文件位置
- 生成器: `cosyvoice/hifigan/generator.py`
- HiFiGAN: `cosyvoice/hifigan/hifigan.py`
- F0预测器: `cosyvoice/hifigan/f0_predictor.py`

### 4.4 Transformer 模块 (`cosyvoice/transformer/`)

#### 4.4.1 组件清单

| 组件 | 文件 | 功能 |
|------|------|------|
| TransformerEncoder | `encoder.py` | 标准Transformer编码器 |
| ConformerEncoder | `encoder.py` | Conformer架构（卷积+自注意力） |
| MultiHeadedAttention | `attention.py` | 多头自注意力 |
| RelPositionMultiHeadedAttention | `attention.py` | 相对位置编码注意力 |
| TransformerEncoderLayer | `encoder_layer.py` | 标准Transformer层 |
| ConformerEncoderLayer | `encoder_layer.py` | Conformer层 |

#### 4.4.2 关键技术

1. **相对位置编码**: ESPnet风格，更好建模长序列
2. **分块掩码**: `subsequent_chunk_mask`，支持流式推理
3. **动态分块**: `add_optional_chunk_mask`，自适应调整块大小
4. **梯度检查点**: 节省显存

### 4.5 Tokenizer 模块 (`cosyvoice/tokenizer/`)

#### 4.5.1 支持的Token类型

**CosyVoice 2.0**
- 多语言支持: 99+种语言
- 音频事件: `<|Speech|>`, `<|BGM|>`, `<|Laughter|>`, `<|Applause|>`, `<|Sing|>`
- 情感标签: `<|HAPPY|>`, `<|SAD|>`, `<|ANGRY|>`, `<|NEUTRAL|>`
- 细粒度控制: `[breath]`, `[laughter]`, `[cough]`, `[noise]`
- 音频结束: `<|endofaudio|>`
- 提示结束: `<|endofprompt|>`

**Fun-CosyVoice 3.0 (新增)**
- 方言控制: 广东话、四川话、上海话、天津话等
- 发音控制:
  - 中文拼音: `[j][ǐ]`
  - 英文音素: `[AH0]`, `[N]` 等 (CMU音素集)
- 日语片假名: レキシ テキ セカイ...

#### 4.5.2 使用示例

```python
# 情感控制
"在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。"

# 笑声插入
"他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。"

# 方言控制 (CosyVoice3)
"好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。"
# + instruct: "请用广东话表达。"

# 发音修复 (CosyVoice3)
"高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。"
# 将"给予"的发音纠正为 jǐ yǔ
```

### 4.6 CLI 模块 (`cosyvoice/cli/`)

#### 4.6.1 核心类

| 类名 | 功能 | 对应版本 |
|------|------|----------|
| `CosyVoice` | 基础推理类 | CosyVoice 1.0 |
| `CosyVoice2` | 扩展推理类 | CosyVoice 2.0 |
| `CosyVoice3` | 最新推理类 | CosyVoice 3.0 |
| `AutoModel` | 自动模型加载 | 通用 |
| `CosyVoiceModel` | 模型推理实现 | CosyVoice 1.0 |
| `CosyVoice2Model` | 模型推理实现 | CosyVoice 2.0 |
| `CosyVoice3Model` | 模型推理实现 | CosyVoice 3.0 |

#### 4.6.2 推理模式

1. **SFT (Supervised Fine-Tuning)**: 使用预训练音色
2. **Zero-shot**: 3秒音频克隆音色
3. **Cross-lingual**: 跨语言音色迁移
4. **Instruct**: 自然语言控制情感/风格/方言
5. **VC (Voice Conversion)**: 音色转换

#### 4.6.3 流式推理

```python
# 流式推理示例
def text_generator():
    yield '收到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '笑容如花儿般绽放。'

for output in cosyvoice.inference_zero_shot(
    text_generator(),  # 生成器输入
    prompt_text='希望你以后能够做的比我还好呦。',
    prompt_wav='./asset/zero_shot_prompt.wav',
    stream=True  # 启用流式
):
    torchaudio.save('output.wav', output['tts_speech'], sample_rate)
```

---

## 五、部署运行时

### 5.1 Python 服务

#### 5.1.1 gRPC 服务
```bash
# 启动服务端
cd runtime/python/grpc
python server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M

# 客户端调用
python client.py --port 50000 --mode zero_shot
```

#### 5.1.2 FastAPI 服务
```bash
# 启动服务端
cd runtime/python/fastapi
python server.py --port 50000 --model_dir iic/CosyVoice-300M

# 客户端调用
python client.py --port 50000 --mode sft
```

### 5.2 NVIDIA Triton + TensorRT-LLM

#### 5.2.1 快速启动
```bash
cd runtime/triton_trtllm
docker compose up -d
```

#### 5.2.2 模型仓库结构
```
model_repo/
├── audio_tokenizer/          # 音频tokenizer
├── speaker_embedding/        # 说话人嵌入
├── token2wav/                # token到波形
├── token2wav_dit/            # DiT版本token2wav
├── cosyvoice2/               # CosyVoice2完整模型
└── cosyvoice2_dit/           # DiT版本
```

#### 5.2.3 客户端
- `client_grpc.py`: gRPC客户端
- `client_http.py`: HTTP客户端
- `offline_inference.py`: 离线批量推理
- `streaming_inference.py`: 流式推理

---

## 六、依赖要求

### 6.1 核心依赖 (`requirements.txt`)

| 依赖 | 版本 | 用途 |
|------|------|------|
| torch | 2.3.1 | PyTorch深度学习框架 |
| transformers | 4.51.3 | Hugging Face模型 |
| vllm | 0.11.0 | vLLM推理加速 |
| tensorrt-cu12 | 10.13.3.9 | TensorRT推理优化 |
| onnxruntime-gpu | 1.18.0 | ONNX推理 |
| deepspeed | 0.15.1 | 分布式训练 |
| gradio | 5.4.0 | Web界面 |
| fastapi | 0.115.6 | Web服务 |
| grpcio | 1.57.0 | gRPC服务 |
| modelscope | 1.20.0 | 模型下载 |
| librosa | 0.10.2 | 音频处理 |
| soundfile | 0.12.1 | 音频读写 |
| whisper | 20231117 | 语音识别 |

### 6.2 硬件要求

- **最低配置**: NVIDIA GPU with 8GB VRAM (推理)
- **推荐配置**: NVIDIA GPU with 16GB+ VRAM (训练)
- **流式推理**: 支持CUDA的GPU
- **TensorRT**: 仅支持Linux + NVIDIA GPU

---

## 七、使用示例

### 7.1 基础使用

```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

# 加载模型
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# SFT推理
for output in cosyvoice.inference_sft('你好，我是通义生成式语音大模型', '中文女'):
    torchaudio.save('sft.wav', output['tts_speech'], cosyvoice.sample_rate)

# Zero-shot克隆
for output in cosyvoice.inference_zero_shot(
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。',
    '希望你以后能够做的比我还好呦。',
    './asset/zero_shot_prompt.wav'
):
    torchaudio.save('zero_shot.wav', output['tts_speech'], cosyvoice.sample_rate)

# 跨语言推理
for output in cosyvoice.inference_cross_lingual(
    '<|en|>And then later on, fully acquiring that company.',
    './asset/cross_lingual_prompt.wav'
):
    torchaudio.save('cross_lingual.wav', output['tts_speech'], cosyvoice.sample_rate)
```

### 7.2 进阶使用

```python
# 自然语言控制 (CosyVoice3)
for output in cosyvoice.inference_instruct2(
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    '请用四川话说这句话。',
    './asset/zero_shot_prompt.wav'
):
    torchaudio.save('instruct.wav', output['tts_speech'], cosyvoice.sample_rate)

# 流式推理
def text_stream():
    yield '第一句话。'
    yield '第二句话。'

for output in cosyvoice.inference_zero_shot(
    text_stream(),
    'prompt文本',
    'prompt.wav',
    stream=True
):
    # 实时获取音频片段
    pass

# 语速调节 (非流式)
for output in cosyvoice.inference_zero_shot(
    '要合成的文本',
    'prompt文本',
    'prompt.wav',
    stream=False,
    speed=1.5  # 1.5倍速
):
    torchaudio.save('fast.wav', output['tts_speech'], cosyvoice.sample_rate)
```

### 7.3 Web界面

```bash
# 启动Gradio Web界面
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

---

## 八、模型下载

### 8.1 ModelScope (国内推荐)

```python
from modelscope import snapshot_download

snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice2-0.5B', 
                  local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', 
                  local_dir='pretrained_models/CosyVoice-300M')
```

### 8.2 HuggingFace (国际用户)

```python
from huggingface_hub import snapshot_download

snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('FunAudioLLM/CosyVoice2-0.5B', 
                  local_dir='pretrained_models/CosyVoice2-0.5B')
```

---

## 九、训练与微调

### 9.1 数据准备

```python
# 提取语音token
python tools/extract_speech_token.py --input_dir data --output_dir tokens

# 提取说话人嵌入
python tools/extract_embedding.py --input_dir data --output_dir embeddings

# 生成Parquet列表
python tools/make_parquet_list.py --input_dir data --output list.parquet
```

### 9.2 训练脚本

```bash
cd examples/libritts/cosyvoice
# 修改 conf/cosyvoice.yaml 配置
python cosyvoice/bin/train.py --config conf/cosyvoice.yaml
```

### 9.3 微调模式

1. **SFT微调**: 使用`CosyVoice-300M-SFT`作为基础模型
2. **Instruct微调**: 使用`CosyVoice-300M-Instruct`作为基础模型
3. **GRPO训练**: 强化学习优化，见`examples/grpo/`

---

## 十、性能优化

### 10.1 推理加速

| 技术 | 加速比 | 适用场景 |
|------|--------|----------|
| vLLM | 2-4x | CosyVoice2/3 |
| TensorRT | 4x | Flow解码器 |
| TorchScript | 1.5x | 所有版本 |
| ONNX | 1.2x | 跨平台部署 |
| FP16 | 2x | 显存优化 |

### 10.2 流式优化

```python
# 启用流式推理
cosyvoice.inference_zero_shot(..., stream=True)

# 关键参数
# - token_hop_len: 25 (CosyVoice2/3)
# - stream_scale_factor: 2
# - pre_lookahead_len: 流式前瞻长度
```

---

## 十一、项目贡献者

- 主要开发: 阿里巴巴通义实验室语音团队
- 致谢项目:
  - [FunASR](https://github.com/modelscope/FunASR)
  - [FunCodec](https://github.com/modelscope/FunCodec)
  - [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
  - [AcademiCodec](https://github.com/yangdongchao/AcademiCodec)
  - [WeNet](https://github.com/wenet-e2e/wenet)

---

## 十二、相关链接

- **论文**:
  - CosyVoice 1.0: [arXiv:2407.05407](https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf)
  - CosyVoice 2.0: [arXiv:2412.10117](https://arxiv.org/pdf/2412.10117)
  - Fun-CosyVoice 3.0: [arXiv:2505.17589](https://arxiv.org/pdf/2505.17589)

- **Demo页面**:
  - Fun-CosyVoice 3.0: https://funaudiollm.github.io/cosyvoice3/
  - CosyVoice 2.0: https://funaudiollm.github.io/cosyvoice2/
  - CosyVoice 1.0: https://fun-audio-llm.github.io

- **模型下载**:
  - ModelScope: https://www.modelscope.cn/models/iic/CosyVoice-300M
  - HuggingFace: https://huggingface.co/FunAudioLLM/CosyVoice-300M

- **代码仓库**:
  - GitHub: https://github.com/FunAudioLLM/CosyVoice

---

## 十三、许可证

本项目采用 [Apache License 2.0](LICENSE) 开源许可证。

---

*文档生成日期: 2026-02-03*  
*分析工具: opencode AI*
