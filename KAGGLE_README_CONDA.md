# CosyVoice Kaggle Conda 部署指南

使用 **Miniconda + 持久化目录** 部署 CosyVoice，解决 Python 版本兼容问题。

## 🎯 核心特性

- ✅ **Conda 环境隔离** - Python 3.10，避免与 Kaggle 系统环境冲突
- ✅ **持久化存储** - 所有数据保存在 `/kaggle/working/cosyvoice_env`
- ✅ **增量部署** - 环境只需创建一次，后续会话可快速恢复
- ✅ **支持会话续跑** - 中断后无需重复下载模型和安装依赖

## 🚀 快速开始

### 1. 上传 Notebook

1. 打开 [Kaggle](https://www.kaggle.com)
2. 创建新的 Notebook
3. 点击 **File → Import Notebook**
4. 上传 `cosyvoice_kaggle_conda.ipynb`

### 2. 配置环境

在 Notebook 右侧设置面板：
- **Accelerator**: `GPU T4 x2` 或 `P100`
- **Internet**: `On`
- **Environment**: 保持默认（Notebook 会自己安装 Miniconda）

### 3. 运行部署

按顺序运行以下关键单元格：

| 步骤 | 单元格 | 说明 | 执行次数 |
|------|--------|------|----------|
| 0 | 配置持久化路径 | 设置工作目录 | 每次会话 |
| 1 | 安装 Miniconda | 仅需执行一次 | 首次 |
| 2 | 克隆代码仓库 | 仅需执行一次 | 首次 |
| 3 | 创建 Conda 环境 | 仅需执行一次（约 10-15 分钟） | 首次 |
| 4 | 下载模型 | 仅需执行一次（约 5-10 分钟） | 首次 |
| 5 | 验证安装 | 检查环境 | 可选 |
| 6 | 测试推理 | 测试模型 | 可选 |
| 7 | 启动 Web UI | 启动服务 | 每次会话 |

## 📁 目录结构

```
/kaggle/working/cosyvoice_env/          # 工作目录（持久化）
├── miniconda3/                         # Miniconda 安装
│   └── envs/cosyvoice/                 # CosyVoice Conda 环境
│       ├── bin/python                  # Python 3.10
│       └── lib/python3.10/...          # 安装的包
├── CosyVoice/                          # 代码仓库
│   ├── cosyvoice/                      # 核心代码
│   ├── webui.py                        # Web 界面
│   └── third_party/                    # 子模块
├── models/                             # 模型文件
│   └── Fun-CosyVoice3-0.5B-2512/       # 下载的模型
├── test_output_*.wav                   # 生成的音频
└── cosyvoice_environment.yml           # 环境配置导出
```

## 🔄 会话恢复流程

如果 Kaggle GPU 会话超时或被中断：

```python
# 只需运行这两步即可恢复：

# 1. 重新配置路径（第0节）
WORK_DIR = '/kaggle/working/cosyvoice_env'
CONDA_DIR = f'{WORK_DIR}/miniconda3'
REPO_DIR = f'{WORK_DIR}/CosyVoice'
MODEL_PATH = f'{WORK_DIR}/models/Fun-CosyVoice3-0.5B-2512'

# 2. 直接启动 Web UI（第7节）
!cd {REPO_DIR} && {CONDA_DIR}/envs/cosyvoice/bin/python webui.py \
    --port 50000 \
    --model_dir {MODEL_PATH} \
    --share
```

**注意**：所有之前下载的数据都在 `/kaggle/working` 中保留！

## 🛠️ 支持的模型

修改第4节中的 `MODEL_NAME` 变量切换模型：

```python
MODEL_NAME = 'Fun-CosyVoice3-0.5B-2512'  # 推荐，多语种支持
# MODEL_NAME = 'CosyVoice2-0.5B'          # 流式推理优化
# MODEL_NAME = 'CosyVoice-300M'           # 轻量级
# MODEL_NAME = 'CosyVoice-300M-Instruct'  # 指令控制模式
```

## 📦 保存为 Kaggle Dataset（长期持久化）

如果想永久保存环境和模型（跨会话、跨账号）：

### 方法1：打包为 Dataset

```python
# 在 Notebook 最后运行：
!tar -czf /kaggle/working/cosyvoice_complete.tar.gz /kaggle/working/cosyvoice_env
# 然后从右侧 Output 下载，再上传到 Kaggle Datasets
```

### 方法2：只保存模型

```python
# 打包模型目录
!tar -czf /kaggle/working/cosyvoice_model.tar.gz -C /kaggle/working/cosyvoice_env/models .
```

然后：
1. 在 Kaggle 主页点击 **Datasets → New Dataset**
2. 上传打包的 `.tar.gz` 文件
3. 下次使用时在 Notebook 中 **Add Data** 挂载

## 🐛 故障排除

### Conda 环境损坏

```python
# 删除环境重新创建
!rm -rf /kaggle/working/cosyvoice_env/miniconda3/envs/cosyvoice
# 然后重新运行第3节
```

### 模型下载失败

```python
# 使用 HuggingFace 镜像（国内访问更快）
!{CONDA_DIR}/envs/cosyvoice/bin/pip install -q huggingface_hub

# 修改下载脚本使用 HF
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 
                  local_dir=MODEL_PATH,
                  endpoint='https://hf-mirror.com')
```

### 磁盘空间不足

Kaggle 提供约 20GB 持久化存储：
- Miniconda: ~3GB
- CosyVoice 环境: ~5GB  
- 模型文件: ~2-4GB
- 剩余空间用于输出

如果空间不够：
```python
# 清理不需要的文件
!rm -rf /kaggle/working/cosyvoice_env/miniconda3/pkgs/*
!rm -rf /kaggle/working/cosyvoice_env/CosyVoice/.git
```

### CUDA 版本不匹配

Kaggle GPU 通常是 CUDA 12.1，Notebook 已配置对应 PyTorch 版本：
```bash
--index-url https://download.pytorch.org/whl/cu121
```

## 📝 高级用法

### 自定义推理脚本

创建 `my_inference.py`：

```python
import sys
sys.path.insert(0, '/kaggle/working/cosyvoice_env/CosyVoice')
sys.path.insert(0, '/kaggle/working/cosyvoice_env/CosyVoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

model = AutoModel(model_dir='/kaggle/working/cosyvoice_env/models/Fun-CosyVoice3-0.5B-2512')

# 零样本克隆
text = '你好，这是克隆的声音。'
prompt = '希望你以后能够做的比我还好呦。'
prompt_wav = '/path/to/prompt.wav'

for i, result in enumerate(model.inference_zero_shot(text, prompt, prompt_wav)):
    torchaudio.save(f'output_{i}.wav', result['tts_speech'], model.sample_rate)
```

运行：
```bash
!/kaggle/working/cosyvoice_env/miniconda3/envs/cosyvoice/bin/python my_inference.py
```

### 多模型共存

```python
# 下载多个模型
MODELS = [
    'Fun-CosyVoice3-0.5B-2512',
    'CosyVoice2-0.5B',
    'CosyVoice-300M-Instruct'
]

for model_name in MODELS:
    model_path = f'{WORK_DIR}/models/{model_name}'
    if not os.path.exists(model_path):
        snapshot_download(f'FunAudioLLM/{model_name}', local_dir=model_path)
```

## 🔗 相关链接

- **项目仓库**: https://github.com/infinite-gaming-studio/CosyVoice
- **原始文档**: https://github.com/FunAudioLLM/CosyVoice
- **Kaggle**: https://www.kaggle.com

## ⚠️ 注意事项

1. **会话时长**: Kaggle GPU 会话最长 9 小时
2. **GPU 限额**: 每周约 30-40 小时 GPU 时间
3. **网络**: 需要开启 Internet 才能下载模型
4. **保存数据**: 会话结束后 `/kaggle/working` 数据保留，但最好在结束前导出重要结果

## 🎉 成功标志

Web UI 启动后应显示：
```
Running on local URL: http://0.0.0.0:50000
Running on public URL: https://xxxx.gradio.live  ← 使用这个链接访问
```

用浏览器打开 public URL 即可使用图形界面！
