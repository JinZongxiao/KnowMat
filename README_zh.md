# KnowMat：材料科学数据抽取 Agentic 流水线

![KnowMat-logo](docs/_static/KnowMat-logo.jpg)

_KnowMat Agentic 流水线示意图，用于从科学文献中抽取结构化材料数据。_

---

## 概述

KnowMat 是一个 AI 驱动的 Agentic 流水线，可将非结构化科学文献（`.pdf` / `.txt`）自动抽取为结构化、机器可读的材料科学数据。基于 **LangGraph** 构建，支持 **OpenAI 兼容的 LLM API**（包括 ERNIE/Qianfan），通过多智能代理协同完成论文解析、成分抽取、工艺条件抽取、表征信息抽取和材料性能抽取。

### 核心能力

- **科研级批处理**：支持整目录批量处理 PDF/TXT 文件；支持**两阶段**工作流：先仅跑 OCR（`--ocr-only`），再统一跑大模型抽取
- **高准确度**：多代理架构，支持最多 3 轮抽取/评估迭代优化
- **双引擎高精度 OCR**：PaddleOCR-VL 1.5（宏观版面与阅读顺序）+ PP-StructureV3（微观复杂表格与公式精修）；可选 MinerU 云端 API 模式（`--mineru-api`）
- **公式与表格增强**：精准提取复杂 HTML 跨行表格与高保真 LaTeX 公式（自动修复化学式上下标）
- **两阶段校验**：规则聚合 + LLM 幻觉修正
- **属性标准化**：自动将属性名称映射为标准形式
- **质量保障**：置信度打分、人工复核标记与复核指南
- **ML友好输出**：结构化 JSON，便于入库和建模

---

## 安装

### 前置要求

1. **Python 3.11**
2. **OpenAI 兼容 LLM API Key**（如 ERNIE/Qianfan）
3. **LangChain API Key**（可选，用于 LangSmith tracing）

### 第一步：克隆仓库

```bash
git clone https://github.com/shiyuasuka/KnowMat.git
cd KnowMat
```

### 第二步：安装环境（选择对应平台）

---

#### macOS（推荐使用 pip）

macOS 无 NVIDIA GPU，使用 CPU 模式。

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装项目和依赖
pip install -e .
pip install -r requirements.txt

# 下载 OCR 模型
python scripts/download_paddleocrvl_models.py --model-dir models/paddleocrvl1_5
```

---

#### Windows / Linux（推荐使用 Conda）

**GPU 模式（NVIDIA 显卡）：**

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate KnowMat

# 安装 Paddle GPU 依赖
pip uninstall -y paddlepaddle paddlepaddle-gpu
pip install -r requirements-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
conda install nvidia::cudnn cuda-version=12 -y

# 下载 OCR 模型
python scripts/download_paddleocrvl_models.py --model-dir models/paddleocrvl1_5
```

**CPU 模式（无 NVIDIA 显卡）：**

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate KnowMat

# 安装 Paddle CPU 依赖
pip install -r requirements-cpu.txt

# 下载 OCR 模型
python scripts/download_paddleocrvl_models.py --model-dir models/paddleocrvl1_5
```

---

### 第三步：配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 填入你的 API 凭证：

```bash
# LLM API 配置
LLM_API_KEY="your_llm_api_key"
LLM_BASE_URL="https://your-openai-compatible-endpoint.com/v2"
LLM_MODEL="your_model_name"

# PaddleOCR-VL 配置
PADDLEOCRVL_MODEL_DIR=models/paddleocrvl1_5
PADDLEOCRVL_VERSION=1.5

# 可选：LangSmith tracing
# LANGCHAIN_API_KEY="your_langchain_api_key"
# LANGCHAIN_TRACING_V2=false
```

**ERNIE/Qianfan 示例：**

```bash
LLM_API_KEY="bce-v3/xxxx"
LLM_BASE_URL="https://qianfan.bj.baidubce.com/v2"
LLM_MODEL="ep_xxxxx"
```

### 第四步：验证安装

```bash
python -m knowmat --help
```

---

### 依赖文件说明

| 文件 | 用途 |
|------|------|
| `environment.yml` | Conda 完整环境定义 |
| `requirements.txt` | pip 基础依赖 |
| `requirements-gpu.txt` | GPU Paddle 依赖（NVIDIA） |
| `requirements-cpu.txt` | CPU Paddle 依赖 |
| `pyproject.toml` | 项目元数据 |

---

## 配置

### 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `LLM_API_KEY` | 是 | - | LLM API 密钥 |
| `LLM_BASE_URL` | 是 | - | OpenAI 兼容 base URL |
| `LLM_MODEL` | 是 | - | 默认模型名称 |
| `PADDLEOCRVL_MODEL_DIR` | 否 | `models/paddleocrvl1_5` | OCR 模型目录 |
| `PADDLEOCRVL_VERSION` | 否 | `1.5` | PaddleOCR-VL 版本 |
| `LANGCHAIN_API_KEY` | 否 | - | LangSmith API 密钥 |
| `LANGCHAIN_TRACING_V2` | 否 | `false` | 启用 LangSmith tracing |
| `MINERU_API_KEY` | 否 | - | MinerU API 密钥（启用 `--mineru-api`） |
| `MINERU_MODEL_VERSION` | 否 | `vlm` | MinerU 模型：`vlm` 或 `doclayout` |
| `MINERU_API_TIMEOUT_SEC` | 否 | `600` | MinerU 轮询超时时间（秒） |
| `MINERU_LANGUAGE` | 否 | `en` | MinerU 文档语言 |
| `PADDLEOCR_API_TOKEN` | 否 | - | PaddleOCR 云端 API Token（启用 `--paddleocr-api`） |
| `PADDLEOCR_API_URL` | 否 | `https://paddleocr.aistudio-app.com/api/v2/ocr/jobs` | PaddleOCR API 地址 |
| `PADDLEOCR_API_TIMEOUT_SEC` | 否 | `600` | PaddleOCR 轮询超时时间（秒） |

### OCR 调优（可选）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OCR_RENDER_DPI` | `300` | 页面渲染分辨率 |
| `OCR_BATCH_SIZE` | `2` | VL 批处理大小（小显存 GPU 设为 1） |
| `OCR_PAGES_PER_RELEASE` | `0` | 每 N 页释放一次 GPU 显存 |
| `KNOWMAT_SKIP_CHEM_REOCR` | 未设置 | 设为 `1` 跳过化学式重 OCR |

OCR 问题排查见 [docs/ocr-cudnn64_9-fix.md](docs/ocr-cudnn64_9-fix.md)。

---

## 使用方法

### 基础命令

```bash
python -m knowmat
```

默认从 `data/raw/` 读取文件，结果输出到 `data/output/`。

### 指定目录

```bash
python -m knowmat --input-folder path/to/papers --output-dir path/to/output
```

### 两阶段工作流（大批量推荐）

**阶段一：仅跑 OCR**

```bash
python -m knowmat --input-folder path/to/papers --ocr-only
```

**阶段二：跑 LLM 抽取**

```bash
python -m knowmat --input-folder path/to/papers
```

### MinerU 云端 API 模式

KnowMat 支持使用 [MinerU](https://mineru.net) 云端 API 作为替代 OCR 后端。MinerU 提供基于 VLM 的高质量 PDF 解析，对复杂表格、公式和图片的识别效果更好。

**配置：**

在 `.env` 中添加 MinerU API 密钥：

```bash
MINERU_API_KEY="your_mineru_api_key"
MINERU_MODEL_VERSION=vlm          # 可选：vlm（默认）、doclayout
MINERU_API_TIMEOUT_SEC=600        # 轮询超时时间（秒）
MINERU_LANGUAGE=en                # 文档语言
```

**使用方法：**

```bash
# 仅 OCR（使用 MinerU API）
python -m knowmat --input-folder path/to/papers --ocr-only --mineru-api

# 全流程（MinerU API + LLM 抽取）
python -m knowmat --input-folder path/to/papers --mineru-api

# 强制重新跑（忽略缓存）
python -m knowmat --input-folder path/to/papers --ocr-only --mineru-api --skip-cached-ocr
```

`--mineru-api` 参数启用 MinerU API 模式。不加此参数时，默认使用本地 PaddleOCR-VL 推理。MinerU API 模式需要在 `.env` 中配置 `MINERU_API_KEY`。

**相比本地 OCR 的优势：**
- 本地无需 GPU
- 更高质量的图片提取（由 MinerU 预裁剪）
- 更优的 VLM 版面分析
- 更好地支持复杂多栏排版

**注意：** MinerU API 需要网络连接，用量受 API 套餐限制。

### PaddleOCR 云端 API 模式

KnowMat 还支持使用 [PaddleOCR 云端 API](https://paddleocr.aistudio-app.com) 作为 OCR 后端。提供与本地模式相同的 PaddleOCR-VL + PP-StructureV3 流水线，但运行在云端（本地无需 GPU）。

**配置：**

在 `.env` 中添加 PaddleOCR API Token：

```bash
PADDLEOCR_API_TOKEN="your_paddleocr_api_token"
PADDLEOCR_API_URL=https://paddleocr.aistudio-app.com/api/v2/ocr/jobs
PADDLEOCR_API_TIMEOUT_SEC=600
```

**使用方法：**

```bash
# 仅 OCR（使用 PaddleOCR 云端 API）
python -m knowmat --input-folder path/to/papers --ocr-only --paddleocr-api

# 全流程（PaddleOCR 云端 API + LLM 抽取）
python -m knowmat --input-folder path/to/papers --paddleocr-api

# 强制重新跑（忽略缓存）
python -m knowmat --input-folder path/to/papers --ocr-only --paddleocr-api --skip-cached-ocr
```

**PP-StructureV3 公式精修 + MinerU：**

当同时配置了 `PADDLEOCR_API_TOKEN` 和 `MINERU_API_KEY` 时，使用 `--mineru-api` 会自动对 MinerU 结果进行 PP-StructureV3 公式/表格精修：

```bash
# MinerU 主 OCR + PP-StructureV3 公式精修
python -m knowmat --input-folder path/to/papers --ocr-only --mineru-api
```

### 进阶参数

```bash
python -m knowmat \
    --input-folder path/to/files \
    --output-dir output/directory \
    --max-runs 3 \
    --workers 4 \
    --force-rerun \
    --enable-property-standardization
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input-folder` | 输入目录（PDF/TXT/MD 文件） | `data/raw` |
| `--output-dir` | 抽取结果输出目录 | `data/output` |
| `--ocr-only` | 仅跑 OCR，跳过 LLM 抽取 | `False` |
| `--max-runs` | 最大抽取/评估轮数 | `1` |
| `--workers` | 并发文件处理数 | `1` |
| `--mineru-api` | 使用 MinerU 云端 API 进行 OCR | `False` |
| `--paddleocr-api` | 使用 PaddleOCR 云端 API 进行 OCR | `False` |
| `--skip-cached-ocr` | 忽略 OCR 缓存，强制重新推理 | `False` |
| `--force-rerun` | 强制重新 OCR 并重新抽取 | `False` |
| `--enable-property-standardization` | 启用属性名标准化 | `False` |
| `--subfield-model` | 子领域识别模型 | `LLM_MODEL` |
| `--extraction-model` | 抽取模型 | `LLM_MODEL` |
| `--evaluation-model` | 评估模型 | `LLM_MODEL` |
| `--manager-model` | 二阶段校验模型 | `LLM_MODEL` |
| `--flagging-model` | 最终质量评估模型 | `LLM_MODEL` |

### Python API

```python
from knowmat.orchestrator import run
import os

result = run(
    pdf_path="path/to/paper.pdf",  # 也支持 .txt / .md
    output_dir="data/output",
    max_runs=3,
    subfield_model=os.getenv("LLM_MODEL"),
    extraction_model=os.getenv("LLM_MODEL"),
    evaluation_model=os.getenv("LLM_MODEL"),
    manager_model=os.getenv("LLM_MODEL"),
    flagging_model=os.getenv("LLM_MODEL"),
)

print(f"Extracted {len(result['final_data']['compositions'])} compositions")
print(f"Confidence: {result.get('confidence_score', 0):.2f}")
print(f"Flagged: {result['flag']}")
```

---

## 输出结构

### 输入目录（`data/raw/`）

```
data/raw/
├── <PaperName>.pdf
└── <PaperName>/
    ├── <PaperName>.md                       # OCR 产出
    ├── <PaperName>.json                     # OCR 结构化数据
    ├── paddleocrvl_parse/                   # （仅 --save-intermediate 时）
    │   ├── page_images/
    │   └── ocr_raw/
    └── _ocr_cache/                          # OCR 缓存
```

### 输出目录（`data/output/`）

```
data/output/
└── <PaperName>/
    ├── <PaperName>_extraction.json          # 最终结构化结果
    ├── <PaperName>_analysis_report.txt      # 可读分析报告
    ├── <PaperName>_runs.json                # 多轮抽取详情
    └── <PaperName>_qa_report.json           # 质量与复核标记
```

### 抽取结果示例

```json
{
  "compositions": [
    {
      "composition": "Zr64.13Cu15.75Ni10.12Al10",
      "composition_normalized": "Zr64Cu16Ni10Al10",
      "processing_conditions": {
        "method": "melt spinning",
        "temperature": "1400 K",
        "cooling_rate": "10^6 K/s",
        "atmosphere": "argon"
      },
      "characterization": {
        "XRD": "amorphous structure confirmed",
        "DSC": "glass transition at 625 K; crystallization at 705 K"
      },
      "properties_of_composition": [
        {
          "property_name": "glass transition temperature",
          "property_symbol": "Tg",
          "value": "625",
          "value_numeric": 625.0,
          "value_type": "exact",
          "units": "K"
        }
      ]
    }
  ]
}
```

---

## 项目结构

```
KnowMat/
├── src/knowmat/              # 主 Python 包
│   ├── __main__.py           # CLI 入口
│   ├── orchestrator.py       # LangGraph 编排
│   ├── nodes/                # LangGraph 节点
│   │   ├── paddleocrvl_parse_pdf.py
│   │   ├── extraction.py
│   │   ├── evaluation.py
│   │   └── ...
│   └── pdf/                  # PDF/OCR 子模块
│       ├── ocr_engine.py
│       └── ...
├── scripts/                  # 工具脚本
│   └── download_paddleocrvl_models.py
├── prompts/                  # LLM 提示词模板
├── configs/                  # 配置目录
├── data/                     # 数据目录
│   ├── raw/                  # 输入文件 + OCR 产出
│   └── output/               # 抽取结果
├── models/                   # OCR 模型权重（gitignored）
├── environment.yml           # Conda 环境
├── requirements*.txt        # pip 依赖
└── .env.example              # 环境变量模板
```

---

## 核心特性

### 多代理架构

- **Parser Agent**：双引擎协同 PDF 解析（PaddleOCR-VL + PP-StructureV3）
- **Subfield Detection Agent**：识别论文类型（实验/计算/机器学习）
- **Extraction Agent**：基于 TrustCall 的结构化数据抽取
- **Evaluation Agent**：质量评估与置信度评分
- **Two-Stage Manager**：
  - Stage 1（Aggregation）：快速规则化合并
  - Stage 2（Validation）：LLM 幻觉检测与修正
- **Flagging Agent**：最终质量评估与人工复核建议

### 数据抽取范围

- 材料成分（元素、化学计量、归一化配方）
- 工艺条件（温度、压力、气氛、时间）
- 表征方法与结果
- 材料性能（ML 友好格式）：
  - 精确值、区间、上下界（`>`、`<`、`>=`、`<=`）
  - 值类型：`exact`、`lower_bound`、`upper_bound`、`range`、`qualitative`

### 属性标准化

自动将属性名称映射为标准形式：
- `"glass transition temp"` → `"glass transition temperature"`
- `"ultimate tensile strength"` → `"tensile strength"`
- `"Young's modulus"` → `"elastic modulus"`

---

## 回归测试

KnowMat 内置回归测试工具，用于 AI 抽取结果与 Ground Truth 对比：

```bash
# GT 模式：AI 抽取结果与 Ground Truth 对比
python tools/regression_diff.py gt --all

# Self 模式：对比两次 AI 运行
python tools/regression_diff.py self --snapshot baseline
python tools/regression_diff.py self --compare baseline

# QA 模式：质量基线检查
python tools/regression_diff.py qa
```

详细用法见 [tools/README.md](tools/README.md)。

---

## 故障排查

### API Key 未设置

```
Error: LLM_API_KEY not set
```

解决：确保 `.env` 中设置了 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`。

### 401 Invalid Model

```
Error code: 401 - invalid_model
```

解决：检查 `.env` 中的 `LLM_MODEL`。使用千帆时，必须填写**推理端点 ID**（形如 `ep_xxxxx`），不能是模型简称。

### PaddleOCR-VL 解析失败

```
Error: Failed to parse PDF with PaddleOCR-VL
```

解决：
- 检查 PDF 是否损坏或加密
- Windows：安装 [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- GPU：确保 CUDA/cuDNN 与 Paddle 版本匹配（见 [docs/platforms.md](docs/platforms.md)）
- 可设置 `KNOWMAT_ALLOW_LEGACY_PADDLEOCR=1` 允许降级到经典 OCR

---

## 引用

如果 KnowMat 对你的研究有帮助，请引用：

```bibtex
@software{knowmat2024,
  title = {KnowMat: Agentic Pipeline for Materials Science Data Extraction},
  author = {Sayeed, Hasan},
  year = {2024},
  url = {https://github.com/hasan-sayeed/KnowMat2}
}
```

---

## 贡献

欢迎贡献代码与建议。详情见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 许可证

MIT License。见 [LICENSE.txt](LICENSE.txt)。
