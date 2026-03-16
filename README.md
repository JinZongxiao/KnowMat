# **KnowMat**：用于材料科学数据抽取的 Agentic 流水线

![KnowMat-logo](docs/_static/KnowMat-logo.jpg)

*图：KnowMat Agentic 流水线示意图，用于从科学文献中抽取结构化材料数据。*

---

## 概述

**KnowMat** 是一个先进的、AI 驱动的 Agentic 流水线，可将非结构化科学文献（`.pdf` / `.txt`）自动抽取为结构化、机器可读的材料科学数据。项目基于 **LangGraph**，并可对接 **OpenAI 兼容的 LLM API**（包括 ERNIE/Qianfan），通过多个智能代理协同完成论文解析、成分抽取、工艺条件抽取、表征信息抽取和材料性能抽取。

### 核心能力

- **科研级批处理**：可批量处理整个目录中的 PDF/TXT 文件；支持**两阶段**：先仅跑 OCR（`--ocr-only`），再统一跑大模型抽取
- **高准确度**：多代理架构，支持最多 3 轮抽取/评估迭代优化
- **高级 PDF 解析**：使用 **PaddleOCR-VL** 进行 OCR 解析
- **两阶段校验**：规则聚合 + LLM 幻觉修正
- **属性标准化**：自动将属性名称映射为标准形式
- **质量保障**：置信度打分、人工复核标记与复核指南
- **面向机器学习**：输出结构化 JSON，便于入库和建模

---

## 关键特性

### 🤖 多代理架构

- **Parser Agent**：基于 PaddleOCR-VL 的 PDF 解析
  - 将 PDF 页面转图片并逐页 OCR
  - 保存页面图片和 OCR 原始结果，便于调试
  - 生成用于后续抽取的清洗文本
- **Subfield Detection Agent**：识别论文类型（实验/计算/机器学习）并动态调整提示词
- **Extraction Agent**：基于 TrustCall 的结构化数据抽取
- **Evaluation Agent**：抽取质量评估与置信度评分（最多 3 轮）
- **Two-Stage Manager**：
  - **Stage 1（Aggregation）**：快速规则化合并多轮结果
  - **Stage 2（Validation）**：LLM 幻觉检测与修正
- **Flagging Agent**：最终质量评估与人工复核建议

### 📊 数据抽取范围

- 材料成分（元素、化学计量、归一化配方）
- 工艺条件（温度、压力、气氛、时间）
- 表征方法与结果
- 材料性能（机器学习友好格式）：
  - 精确值、区间、上下界（`>`、`<`、`≥`、`≤`）
  - 数值抽取（并正确处理定性属性）
  - 数值类型：`exact`、`lower_bound`、`upper_bound`、`range`、`qualitative`

### 🔬 属性标准化

- 使用配置的 LLM 将属性名称自动映射为标准形式
- 支持符号、缩写和领域术语

### 🛡️ 质量保障

- 每轮抽取均有置信度评分
- 基于评估反馈的幻觉检测
- 人工复核标记机制
- 详细推理说明与复核指南

### ⚡ 工程化能力

- 支持整目录批处理
- 每篇论文独立输出目录
- 完整 JSON + 可读分析报告 + 多轮记录
- 支持 LangSmith tracing 调试
- 支持按代理单独配置模型（成本/质量平衡）
- **回归测试工具**：自动化 AI vs Ground Truth 对比，量化关键指标

---

## 输出目录结构

- **输入目录**（默认 `data/raw/`）：用户放置待处理 PDF 及 .txt/.md 的目录。**OCR 中间产物**（`.md` 与 `.json`）始终写在此目录下，按论文分子目录：`<input-folder>/<论文基名>/<论文基名>.md` 与 `<论文基名>.json`。
- **输出目录**（默认 `data/output/`）：**仅存放 LLM 抽取结果**（`_extraction.json`、`_analysis_report.txt`、`_runs.json`、`_qa_report.json` 等），与 raw 分离。可通过 `--output-dir` 覆盖。

**输入目录**（如 `data/raw/`）下仅包含 PDF 与 OCR 产出：

```text
data/raw/
├── <PaperName>.pdf
└── <PaperName>/
    ├── <PaperName>.md                       # OCR 产出
    ├── <PaperName>.json                     # OCR 结构化
    ├── paddleocrvl_parse/                   # 仅 PDF 且 save_intermediate 时
    │   ├── page_images/
    │   └── ocr_raw/
    └── txt_parse/                           # TXT 输入时
```

**输出目录**（默认 `data/output/`）下为每篇论文的抽取结果：

```text
data/output/
└── <PaperName>/
    ├── <PaperName>_extraction.json          # 最终结构化结果
    ├── <PaperName>_analysis_report.txt      # 可读分析报告
    ├── <PaperName>_runs.json                # 各轮抽取详情
    └── <PaperName>_qa_report.json           # 质量与复核标记
```

---

## 安装

### 前置要求

1. **Python 3.11**
2. **Conda（推荐，用于管理虚拟环境）**
3. **OpenAI 兼容 LLM API Key**（例如 ERNIE/Qianfan）
4. **LangChain API Key**（可选，用于 LangSmith tracing）

### 快速开始（推荐）

1. **克隆仓库**

```bash
git clone https://github.com/shiyuasuka/KnowMat.git
cd KnowMat
```

2. **运行一键环境脚本**

Windows（PowerShell）：

```powershell
.\scripts\setup_env.ps1
```

仅 CPU：

```powershell
.\scripts\setup_env.ps1 -CPU
```

Linux / macOS：

```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

仅 CPU（Linux）：

```bash
./scripts/setup_env.sh --cpu
```

说明：脚本会创建/更新 `KnowMat` conda 环境、按平台安装 Paddle 与 OCR 依赖、下载 OCR 模型到 `models/paddleocrvl1_5`。多平台差异详见 [docs/platforms.md](docs/platforms.md)；若 OCR 报 `cudnn64_9.dll`，见 [docs/ocr-cudnn64_9-fix.md](docs/ocr-cudnn64_9-fix.md)。

3. **配置 `.env`**

```bash
cp .env.example .env
```

填写：

```bash
LLM_API_KEY=<你的_llm_api_key>
LLM_BASE_URL=<你的_openai_兼容_base_url>
LLM_MODEL=<你的模型名称或端点>
PADDLEOCRVL_MODEL_DIR=models/paddleocrvl1_5
LANGCHAIN_API_KEY=<你的_langchain_api_key>  # 可选
LANGCHAIN_TRACING_V2=false                  # 可选
```

ERNIE/Qianfan 示例：

```bash
LLM_API_KEY="bce-v3/xxxx"
LLM_BASE_URL="https://qianfan.bj.baidubce.com/v2"
LLM_MODEL="ep_xxxxx"
```

也可直接设置环境变量：

```bash
# Windows PowerShell
$env:LLM_API_KEY="your_key"
$env:LLM_BASE_URL="https://qianfan.bj.baidubce.com/v2"
$env:LLM_MODEL="ep_xxxxx"

# Linux/Mac
export LLM_API_KEY="your_key"
export LLM_BASE_URL="https://qianfan.bj.baidubce.com/v2"
export LLM_MODEL="ep_xxxxx"
```

4. **验证安装**

```bash
python -m knowmat --help
```

### 手动安装（可选）

当你不使用一键脚本时，可按下面步骤手动安装：

1. 创建并激活 conda 环境

```bash
conda env create -f environment.yml
conda activate KnowMat
```

2. 按需安装 GPU 依赖（Windows/Linux + NVIDIA）

```bash
pip install -r requirements-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
conda install nvidia::cudnn cuda-version=12
```

3. 下载 OCR 模型

```bash
python scripts/download_paddleocrvl_models.py --model-dir models/paddleocrvl1_5
```

### 开发者工作流

#### 本地开发与测试

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

#### CI

仓库内包含 GitHub Actions 工作流（`.github/workflows/ci.yml`），在推送到 `main`/`master` 或 PR 时自动执行：

- 依赖安装（含 `.[dev]`）
- 代码风格检查：`ruff`、`black --check`
- 类型检查：`mypy`
- 单元测试：`pytest`

#### Prompt 模板维护

当前核心提示词集中在 `prompts/`，代码通过 `src/knowmat/prompt_loader.py` 统一加载。

维护建议：

1. 优先修改模板文件，不在节点 Python 中硬编码长提示词。
2. 新增 YAML 模板时，在对应节点声明 required keys 校验。
3. 修改后至少运行一次 lint/语法检查和一条小样本流程验证。

轻量校验命令（本地与 CI 同步）：

```bash
python scripts/validate_prompts.py
```

---

## 使用方法

### 推荐目录与自动处理逻辑

**输入**：用户将待处理的 PDF 或 .txt/.md 放在 **`data/raw/`**。不指定 `--input-folder` 时，默认从此目录读取。OCR 产出的 .md/.json 始终写回 **`data/raw/<论文基名>/`**，与原文同区。

**输出**：**LLM 抽取结果**默认写入 **`data/output/`**，与 raw 分离。不指定 `--output-dir` 时使用 `data/output`；指定后则写入该目录下的 `<论文基名>/`。

程序按「同名文件基名」执行：

1. 若已存在同名的 `xxx.txt` 或 `xxx/xxx.md`（如先前 OCR 生成），则**直接进入大模型抽取**（或仅在 `--ocr-only` 时**跳过**该文件）；
2. 若存在 `xxx.pdf` 且不存在对应 `xxx/xxx.md` 或 `xxx.txt`，先对 PDF 做 OCR，在 **输入目录** 下生成 `xxx/xxx.md` 与 `xxx/xxx.json`，再用该 .md 进入大模型抽取；
3. 同一基名下已有 .md 或 .txt 时优先复用，不重复 OCR；
4. 使用 **`--force-rerun`** 时，会重新对所有 PDF 做 OCR 并重新抽取，忽略已有 .md 与 extraction JSON。

**两阶段用法（先批量 OCR，再跑大模型）**：使用 `--ocr-only` 仅跑 OCR，生成全部 `<input>/<论文基名>/<论文基名>.md` 与 `.json` 后，去掉该参数再运行即只做 LLM 抽取，结果写入 `data/output/`（见下文「仅跑 OCR」）。

默认目录下的布局示例：

```text
data/raw/                    # 输入 + OCR 中间产物
├── 论文A.pdf
├── 论文A/
│   ├── 论文A.md
│   └── 论文A.json
└── ...

data/output/                 # 抽取结果（默认）
├── 论文A/
│   ├── 论文A_extraction.json
│   ├── 论文A_analysis_report.txt
│   ├── 论文A_runs.json
│   └── 论文A_qa_report.json
└── ...
```

重点查看输出目录中的：

- `<论文基名>_extraction.json`：结构化抽取结果
- `<论文基名>_analysis_report.txt`：分析报告
- `<论文基名>_runs.json`：多轮运行明细
- `<论文基名>_qa_report.json`：质量与复核标记

### 基础命令行用法

使用默认目录（推荐）：

```bash
python -m knowmat
```

等价于：从 **`data/raw`** 读取输入，OCR 中间产物写回 **`data/raw/<论文基名>/`**，LLM 抽取结果写入 **`data/output/<论文基名>/`**。

指定输入/输出目录：

```bash
python -m knowmat --input-folder path/to/files
# OCR 在 path/to/files/<论文基名>/，抽取结果在 data/output/<论文基名>/

python -m knowmat --input-folder path/to/files --output-dir path/to/output
# OCR 在 path/to/files/ 下，LLM 抽取结果在 path/to/output/<论文基名>/ 下
```

行为说明：

- **`.pdf`**：若无对应 .md，先用 PaddleOCR-VL 解析，在输入目录下生成 `<基名>/<基名>.md` 与 `.json`，再进行 LLM 抽取；结果写入输出目录。
- **`.txt` / 已有 `.md`**：跳过 OCR，直接进入 LLM 抽取，结果写入输出目录。

### 仅跑 OCR（两阶段用法）

若希望先批量跑完 OCR，再统一或分批跑大模型解析，可使用 `--ocr-only`：

```bash
# 第一步：只对 PDF 做 OCR，输出到 <input-folder>/<论文基名>/<论文基名>.md 和 .json
python -m knowmat --input-folder path/to/pdfs --ocr-only

# 第二步：去掉 --ocr-only，对已生成的 .md 做 LLM 抽取（不会重复 OCR），结果写入 data/output
python -m knowmat --input-folder path/to/pdfs
```

适用场景：OCR 耗时较长时可先集中跑完；或需在不同机器/环境分别跑 OCR 与 LLM 时使用。

### 进阶参数示例

```bash
python -m knowmat \
    --input-folder path/to/files \
    --output-dir output/directory \
    --max-runs 1 \
    --workers 4 \
    --force-rerun \
    --only 1-2024-MSEA-Ti₄₂Hf₂₁Nb₂₁V₁₆-DED 2-2025-Acta-Nb₁₅Ta₁₀W₇₅---添加NbC\ 纳米沉淀 \
    --extraction-model ${LLM_MODEL} \
    --evaluation-model ${LLM_MODEL} \
    --manager-model ${LLM_MODEL} \
    --subfield-model ${LLM_MODEL} \
    --flagging-model ${LLM_MODEL}
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input-folder` | 输入目录（用户放置待处理 PDF/.txt/.md 的目录） | `data/raw` |
| `--pdf-folder` | `--input-folder` 的旧别名 | - |
| `--output-dir` | 抽取结果输出目录；不指定时为 `data/output`，与输入（raw）分离 | `data/output` |
| `--ocr-only` | 仅跑 OCR，不跑 LLM 抽取；输出为 `<input-folder>/<基名>/<基名>.md` 与 `.json` | `False` |
| `--ocr-workers` | 并发 OCR 的 PDF 数量（有 GPU 时建议 1） | `1` |
| `--ocr-log-level` | OCR/PaddleX 日志级别（如 DEBUG、INFO、WARNING） | - |
| `--max-runs` | 每篇论文的最大抽取/评估轮数 | `1` |
| `--workers` | 并发处理文件数（LLM 抽取） | `1` |
| `--full-pipeline` | 启用完整多阶段流水线 | `False` |
| `--enable-property-standardization` | 启用属性标准化后处理 | `False` |
| `--force-rerun` | 强制重新 OCR 并重新抽取所有论文，忽略已有 .md 与 `_extraction.json` | `False` |
| `--only` | 仅处理指定文件（按 stem 或完整文件名匹配，可传多个） | - |
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

## 输出示例
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
        },
        {
          "property_name": "fracture strength",
          "property_symbol": "σ_max",
          "value": ">1800",
          "value_numeric": 1800.0,
          "value_type": "lower_bound",
          "units": "MPa"
        }
      ]
    }
  ]
}
```

---

## 项目结构

```text
├── AUTHORS.md              <- 开发者与维护者列表
├── CHANGELOG.md            <- 版本变更记录（新功能与修复）
├── CONTRIBUTING.md         <- 项目贡献指南
├── LICENSE.txt             <- MIT 许可证
├── README.md               <- 项目说明文档（当前文件）
├── .env.example            <- 环境变量模板（仅占位符，不含真实密钥）
├── Dockerfile              <- 容器构建定义
├── environment.yml         <- Conda 环境依赖定义
├── pyproject.toml          <- 构建系统配置
├── requirements.txt        <- pip 依赖列表（补充安装场景）
├── setup.cfg               <- 包元数据与依赖配置
├── setup.py                <- 安装脚本（已弱化，建议使用 pip install -e .）
├── tox.ini                 <- 多环境测试配置
├── .coveragerc             <- 覆盖率配置
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml          <- CI：lint/type-check/test
├── .isort.cfg              <- isort 排序配置
├── .pre-commit-config.yaml <- 预提交钩子配置
├── .readthedocs.yml        <- Read the Docs 构建配置
│
├── prompts/                <- 提示词模板（YAML/TXT）
│   ├── README.md                       <- 说明 prompts 文件夹用途及各模板用法
│   ├── evaluation.yaml                 <- 评估抽取质量相关的提示词模板
│   ├── flagging.yaml                   <- 质量标记/错误标记等相关提示词
│   ├── subfield_detection.yaml         <- 领域或子类领域自动识别用的提示词
│   ├── extraction_system_template.txt  <- 系统角色（System）的大模型提示模板
│   └── extraction_user_template.txt    <- 用户输入（User）的大模型提示模板
│
├── data/                   <- 数据目录
│   ├── raw/                <- 原始输入与 OCR 中间产物（.pdf/.txt + <PaperName>/*.md, *.json）
│   ├── output/             <- 抽取结果输出目录（默认）
│   │   └── <PaperName>/    <- 按论文分组的抽取结果
│   ├── external/           <- 外部数据或中间产物
│   └── interim/            <- 临时中间数据
│
├── src/                    <- 源代码目录
│   └── knowmat/            <- 主包
│       ├── __init__.py
│       ├── __main__.py     <- CLI 入口
│       ├── orchestrator.py <- 流水线编排（LangGraph）
│       ├── app_config.py   <- 应用配置（默认输入/输出目录与模型配置）
│       ├── config.py       <- 环境变量加载与运行时配置
│       ├── states.py       <- LangGraph 状态定义
│       ├── extractors.py   <- TrustCall/Pydantic 抽取结构定义
│       ├── prompt_generator.py <- 动态提示词生成
│       ├── prompt_loader.py <- 外部模板加载器（prompts/*）
│       ├── post_processing.py  <- 属性标准化后处理
│       ├── schema_converter.py <- 内部格式到目标 HEA  schema 转换（domain_rules 驱动）
│       ├── report_writer.py    <- 抽取分析报告生成
│       ├── domain_rules.py     <- 领域规则加载器（读 domain_rules.yaml）
│       ├── domain_rules.yaml   <- 领域规则配置（相推断、沉淀检测、工艺分类等）
│       ├── properties.json     <- 属性标准化词库
│       ├── pdf/            <- PDF 解析子包（拆分模块）
│       │   ├── __init__.py            <- 子包初始化
│       │   ├── blocks.py              <- PDF 文档块（段落、表格、图片等）解析
│       │   ├── doi_extractor.py       <- 从 PDF 中自动识别并提取 DOI
│       │   ├── html_cleaner.py        <- PDF 转 HTML 后内容结构清洗与规范化
│       │   ├── ocr_engine.py          <- OCR 引擎接口（如 PaddleOCR），实现图片转文本
│       │   └── section_normalizer.py  <- 章节与分段结构标准化（规范章节标题等）
│       └── nodes/          <- 各代理节点实现
│           ├── __init__.py
│           ├── paddleocrvl_parse_pdf.py  <- PDF/TXT 解析节点（含 OCR 逻辑）
│           ├── docling_parse_pdf.py      <- 兼容层：原 docling 接口 → PaddleOCR-VL
│           ├── subfield_detection.py     <- 子领域识别节点
│           ├── extraction.py             <- 结构化抽取节点
│           ├── evaluation.py             <- 抽取质量评估节点
│           ├── aggregator.py             <- Manager 第 1 阶段：规则聚合
│           ├── validator.py              <- Manager 第 2 阶段：LLM 校验修正
│           ├── schema_convert.py         <- 内部格式 → 目标 schema 转换节点
│           ├── standardize.py            <- 属性名标准化节点（可选，依赖 properties.json）
│           └── flagging.py               <- 最终质量标记节点
│
├── tools/                  <- 开发工具集
│   ├── regression_diff.py  <- 回归测试工具（AI vs GT 对比）
│   └── README.md           <- 工具使用说明
│
├── scripts/                <- 脚本与环境准备
│   ├── compare_to_manual.py       <- 与人工标注对比
│   ├── download_paddleocrvl_models.py <- PaddleOCR-VL 模型下载
│   ├── setup_env.ps1              <- Windows 一键环境部署（Conda + Paddle + OCR）
│   ├── setup_env.sh               <- Linux/macOS 一键环境部署
│   ├── validate_prompts.py        <- 模板完整性校验脚本
│   └── train_model.py             <- 模型训练脚本
│
├── reports/                <- 回归与 QA 报告输出目录
│   ├── regression_*.md    <- Markdown 格式报告
│   └── regression_*.json  <- JSON 格式报告
│
├── tests/                  <- 单元测试（pytest）
│   ├── conftest.py         <- pytest fixtures
│   ├── test_domain_rules.py
│   └── test_schema_converter.py
│
├── notebooks/              <- 数据分析与实验笔记
│   ├── create_annotation.ipynb <- 标注创建
│   └── template.ipynb      <- 实验模板
│
├── models/                 <- 本地模型存放目录（.gitignore）
├── references/             <- 参考文档或资料
└── docs/                   <- 文档目录（Sphinx，已同步中文说明）
    ├── conf.py             <- Sphinx 配置
    ├── Makefile            <- 文档构建
    ├── index.md, readme.md, changelog.md, contributing.md, authors.md, license.md
    └── _static/            <- 静态资源
```

---

## 高级能力

### 两阶段 Manager 架构

KnowMat 将“合并”和“校验”分离，兼顾效率与可靠性：

**Stage 1 - Aggregation（规则化）**：
- 快速、确定性地合并多轮抽取结果
- 无 LLM 调用（零额外模型成本）
- 以高置信度轮次为基础补全信息

**Stage 2 - Validation（LLM 校验）**：
- 基于评估反馈定位幻觉字段
- 修正疑似幻觉内容
- 检查输出是否满足机器学习可用格式
- 提供重试机制与人工复核建议

### 属性标准化

PostProcessor 会将抽取出的属性名称映射为标准术语。  
示例：

- `"Dimensionless figure of merit ZT"` → `"thermoelectric figure of merit"`
- `"glass transition temp"` → `"glass transition temperature"`
- `"ultimate tensile strength"` → `"tensile strength"`
- `"Young's modulus"` → `"elastic modulus"`

### 批处理能力

```bash
python -m knowmat --input-folder data/raw/papers --output-dir data/output
```

控制台示例：

```text
Processing file 1/150: paper001.pdf
Processing file 2/150: notes002.txt
...

Total files: 150
Successful: 147
Failed: 3
Flagged for review: 23
Total compositions: 2,341
```

### LangSmith Tracing

启用后可在 LangSmith 观察完整链路：

- 每次 LLM 调用与响应
- 抽取决策与提示词效果
- 各代理交互过程
- Token 消耗与成本

访问：<https://smith.langchain.com/>

---

## 模型选择与成本优化

支持按代理单独配置模型，以平衡成本与精度。  
使用 ERNIE/Qianfan 时，设置 `LLM_MODEL` 后可按需覆盖各代理模型。

推荐配置（生产）：

```bash
--subfield-model ${LLM_MODEL}
--extraction-model ${LLM_MODEL}
--evaluation-model ${LLM_MODEL}
--manager-model ${LLM_MODEL}
--flagging-model ${LLM_MODEL}
```

---

## 工具集

### 回归测试工具（Regression Diff - 三模式）

KnowMat 提供了功能强大的回归测试工具 `tools/regression_diff.py`，支持三种运行模式：

| 模式 | 说明 | 需要 GT | 适用场景 |
|------|------|---------|----------|
| **gt** | AI vs Ground Truth | ✅ 需要 | 精确评估有手工标注的论文（6篇） |
| **self** | Run vs Run（自回归） | ❌ 不需要 | 对比两次 AI 抽取结果，验证 prompt 优化效果（所有论文） |
| **qa** | 质量基线检查 | ❌ 不需要 | 快速发现完全失败或质量极差的论文（所有论文） |

#### 核心功能

**GT 模式**（AI vs Ground Truth）：

- 结构指标对比（Materials/Samples/Properties 数量）
- 字段准确率评估（DOI、Main_Phase、Process_Category 等）
- 温度质量检查（偏移统计）
- 成分解析质量（非法元素、百分比和校验）

**Self 模式**（Run vs Run）：
- 创建快照保存当前抽取结果
- 对比两次抽取的差异（Materials/Samples/Properties 数量变化）
- 检测 DOI/Phase/Process 的变化趋势
- 无需 GT，适用于所有论文

**QA 模式**（质量基线检查）：
- 扫描所有论文抽取结果，计算质量得分（0-100）
- 红线告警：自动标记需人工复核的论文
- 质量指标：材料数、属性数、DOI 存在率、Phase 填充率等
- 按质量得分排序，快速定位最差论文

#### 快速开始

```bash
# GT 模式：对比 6 篇有手工标注的论文
python tools/regression_diff.py gt --all

# Self 模式：保存快照并对比变化
python tools/regression_diff.py self --snapshot baseline
# (修改 prompt 后重跑 AI 抽取)
python tools/regression_diff.py self --compare baseline

# QA 模式：扫描所有论文质量
python tools/regression_diff.py qa

# 向后兼容（等价于 gt --all）
python tools/regression_diff.py --all
```

#### 输出报告

**Markdown 报告**（人类可读）：
- 总体指标表格（结构指标、字段准确率、温度质量、成分质量）
- 逐篇详细对比（标记 ✅ 达标 / ❌ 未达标）
- 具体问题列表（成分错误、DOI 缺失等）

**JSON 报告**（机器可读）：
- 结构化数据，便于 CI/CD 集成
- 包含所有指标的详细数值
- 支持程序化分析和趋势监控

#### 典型工作流

```bash
# 1. 基线测试（优化前）
python tools/regression_diff.py --all --output reports/baseline

# 2. 修改 prompt 或 schema
# (编辑 src/knowmat/prompt_generator.py 或 extractors.py)

# 3. 重跑 AI 抽取
python -m knowmat --force-rerun --max-runs 1

# 4. 对比验证（优化后）
python tools/regression_diff.py --all --output reports/after_optimization

# 5. 对比两次报告，确认关键指标是否改善：
#    - DOI 命中率是否提升？
#    - Main_Phase 填充率是否改善？
#    - Process_Category Unknown 占比是否降低？
#    - 温度偏移是否消除？
#    - 成分解析问题是否减少？
```

#### 控制台输出示例

```
============================================================
回归测试摘要
============================================================

📊 关键指标:
  DOI 命中率:             0.0% ❌
  Main_Phase 填充率:      16.7% ❌
  Process_Category 准确率: 16.7% ❌
  温度平均偏移:           127.09 K ❌

⚠️ 成分质量问题:
  非法元素:   1 篇
  百分比异常: 5 篇
  空成分:     1 篇
============================================================
```

#### 评价标准

工具采用以下阈值标记 ✅ 达标 / ❌ 未达标：

- **DOI 命中率** ≥ 80%
- **Main_Phase 填充率** ≥ 70%
- **Process_Category 非 Unknown** ≥ 85%
- **温度平均偏移** < 0.5 K
- **无成分解析问题**（无非法元素、百分比和正常）

详细使用说明见 [`tools/README.md`](tools/README.md)。

---

## 故障排查

### 常见问题

**1）API Key 未设置**

```text
Error: LLM_API_KEY not set
```

解决：确保设置 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`。

**2）401 invalid_model / 识别不出材料**

```text
Error code: 401 - invalid_model ... The model does not exist or you do not have access to it.
```

原因：大模型抽取阶段被 API 拒绝，因此不会生成材料抽取结果（`_extraction.json` 等）。

解决：检查 `.env` 中的 **`LLM_MODEL`**。
- **千帆（Qianfan）**：必须填写在控制台创建的**推理端点 ID**（形如 `ep_xxxxx`），不能是模型简称；并确认当前 API Key 对应账号有该端点的访问权限。
- 若使用其他 OpenAI 兼容 API，填写对方要求的模型名或端点 ID。

**3）PaddleOCR-VL 解析失败**

```text
Error: Failed to parse PDF with PaddleOCR-VL
A dependency error occurred during pipeline creation.
信息: 用提供的模式无法找到文件。
```

解决：
- 检查 PDF 是否损坏或加密，必要时重新下载。
- **依赖错误 / “Could not find files for the given pattern(s)”**：若 PaddleOCR-VL 初始化失败（依赖或 pipeline 创建报错），程序会**自动回退到经典 PaddleOCR** 继续执行；若仍失败，多为 PaddlePaddle 运行环境问题。Windows 请安装 [Visual C++ Redistributable](https://learn.microsoft.com/zh-cn/cpp/windows/latest-supported-vc-redist)；使用 GPU 时确认 CUDA/cuDNN 与 Paddle 版本匹配，按 [docs/platforms.md](docs/platforms.md) 进行平台化安装。
- **“用提供的模式无法找到文件”**：若 PDF 路径或文件名含非 ASCII 字符（如下标 ₄₂、中文等），程序会自动复制到临时 ASCII 路径再解析；若仍报错，可先重命名为纯英文路径再试。
- 需跳过“Checking connectivity to the model hosters”时，可设置环境变量 `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`（离线或内网时常用）；程序在创建引擎时也会自动尝试设置。

**4）属性标准化失败**

```text
Warning: Property standardization failed
```

解决：确认属性配置文件存在且为合法 JSON。

**5）ERNIE 接口报 Unrecognized function call PatchFunction**

```text
Unrecognized function call PatchFunction
```

说明：该提示来自 **LLM 客户端**（TrustCall/LangChain）在解析 ERNIE/千帆 API 的**返回结果**时发现工具调用名不是我们请求的 `CompositionList`，而是 `PatchFunction`。**ERNIE 官方文档中未公开该接口**，很可能是其 OpenAI 兼容层或内部实现返回的非标准工具名。

影响：客户端无法按预期解析工具调用，可能伴随 `CompositionList` 的校验错误；我们已在 schema 中为必填字段增加默认值，部分返回仍可通过校验并写入结果（QA 会标记需复核）。

---

## 引用

如果你的研究使用了 KnowMat，请引用：

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

欢迎贡献代码与建议。详情见 `CONTRIBUTING.md`。

基本流程：

1. Fork 仓库
2. 新建分支（如 `feature/amazing-feature`）
3. 提交修改
4. 推送分支
5. 发起 Pull Request

---

## 许可证

本项目使用 MIT License，详见 `LICENSE.txt`。

---

## 致谢

- Built with [PyScaffold](https://pyscaffold.org/)
- Powered by [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- PDF parsing by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

---

## 支持

- GitHub Issues：<https://github.com/hasan-sayeed/KnowMat2/issues>
- Email：hasan.sayeed@utah.edu
