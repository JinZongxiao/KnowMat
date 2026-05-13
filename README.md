# KnowMat: Agentic Pipeline for Materials Science Data Extraction

![KnowMat-logo](docs/_static/KnowMat-logo.jpg)

_KnowMat agentic pipeline for extracting structured materials data from scientific literature._

---

## Overview

KnowMat is an AI-driven, agentic pipeline that automatically extracts structured, machine-readable materials science data from unstructured scientific documents (`.pdf` / `.txt`). Built on **LangGraph** with support for **OpenAI-compatible LLM APIs** (including ERNIE/Qianfan), it coordinates multiple intelligent agents to perform paper parsing, composition extraction, processing condition extraction, characterization extraction, and material properties extraction.

### Core Capabilities

- **Research-grade batch processing**: Process entire directories of PDF/TXT files; supports **two-stage** workflow: run OCR only (`--ocr-only`) first, then batch LLM extraction
- **High accuracy**: Multi-agent architecture with up to 3 rounds of extraction/evaluation iteration
- **Dual-engine OCR**: PaddleOCR-VL 1.5 (layout + reading order) + PP-StructureV3 (complex tables & formulas); optional MinerU cloud API mode (`--mineru-api`); optional PaddleOCR cloud API mode (`--paddleocr-api`)
- **Batch parallel processing**: `--batch` mode for large-scale parallel OCR submission + LLM extraction with persistent state tracking, crash recovery, and multi-key rotation
- **Formula & table enhancement**: Precise HTML table extraction and high-fidelity LaTeX formulas (auto-fixes chemical subscripts)
- **Two-stage validation**: Rule aggregation + LLM hallucination correction
- **Property standardization**: Auto-mapping attribute names to standard forms
- **Quality assurance**: Confidence scoring, human review flags & guidelines

---

## Installation

### Prerequisites

1. **Python 3.11**
2. **OpenAI-compatible LLM API Key** (e.g., MiniMax, ERNIE/Qianfan)
3. **OCR**: PaddleOCR cloud API token or MinerU API key (recommended), or NVIDIA GPU for local inference

### Step 1: Clone the Repository

```bash
git clone https://github.com/shiyuasuka/KnowMat.git
cd KnowMat
```

### Step 2: Install Environment

---

#### Option A: Cloud OCR API (Recommended — no GPU required)

Uses PaddleOCR cloud API or MinerU API for PDF parsing. No local GPU or model download needed.

**Using venv (macOS / Linux / Windows):**

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -e .
pip install -r requirements.txt
```

**Using Conda:**

```bash
conda env create -f environment.yml
conda activate KnowMat
```

> **Note (non-git downloads):** If you downloaded a zip/tarball instead of `git clone`, the editable install inside `environment.yml` may fail with a `setuptools-scm` error. Fix by running:
> ```bash
> # Windows PowerShell:
> $env:SETUPTOOLS_SCM_PRETEND_VERSION="1.0.0"
> pip install -e .
>
> # Linux / macOS:
> SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 pip install -e .
> ```

> No OCR model download needed — OCR runs in the cloud.

---

#### Option B: Local GPU OCR (NVIDIA GPU required)

Run PaddleOCR-VL locally with GPU inference. Suitable for offline environments or large-scale local processing.

**Using Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate KnowMat

# Install Paddle GPU runtime
pip uninstall -y paddlepaddle paddlepaddle-gpu
pip install -r requirements-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
conda install nvidia::cudnn cuda-version=12 -y

# Download OCR models
python scripts/download_paddleocrvl_models.py --model-dir models/paddleocrvl1_5
```

**Using venv:**

```bash
python -m venv venv
source venv/bin/activate

pip install -e .
pip install -r requirements.txt

# Install Paddle GPU runtime
pip install -r requirements-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# Download OCR models
python scripts/download_paddleocrvl_models.py --model-dir models/paddleocrvl1_5
```

> **Note:** Local OCR requires NVIDIA GPU (CUDA). CPU-only local inference is no longer supported — use Option A (cloud API mode) for environments without a GPU.

---

### Step 3: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your API credentials:

```bash
# LLM API Configuration
LLM_API_KEY="your_llm_api_key"
LLM_BASE_URL="https://api.minimaxi.com/v1"
LLM_MODEL="MiniMax-M2.7"

# Cloud OCR API (required for Option A, pick one)
PADDLEOCR_API_TOKEN="your_paddleocr_api_token"
# or
MINERU_API_KEY="your_mineru_api_key"

# Local GPU OCR (required for Option B)
# PADDLEOCRVL_MODEL_DIR=models/paddleocrvl1_5
# PADDLEOCRVL_VERSION=1.5

# Optional: LangSmith tracing
# LANGCHAIN_API_KEY="your_langchain_api_key"
# LANGCHAIN_TRACING_V2=false
```

**ERNIE/Qianfan Example:**

```bash
LLM_API_KEY="bce-v3/xxxx"
LLM_BASE_URL="https://qianfan.bj.baidubce.com/v2"
LLM_MODEL="ep_xxxxx"
```

### Step 4: Verify Installation

```bash
python -m knowmat --help
```

---

### Dependency Files Reference

| File | Purpose |
|------|---------|
| `environment.yml` | Conda full environment definition |
| `requirements.txt` | pip base dependencies |
| `requirements-gpu.txt` | GPU Paddle dependencies (NVIDIA, Option B only) |
| `pyproject.toml` | Project metadata |

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | Yes | - | Your LLM API key |
| `LLM_BASE_URL` | Yes | - | OpenAI-compatible base URL |
| `LLM_MODEL` | Yes | - | Default model name |
| `PADDLEOCRVL_MODEL_DIR` | No | `models/paddleocrvl1_5` | OCR model directory |
| `PADDLEOCRVL_VERSION` | No | `1.5` | PaddleOCR-VL version |
| `LANGCHAIN_API_KEY` | No | - | LangSmith API key |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `MINERU_API_KEY` | No | - | MinerU API key (enables `--mineru-api`) |
| `MINERU_MODEL_VERSION` | No | `vlm` | MinerU model: `vlm` or `doclayout` |
| `MINERU_API_TIMEOUT_SEC` | No | `600` | MinerU polling timeout (seconds) |
| `MINERU_LANGUAGE` | No | `en` | Document language for MinerU |
| `PADDLEOCR_API_TOKEN` | No | - | PaddleOCR cloud API token (enables `--paddleocr-api`) |
| `PADDLEOCR_API_TOKENS` | No | - | Multiple PaddleOCR tokens, comma-separated (for `--batch` mode) |
| `PADDLEOCR_API_URL` | No | `https://paddleocr.aistudio-app.com/api/v2/ocr/jobs` | PaddleOCR API endpoint |
| `PADDLEOCR_API_TIMEOUT_SEC` | No | `600` | PaddleOCR polling timeout (seconds) |
| `MINERU_API_KEYS` | No | - | Multiple MinerU keys, comma-separated (for `--batch` mode) |

### OCR Tuning (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_RENDER_DPI` | `300` | Page rendering resolution |
| `OCR_BATCH_SIZE` | `2` | VL batch size (set to 1 for small GPUs) |
| `OCR_PAGES_PER_RELEASE` | `0` | Release GPU memory every N pages |
| `KNOWMAT_SKIP_CHEM_REOCR` | unset | Set to `1` to skip chemical formula re-OCR |

For troubleshooting OCR issues, see [docs/ocr-cudnn64_9-fix.md](docs/ocr-cudnn64_9-fix.md).

---

## Usage

### Basic Command

```bash
python -m knowmat
```

This processes files from `data/raw/` and outputs results to `data/output/`.

### Process Specific Directory

```bash
python -m knowmat --input-folder path/to/papers --output-dir path/to/output
```

### Two-Stage Workflow (Recommended for Large Batches)

**Stage 1: Run OCR only**

```bash
python -m knowmat --input-folder path/to/papers --ocr-only
```

**Stage 2: Run LLM extraction**

```bash
python -m knowmat --input-folder path/to/papers
```

This generates `.md` files from PDFs first, then processes them with LLM.

### PaddleOCR Cloud API Mode

KnowMat supports using the [PaddleOCR cloud API](https://paddleocr.aistudio-app.com) as an OCR backend. This provides the same PaddleOCR-VL + PP-StructureV3 pipeline as the local mode, but runs on cloud infrastructure (no local GPU needed).

**Setup:**

Add your PaddleOCR API token to `.env`:

```bash
PADDLEOCR_API_TOKEN="your_paddleocr_api_token"
PADDLEOCR_API_URL=https://paddleocr.aistudio-app.com/api/v2/ocr/jobs
PADDLEOCR_API_TIMEOUT_SEC=600
```

**Usage:**

```bash
# OCR only with PaddleOCR cloud API
python -m knowmat --input-folder path/to/papers --ocr-only --paddleocr-api

# Full pipeline with PaddleOCR cloud API
python -m knowmat --input-folder path/to/papers --paddleocr-api

# Force re-run (ignore cache)
python -m knowmat --input-folder path/to/papers --ocr-only --paddleocr-api --skip-cached-ocr
```

**PP-StructureV3 formula refinement for MinerU:**

When both `PADDLEOCR_API_TOKEN` and `MINERU_API_KEY` are configured, using `--mineru-api` will automatically apply PP-StructureV3 formula/table refinement on MinerU results:

```bash
# MinerU primary OCR + PP-StructureV3 formula refinement
python -m knowmat --input-folder path/to/papers --ocr-only --mineru-api
```

### MinerU Cloud API Mode

KnowMat supports using [MinerU](https://mineru.net) cloud API as an alternative OCR backend. MinerU provides high-quality PDF parsing with VLM-based layout recognition, producing better results for complex tables, formulas, and figures.

**Setup:**

Add your MinerU API key to `.env`:

```bash
MINERU_API_KEY="your_mineru_api_key"
MINERU_MODEL_VERSION=vlm          # Options: vlm (default), doclayout
MINERU_API_TIMEOUT_SEC=600        # Polling timeout in seconds
MINERU_LANGUAGE=en                # Document language
```

**Usage:**

```bash
# OCR only with MinerU API
python -m knowmat --input-folder path/to/papers --ocr-only --mineru-api

# Full pipeline with MinerU API
python -m knowmat --input-folder path/to/papers --mineru-api

# Force re-run (ignore cache)
python -m knowmat --input-folder path/to/papers --ocr-only --mineru-api --skip-cached-ocr
```

The `--mineru-api` flag activates MinerU API mode. Without this flag, the local PaddleOCR-VL inference is used (default behavior). MinerU API mode requires `MINERU_API_KEY` to be set in `.env`.

**Advantages over local OCR:**
- No GPU required on the local machine
- Higher quality figure extraction (pre-cropped by MinerU)
- Better VLM-based layout analysis
- Supports complex multi-column layouts

**Note:** MinerU API requires network access and has usage limits based on your API plan.

### Batch Parallel Mode (Large-Scale Processing)

For processing tens of thousands of PDFs, KnowMat provides a `--batch` mode that uses an asyncio event loop with persistent SQLite state tracking. This enables:

- **Fire-and-forget OCR submission**: Submit many PDFs concurrently to the cloud OCR API without blocking
- **Crash recovery**: On restart, automatically resumes from the SQLite state database
- **Multi-key rotation**: Distribute load across multiple API tokens with adaptive rate-limit cooldown
- **Streaming LLM processing**: As soon as any OCR result completes, immediately start LLM extraction in parallel

**Setup (multi-key):**

Add comma-separated tokens to `.env`:

```bash
# Multiple PaddleOCR API tokens (comma-separated)
PADDLEOCR_API_TOKENS=token_a,token_b,token_c

# Or use single token (backwards compatible)
PADDLEOCR_API_TOKEN=your_single_token

# Multiple MinerU keys (comma-separated)
MINERU_API_KEYS=key1,key2
```

**Usage:**

```bash
# Large-scale parallel processing with PaddleOCR API
python -m knowmat --input-folder path/to/papers --paddleocr-api --batch \
    --max-ocr-concurrent 30 --max-llm-concurrent 8

# With MinerU API
python -m knowmat --input-folder path/to/papers --mineru-api --batch \
    --max-ocr-concurrent 20 --max-llm-concurrent 4

# Resume after crash (automatically detects existing state DB)
python -m knowmat --input-folder path/to/papers --paddleocr-api --batch

# Custom state database path
python -m knowmat --input-folder path/to/papers --paddleocr-api --batch \
    --batch-db /path/to/state.db

# Check processing status via SQLite
sqlite3 path/to/papers/.knowmat_batch.db \
    "SELECT status, count(*) FROM tasks GROUP BY status"
```

**Batch mode CLI arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch` | Enable batch parallel mode (requires `--paddleocr-api` or `--mineru-api`) | `False` |
| `--max-ocr-concurrent` | Max concurrent OCR API submissions in flight | `20` |
| `--max-llm-concurrent` | Max concurrent LLM extraction threads | `4` |
| `--batch-db` | Path to SQLite state database | `<input-folder>/.knowmat_batch.db` |
| `--ocr-poll-interval` | Seconds between OCR job poll cycles | `10` |

**Progress output:**

```
[BATCH] 12:34:56 | done: 450/10000 | ocr_submitted: 30 | llm: 8 | pending: 9512 | failed: 0 | rate: 2.1/min | keys: 3/3 healthy
```

**Note:** `--batch` mode is completely independent from the default local-OCR streaming mode. Without `--batch`, the original `ThreadPoolExecutor`-based workflow runs unchanged.

### Advanced Options

```bash
python -m knowmat \
    --input-folder path/to/files \
    --output-dir output/directory \
    --max-runs 3 \
    --workers 4 \
    --force-rerun \
    --enable-property-standardization
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input-folder` | Input directory (PDF/TXT/MD files) | `data/raw` |
| `--output-dir` | Output directory for extractions | `data/output` |
| `--ocr-only` | Run OCR only, skip LLM extraction | `False` |
| `--max-runs` | Max extraction/evaluation rounds | `1` |
| `--workers` | Concurrent file processing | `1` |
| `--mineru-api` | Use MinerU cloud API for OCR | `False` |
| `--paddleocr-api` | Use PaddleOCR cloud API for OCR | `False` |
| `--skip-cached-ocr` | Ignore OCR cache, force re-inference | `False` |
| `--force-rerun` | Force re-OCR and re-extraction | `False` |
| `--enable-property-standardization` | Enable property name standardization | `False` |
| `--subfield-model` | Subfield detection model | `LLM_MODEL` |
| `--extraction-model` | Extraction model | `LLM_MODEL` |
| `--evaluation-model` | Evaluation model | `LLM_MODEL` |
| `--manager-model` | Two-stage manager model | `LLM_MODEL` |
| `--flagging-model` | Flagging model | `LLM_MODEL` |
| `--batch` | Enable batch parallel mode | `False` |
| `--max-ocr-concurrent` | (Batch) Max concurrent OCR submissions | `20` |
| `--max-llm-concurrent` | (Batch) Max concurrent LLM threads | `4` |
| `--batch-db` | (Batch) SQLite state DB path | `<input>/.knowmat_batch.db` |
| `--ocr-poll-interval` | (Batch) OCR poll interval (seconds) | `10` |

### Python API

```python
from knowmat.orchestrator import run
import os

result = run(
    pdf_path="path/to/paper.pdf",  # Also supports .txt / .md
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

## Output Structure

### Input Directory (`data/raw/`)

```
data/raw/
├── <PaperName>.pdf
└── <PaperName>/
    ├── <PaperName>.md                       # OCR output
    ├── <PaperName>.json                     # OCR structured data
    ├── paddleocrvl_parse/                   # (if --save-intermediate)
    │   ├── page_images/
    │   └── ocr_raw/
    └── _ocr_cache/                          # OCR cache
```

### Output Directory (`data/output/`)

```
data/output/
└── <PaperName>/
    ├── <PaperName>_extraction.json          # Final structured result
    ├── <PaperName>_analysis_report.txt       # Human-readable analysis
    ├── <PaperName>_runs.json                 # Multi-round extraction details
    └── <PaperName>_qa_report.json            # Quality & review flags
```

### Example Extraction Output

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

## Project Structure

```
KnowMat/
├── src/knowmat/              # Main Python package
│   ├── __main__.py           # CLI entry point
│   ├── orchestrator.py       # LangGraph orchestration
│   ├── nodes/                # LangGraph nodes
│   │   ├── paddleocrvl_parse_pdf.py
│   │   ├── extraction.py
│   │   ├── evaluation.py
│   │   └── ...
│   ├── pdf/                  # PDF/OCR submodule
│   │   ├── ocr_engine.py
│   │   └── ...
│   └── batch/                # Batch parallel processing
│       ├── batch_runner.py   # Asyncio orchestrator
│       ├── task_store.py     # SQLite state persistence
│       ├── key_pool.py       # Multi-key rotation
│       └── ocr_dispatcher.py # Async OCR lifecycle
├── scripts/                  # Utility scripts
│   └── download_paddleocrvl_models.py
├── prompts/                  # LLM prompt templates
├── configs/                  # Configuration directory
├── data/                     # Data directories
│   ├── raw/                  # Input files + OCR output
│   └── output/               # Extraction results
├── models/                   # OCR model weights (gitignored)
├── environment.yml           # Conda environment
├── requirements*.txt         # pip dependencies
└── .env.example              # Environment template
```

---

## Key Features

### Multi-Agent Architecture

- **Parser Agent**: Dual-engine PDF parsing with PaddleOCR-VL + PP-StructureV3
- **Subfield Detection Agent**: Identifies paper type (experimental/computational/ML)
- **Extraction Agent**: Structured data extraction with TrustCall
- **Evaluation Agent**: Quality assessment with confidence scoring
- **Two-Stage Manager**:
  - Stage 1 (Aggregation): Fast rule-based merging
  - Stage 2 (Validation): LLM hallucination detection & correction
- **Flagging Agent**: Final quality assessment & human review suggestions

### Data Extraction Coverage

- Material compositions (elements, stoichiometry, normalized formulas)
- Processing conditions (temperature, pressure, atmosphere, time)
- Characterization methods & results
- Material properties (ML-friendly formats):
  - Exact values, ranges, bounds (`>`, `<`, `>=`, `<=`)
  - Value types: `exact`, `lower_bound`, `upper_bound`, `range`, `qualitative`

### Property Standardization

Auto-maps attribute names to standard forms:
- `"glass transition temp"` → `"glass transition temperature"`
- `"ultimate tensile strength"` → `"tensile strength"`
- `"Young's modulus"` → `"elastic modulus"`

---

## Regression Testing

KnowMat includes a regression testing tool for AI vs Ground Truth comparison:

```bash
# GT mode: Compare AI extractions against ground truth
python tools/regression_diff.py gt --all

# Self mode: Compare two AI runs
python tools/regression_diff.py self --snapshot baseline
python tools/regression_diff.py self --compare baseline

# QA mode: Quality baseline check
python tools/regression_diff.py qa
```

For details, see [tools/README.md](tools/README.md).

---

## Troubleshooting

### API Key Not Set

```
Error: LLM_API_KEY not set
```

Solution: Ensure `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` are set in `.env`.

### 401 Invalid Model

```
Error code: 401 - invalid_model
```

Solution: Check `LLM_MODEL` in `.env`. For Qianfan, use the **inference endpoint ID** (e.g., `ep_xxxxx`), not the model name.

### PaddleOCR-VL Parsing Failed

```
Error: Failed to parse PDF with PaddleOCR-VL
```

Solutions:
- Check if PDF is corrupted or encrypted
- For Windows: Install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- For GPU: Ensure CUDA/cuDNN matches Paddle version (see [docs/platforms.md](docs/platforms.md))
- Set `KNOWMAT_ALLOW_LEGACY_PADDLEOCR=1` to allow fallback to classic OCR

---

## Citation

If KnowMat aids your research, please cite:

```bibtex
@software{knowmat2024,
  title = {KnowMat: Agentic Pipeline for Materials Science Data Extraction},
  author = {Sayeed, Hasan},
  year = {2024},
  url = {https://github.com/hasan-sayeed/KnowMat2}
}
```

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT License. See [LICENSE.txt](LICENSE.txt).
