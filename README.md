# KnowMat: Agentic Pipeline for Materials Science Data Extraction

![KnowMat-logo](docs/_static/KnowMat-logo.jpg)

_KnowMat agentic pipeline for extracting structured materials data from scientific literature._

---

## Overview

KnowMat is an AI-driven, agentic pipeline that automatically extracts structured, machine-readable materials science data from unstructured scientific documents (`.pdf` / `.txt`). Built on **LangGraph** with support for **OpenAI-compatible LLM APIs** (including ERNIE/Qianfan), it coordinates multiple intelligent agents to perform paper parsing, composition extraction, processing condition extraction, characterization extraction, and material properties extraction.

### Core Capabilities

- **Research-grade batch processing**: Process entire directories of PDF/TXT files; supports **two-stage** workflow: run OCR only (`--ocr-only`) first, then batch LLM extraction
- **High accuracy**: Multi-agent architecture with up to 3 rounds of extraction/evaluation iteration
- **Dual-engine OCR**: PaddleOCR-VL 1.5 (layout + reading order) + PP-StructureV3 (complex tables & formulas); optional MinerU cloud API mode (`--mineru-api`)
- **Formula & table enhancement**: Precise HTML table extraction and high-fidelity LaTeX formulas (auto-fixes chemical subscripts)
- **Two-stage validation**: Rule aggregation + LLM hallucination correction
- **Property standardization**: Auto-mapping attribute names to standard forms
- **Quality assurance**: Confidence scoring, human review flags & guidelines
- **ML-ready output**: Structured JSON for database入库 and modeling

---

## Installation

### Prerequisites

1. **Python 3.11**
2. **OpenAI-compatible LLM API Key** (e.g., ERNIE/Qianfan)
3. **LangChain API Key** (optional, for LangSmith tracing)

### Step 1: Clone the Repository

```bash
git clone https://github.com/shiyuasuka/KnowMat.git
cd KnowMat
```

### Step 2: Install Environment (Select Your Platform)

---

#### macOS (Recommended: pip)

macOS has no NVIDIA GPU, uses CPU mode.

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install project and dependencies
pip install -e .
pip install -r requirements.txt

# Download OCR models
python scripts/download_paddleocrvl_models.py --model-dir models/paddleocrvl1_5
```

---

#### Windows / Linux (Recommended: Conda)

**GPU mode (NVIDIA GPU):**

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate KnowMat

# Install Paddle GPU dependencies
pip uninstall -y paddlepaddle paddlepaddle-gpu
pip install -r requirements-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
conda install nvidia::cudnn cuda-version=12 -y

# Download OCR models
python scripts/download_paddleocrvl_models.py --model-dir models/paddleocrvl1_5
```

**CPU mode (no NVIDIA GPU):**

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate KnowMat

# Install Paddle CPU dependencies
pip install -r requirements-cpu.txt

# Download OCR models
python scripts/download_paddleocrvl_models.py --model-dir models/paddleocrvl1_5
```

---

### Step 3: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your API credentials:

```bash
# LLM API Configuration
LLM_API_KEY="your_llm_api_key"
LLM_BASE_URL="https://your-openai-compatible-endpoint.com/v2"
LLM_MODEL="your_model_name"

# PaddleOCR-VL Configuration
PADDLEOCRVL_MODEL_DIR=models/paddleocrvl1_5
PADDLEOCRVL_VERSION=1.5

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
| `requirements-gpu.txt` | GPU Paddle dependencies (NVIDIA) |
| `requirements-cpu.txt` | CPU Paddle dependencies |
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
| `--skip-cached-ocr` | Ignore OCR cache, force re-inference | `False` |
| `--force-rerun` | Force re-OCR and re-extraction | `False` |
| `--enable-property-standardization` | Enable property name standardization | `False` |
| `--subfield-model` | Subfield detection model | `LLM_MODEL` |
| `--extraction-model` | Extraction model | `LLM_MODEL` |
| `--evaluation-model` | Evaluation model | `LLM_MODEL` |
| `--manager-model` | Two-stage manager model | `LLM_MODEL` |
| `--flagging-model` | Flagging model | `LLM_MODEL` |

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
│   └── pdf/                  # PDF/OCR submodule
│       ├── ocr_engine.py
│       └── ...
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
