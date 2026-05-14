# sci-align 集成使用手册

本文档说明 sci-align 图像-文本对齐功能在 KnowMat 中的集成方式、配置参数和使用方法。

---

## 目录

1. [安装](#安装)
2. [功能概述](#功能概述)
3. [核心原理](#核心原理)
4. [模块结构](#模块结构)
5. [两种使用模式](#两种使用模式)
6. [模式一：KnowMat 流水线集成](#模式一knowmat-流水线集成)
7. [模式二：独立数据集构建](#模式二独立数据集构建)
8. [全部配置参数](#全部配置参数)
9. [输出格式详解](#输出格式详解)
10. [与原始 sci-align 的差异](#与原始-sci-align-的差异)

---

## 安装

### 前置要求

- Python **3.11+**
- 如需 GPU 加速：CUDA 12.x + 对应版本 PyTorch

### 第一步：克隆并安装 KnowMat

```bash
git clone https://github.com/shiyuasuka/KnowMat.git
cd KnowMat
pip install -e .
pip install -r requirements.txt
```

> 使用 Conda 的话：
> ```bash
> conda env create -f environment.yml
> conda activate KnowMat
> ```

### 第二步：安装 CLIP 依赖

图像-文本对齐使用 HuggingFace CLIP（`openai/clip-vit-base-patch32`），需要额外安装：

```bash
pip install torch torchvision transformers huggingface_hub pillow
```

GPU 版（CUDA 12.x）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers huggingface_hub pillow
```

### 第三步：配置环境变量

在项目根目录新建 `.env` 文件（参考 `.env.example`），至少填写：

```ini
# LLM API（KnowMat 主流程需要，对齐模块本身不需要）
LLM_API_KEY=your_key_here
LLM_BASE_URL=https://your-endpoint/v1

# 图像-文本对齐（以下为默认值，可按需修改）
KNOWMAT2_ALIGNMENT_ENABLED=true
KNOWMAT2_ALIGNMENT_MODEL=clip
KNOWMAT2_ALIGNMENT_DEVICE=cpu        # 有 GPU 改为 cuda
KNOWMAT2_ALIGNMENT_TOP_K=5
KNOWMAT2_ALIGNMENT_CAPTION_BLEND=0.3
KNOWMAT2_ALIGNMENT_SAVE_DATASET=false
```

### 第四步：验证安装

```bash
python -c "from knowmat.image_text_alignment import ImageTextAligner; print('OK')"
```

输出 `OK` 即安装成功。也可运行对齐逻辑验证：

```bash
python tests/test_alignment/test_vs_scialign.py
# 预期：SUMMARY (17 images)  Rank-1 exact match: 17/17 (100%)
```

---

## 功能概述

sci-align 解决的问题：科学论文中的图片（Figure 1、Figure 2…）与正文描述之间的自动匹配。

给定一篇论文的 OCR 结果，该模块会：

1. 提取所有**图片**（含图注）→ `VisualToken`
2. 提取所有**正文句子**（含表格标题）→ `SentenceToken`
3. 用 CLIP 对图片和句子分别编码，计算余弦相似度
4. 结合**图号锚点**（句子中提到 Fig. 3 → 与 Figure 3 强匹配）和**页码邻近**奖励进行重排序
5. 为每张图片输出 Top-K 最相关句子，附带置信度标签

---

## 核心原理

### 相似度计算

```
final_score = cosine(img_vec, sent_vec) + bonus
```

| 情况 | bonus |
|------|-------|
| 句子中提到了该图号（Fig. 3 ↔ Figure 3） | **+0.20**（图号锚点奖励） |
| 句子和图片在相邻页（±1 页以内） | **+0.05**（页码邻近奖励） |
| 句子提到了别的图号但不是该图 | **−0.15**（错误锚点惩罚） |

### 置信度标签

| 标签 | 条件 |
|------|------|
| `high` | 命中图号锚点，或 final_score ≥ 0.6 |
| `medium` | final_score ∈ [0.4, 0.6) |
| `low` | final_score < 0.4 |

### caption_blend（可选增强）

将图片向量与图注文本向量线性混合，改善视觉内容模糊的图片的匹配效果：

```
img_vec = (1 - α) × img_vec + α × caption_text_vec
img_vec = L2_normalize(img_vec)
```

推荐值 `α = 0.3`（70% 图像 + 30% 图注语义）。

---

## 模块结构

```
src/knowmat/image_text_alignment/
├── __init__.py              # 公开接口：ImageTextAligner、OcrItemTokenizer 等
├── aligner.py               # 核心：embedding + 相似度计算 + 重排序 + 数据集导出
├── tokenizer.py             # OcrItemTokenizer：ocr_items → VisualToken + SentenceToken
├── config.py                # AlignmentConfig 配置类
├── exporter.py              # 转换为 KnowMat extraction.json 格式
├── dataset_builder.py       # DatasetBuilder：从 MinerU 输出目录批量构建数据集
├── build_dataset.py         # CLI 入口（python -m ...build_dataset）
└── embeddings/
    ├── __init__.py          # 注册表：register_embedding / get_embedding
    ├── base.py              # EmbeddingAdapter 抽象基类
    └── clip_adapter.py      # CLIP 实现（openai/clip-vit-base-patch32）

src/knowmat/nodes/
└── image_text_alignment.py  # LangGraph 节点：align_images_with_text

tests/test_alignment/
├── test_vs_scialign.py      # 与 sci-align baseline 对比（17/17 rank-1 完全匹配）
└── test_mineru_pipeline.py  # 完整流水线测试（MinerU API → tokenize → align → 对比）
```

---

## 两种使用模式

| 模式 | 适用场景 | 入口 |
|------|----------|------|
| **流水线模式** | KnowMat LangGraph 全流程（PDF → 结构化抽取） | `align_images_with_text` 节点自动执行 |
| **数据集模式** | 批量构建 sci-align 格式对齐数据集 | `DatasetBuilder` 或 CLI |

---

## 模式一：KnowMat 流水线集成

### 工作位置

图像-文本对齐作为 KnowMat LangGraph 流水线的第二个节点自动执行：

```
START
  └─▶ parse_pdf
        └─▶ align_images_with_text   ← sci-align 对齐在此发生
              └─▶ detect_sub_field
                    └─▶ extract_data
                          └─▶ ... → END
```

### 开启方式

在项目根目录的 `.env` 文件中添加：

```ini
# 启用图像-文本对齐（默认已启用，可省略）
KNOWMAT2_ALIGNMENT_ENABLED=true

# Embedding 模型（目前支持 clip）
KNOWMAT2_ALIGNMENT_MODEL=clip

# 运算设备：cpu 或 cuda
KNOWMAT2_ALIGNMENT_DEVICE=cpu

# 每张图片返回 Top-K 最相关句子
KNOWMAT2_ALIGNMENT_TOP_K=5

# caption_blend：混合图注语义，推荐 0.3，0 表示关闭
KNOWMAT2_ALIGNMENT_CAPTION_BLEND=0.3

# 同时将对齐结果保存为 sci-align 格式数据集文件
KNOWMAT2_ALIGNMENT_SAVE_DATASET=true
```

### 流水线输出

对齐结果以 `Image_Text_Alignments` 字段嵌入 `extraction.json`：

```json
{
  "Paper_Metadata": { ... },
  "Image_Text_Alignments": [
    {
      "image_id": "10.1007_xxx::image_0000",
      "image_path": "/path/to/figures/fig1.png",
      "figure_num": "1",
      "caption": "Figure 1. SEM image of ...",
      "normalized_figure_id": "fig_1",
      "page_number": 3,
      "related_sentences": [
        {
          "rank": 1,
          "text": "As shown in Figure 1, the microstructure exhibits ...",
          "score": 0.7241,
          "cosine_score": 0.5241,
          "caption_text_cosine": 0.4812,
          "confidence": "high",
          "has_same_figure_anchor": true,
          "has_wrong_figure_anchor": false,
          "mentioned_figures": ["fig_1"],
          "source": "paragraph",
          "page": 3,
          "section": "Results"
        },
        ...
      ]
    }
  ]
}
```

### 关闭对齐

不需要对齐时直接在 `.env` 中设置：

```ini
KNOWMAT2_ALIGNMENT_ENABLED=false
```

---

## 模式二：独立数据集构建

与 sci-align 原始工具相同，输出完整的 JSONL + npy 数据集，可直接用于下游训练或评估。

### 方式 A：命令行

```bash
cd KnowMat
python -m knowmat.image_text_alignment.build_dataset \
    --input_dir  /path/to/mineru_outputs \
    --output_dir /path/to/my_dataset \
    --model      clip \
    --device     cpu \
    --top_k      5 \
    --caption_blend 0.3
```

`input_dir` 结构支持两种：

```
# 单篇论文（目录下直接有 *_content_list.json）
/mineru_outputs/
  └── 10.1007_xxx_content_list.json
  └── images/

# 多篇论文（每个子目录一篇论文）
/mineru_outputs/
  ├── paper_001/
  │   ├── paper_001_content_list.json
  │   └── images/
  └── paper_002/
      ├── paper_002_content_list.json
      └── images/
```

### 方式 B：Python 代码

```python
from knowmat.image_text_alignment.dataset_builder import DatasetBuilder, DatasetBuildConfig

cfg = DatasetBuildConfig(
    input_dir="/path/to/mineru_outputs",
    output_dir="/path/to/my_dataset",
    model="clip",
    device="cpu",
    top_k=5,
    caption_blend=0.3,
    save_embeddings=True,
)
result = DatasetBuilder(cfg).build()
print(result)
# DatasetBuildResult(papers=10, images=170, sentences=2700, direct=170, topk=850, elapsed=42.3s)
```

### 方式 C：传入已转换的 ocr_items

如果已经通过 `convert_mineru_to_knowmat` 转换好了 ocr_items，可直接传入：

```python
from knowmat.image_text_alignment.dataset_builder import DatasetBuilder, DatasetBuildConfig

result = DatasetBuilder(cfg).build_from_ocr_items({
    "paper_id_1": ocr_items_1,
    "paper_id_2": ocr_items_2,
})
```

### 方式 D：在 KnowMat 流水线中同时输出数据集

开启 `KNOWMAT2_ALIGNMENT_SAVE_DATASET=true` 后，流水线每处理一篇论文，会在 `output_dir` 下**同时**写入数据集文件和 `extraction.json`，embedding 只计算一次，不重复。

---

## 全部配置参数

### `.env` 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `KNOWMAT2_ALIGNMENT_ENABLED` | `true` | 是否启用图像-文本对齐 |
| `KNOWMAT2_ALIGNMENT_MODEL` | `clip` | Embedding 模型，目前支持 `clip` |
| `KNOWMAT2_ALIGNMENT_DEVICE` | `cpu` | 运算设备，`cpu` 或 `cuda` |
| `KNOWMAT2_ALIGNMENT_TOP_K` | `5` | 每张图片返回的最相关句子数量 |
| `KNOWMAT2_ALIGNMENT_MIN_SCORE` | `0.0` | final_score 低于此值的结果被过滤 |
| `KNOWMAT2_ALIGNMENT_BATCH_SIZE` | `32` | Embedding 批处理大小 |
| `KNOWMAT2_ALIGNMENT_CAPTION_BLEND` | `0.0` | 图注混合比例，推荐 `0.3`，`0` 关闭 |
| `KNOWMAT2_ALIGNMENT_SAVE_DATASET` | `false` | 流水线运行时同时输出 JSONL 数据集 |

### AlignmentConfig（代码中直接使用）

```python
from knowmat.image_text_alignment import AlignmentConfig

cfg = AlignmentConfig(
    enabled=True,
    model="clip",
    device="cuda",
    top_k=5,
    min_score=0.0,
    batch_size=32,
    caption_blend=0.3,
    save_dataset=True,
)
```

### DatasetBuildConfig（独立数据集模式）

```python
from knowmat.image_text_alignment.dataset_builder import DatasetBuildConfig

cfg = DatasetBuildConfig(
    input_dir="...",         # MinerU 输出根目录
    output_dir="...",        # 数据集输出目录
    model="clip",
    device="cpu",
    top_k=5,
    min_score=0.0,
    batch_size=32,
    caption_blend=0.3,
    save_embeddings=True,    # 是否保存 .npy 文件
)
```

---

## 输出格式详解

数据集模式（`DatasetBuilder` 或 `KNOWMAT2_ALIGNMENT_SAVE_DATASET=true`）在 `output_dir` 下生成以下文件：

```
output_dir/
├── image_units.jsonl         每张图片的元数据
├── sentence_units.jsonl      每条句子的元数据
├── image_embeddings.npy      图片 embedding 矩阵 [N_img, 512]，L2 归一化
├── sentence_embeddings.npy   句子 embedding 矩阵 [N_txt, 512]，L2 归一化
├── embedding_index.json      token_id ↔ embedding 行号的映射
├── direct_pairs.jsonl        图片 ↔ 自身图注，score=1.0（直接对）
├── topk_pairs.jsonl          图片 → Top-K 正文句子（重排序后）
├── image_topk.jsonl          同上，每行一张图片（兼容旧格式）
├── text_topk.jsonl           句子 → Top-K 图片（反向检索）
├── metadata.json             统计摘要
└── human_check.md            可读的对齐结果报告（每张图一节）
```

### topk_pairs.jsonl 字段说明

```jsonc
{
  "pair_type": "topk_similarity",
  "visual_token_id": "10.1007_xxx::image_0000",
  "visual_paper_id": "10.1007_xxx",
  "image_path": "/path/to/fig1.png",
  "normalized_figure_id": "fig_1",     // 标准化图号，如 fig_1、fig_3a
  "text_token_id": "10.1007_xxx::sent::42",
  "text": "As shown in Figure 1, ...",
  "source": "paragraph",               // paragraph / table_content / table_caption
  "rank": 1,
  "cosine_score": 0.5241,             // 原始 CLIP 余弦相似度
  "caption_text_cosine": 0.4812,      // 图注文本 ↔ 句子的余弦（诊断用）
  "final_score": 0.7241,              // 加权重排后的最终分数
  "score": 0.7241,                    // 同 final_score（兼容字段）
  "mentioned_figures": ["fig_1"],     // 句子中提到的图号列表
  "matched_figure_id": "fig_1",       // 命中锚点的图号（null 表示未命中）
  "has_same_figure_anchor": true,     // 句子提到了该图
  "has_wrong_figure_anchor": false,   // 句子提到了别的图
  "confidence": "high"                // high / medium / low
}
```

### human_check.md 样例

运行后可用 VSCode 或任意 Markdown 查看器打开，格式如下：

```markdown
## image_0000  `10.1007_xxx`
- **Image**: /path/to/fig1.png
- **Caption**: Figure 1. SEM image of EPDM/NBR blends.
- **Parsed figure ID**: `fig_1`
- **Page**: 3

| Rank | Cosine | CapTxt | Final | Conf   | same | wrong | Figures | Text |
|------|--------|--------|-------|--------|------|-------|---------|------|
| 1    | 0.5241 | 0.4812 | 0.7241| high   | ✓    | ✗     | fig_1   | As shown in Figure 1, ... |
| 2    | 0.3523 | 0.2901 | 0.4023| medium | ✗    | ✗     | -       | A scanning electron microscope ... |
```

---

## 与原始 sci-align 的差异

| 特性 | 原始 sci-align | KnowMat 集成版 |
|------|---------------|----------------|
| 输入格式 | `*_content_list.json`（MinerU 直接输出） | `ocr_items`（KnowMat 内部格式） |
| 流水线集成 | 独立运行 | 作为 LangGraph 节点嵌入 |
| 数据集输出 | 始终输出 | 可选（`save_dataset=True`） |
| `caption_blend` | ✓ | ✓ |
| `cosine_score` 字段 | ✓ | ✓（新增） |
| `caption_text_cosine` 字段 | ✓ | ✓（新增） |
| 重排序逻辑 | 与本版完全一致 | ✓（17/17 rank-1 完全匹配） |
| Embedding 后端 | CLIP / BiomedCLIP / CN-CLIP / SigLIP2 / ERNIE-ViL | 当前仅 CLIP（其余可按需移植） |
| 反向检索（句子 → 图片） | ✓ `text_topk.jsonl` | ✓ |
| `direct_pairs`（图注直接对） | ✓ | ✓ |

---

## 快速验证

### 验证对齐逻辑是否与 sci-align 一致

```bash
cd KnowMat
python tests/test_alignment/test_vs_scialign.py
# 预期输出：
# SUMMARY (17 images)
#   Rank-1 exact match : 17/17  (100%)
#   Top-5 text overlap: 85/85  (100%)
```

### 验证完整 MinerU → 对齐流水线

```bash
python tests/test_alignment/test_mineru_pipeline.py
# 需要在 .env 中设置 MINERU_API_KEY
```
