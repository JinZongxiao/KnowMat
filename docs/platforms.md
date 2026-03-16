# 多平台与硬件适配说明

KnowMat 支持 **Windows**、**Linux**、**macOS**。**基础环境与 GPU 已分开配置**：默认仅装 CPU 版 Paddle，需 GPU 时再单独安装。

---

## 1. 基础环境 vs GPU（分开配置）

| 层级 | 内容 | 适用 |
|------|------|------|
| **基础（KnowMat）** | `environment.yml` / `requirements.txt`：CPU 版 `paddlepaddle`、`paddleocr[all]` 等 | 所有平台，`conda env create -f environment.yml` 即可 |
| **GPU 可选** | `requirements-gpu.txt` 或一键脚本中的「安装 GPU 版」步骤：`paddlepaddle-gpu` (cu129) + cuDNN | 仅 Windows/Linux + NVIDIA 显卡 |

这样 **Mac / 无 GPU 机器** 直接按 yml 或 requirements.txt 装，不会碰 GPU 包；**有 GPU** 的再用脚本或 `pip install -r requirements-gpu.txt -i ...` 叠加。

---

## 2. 平台与脚本对应

| 平台 | 推荐脚本 | Paddle 默认 | 说明 |
|------|----------|-------------|------|
| **Windows** | `scripts\setup_env.ps1` | GPU (cu129) | 需 NVIDIA 显卡 + CUDA；可选 `-CPU` 仅装 CPU 版 |
| **Linux** | `scripts/setup_env.sh` | GPU (cu129) | 有 NVIDIA 显卡时用 GPU；无则加 `--cpu` |
| **macOS (Intel / Apple Silicon)** | `scripts/setup_env.sh` | **CPU** | 无 NVIDIA GPU，脚本自动用 CPU |

---

## 3. Windows + NVIDIA GPU（含 4060、CUDA 12.7）

- **当前配置**：脚本使用 **cu129** 源安装 `paddlepaddle-gpu==3.3.0`。
- **CUDA 12.7 兼容性**：Paddle 官方提供 cu118、cu123、cu126、**cu129** 等预编译包，**无单独 cu127**。  
  **cu129 包可在 CUDA 12.7 驱动下正常使用**（CUDA 12.x 向后兼容），因此你本机 **4060 + CUDA 12.7** 用 cu129 即可，无需改脚本。
- **若安装或运行报错**：
  - 确认已安装 [Visual C++ Redistributable](https://learn.microsoft.com/zh-cn/cpp/windows/latest-supported-vc-redist)。
  - GPU OCR 需 cuDNN 9：`conda install nvidia::cudnn cuda-version=12`，见 [ocr-cudnn64_9-fix.md](ocr-cudnn64_9-fix.md)。
  - 仅用 CPU 跑 OCR：`.\scripts\setup_env.ps1 -CPU` 或设置 `KNOWMAT_OCR_DEVICE=cpu`。

---

## 4. Linux

- 有 **NVIDIA GPU**：直接运行 `./scripts/setup_env.sh`，会安装 GPU 版 (cu129) 与 cuDNN；CUDA 12.x 驱动下与 Windows 同理，用 cu129 即可。
- **无 GPU 或仅用 CPU**：`./scripts/setup_env.sh --cpu`。
- 路径与脚本均为 Linux 风格（`/`），无额外修改。

---

## 5. macOS

- **Intel / Apple Silicon 均无 NVIDIA GPU**，只能使用 **CPU 版 Paddle**。
- **推荐**：`./scripts/setup_env.sh`（脚本会检测 `Darwin` 并自动使用 CPU，无需加 `--cpu`）。
- 若需 M 系列芯片上加速，请按 PaddleOCR 官方最新 MLX-VLM 指南单独配置。

---

## 6. environment.yml / requirements 说明

- **environment.yml**、**requirements.txt**：仅含 **CPU 版** Paddle（`paddlepaddle>=3.0.0`），跨平台通用，无 GPU 依赖。
- **requirements-gpu.txt**：仅 GPU 相关（`paddlepaddle-gpu==3.3.0` + `paddleocr[all]`），需配合国内源与 cuDNN 使用，见文件内注释。
- **cuDNN**：仅 Windows/Linux GPU 需要，由脚本或 `conda install nvidia::cudnn cuda-version=12` 安装，不写入 yml。

---

## 7. 小结

| 你的环境 | 建议 |
|----------|------|
| Windows，4060，CUDA 12.7 | 使用 `setup_env.ps1`，保持 cu129；cu129 兼容 12.7，无需改。 |
| Windows，无 GPU / 仅 CPU | `setup_env.ps1 -CPU` |
| Linux，有 NVIDIA GPU | `setup_env.sh`，cu129 适用于 CUDA 12.x |
| Linux，无 GPU | `setup_env.sh --cpu` |
| macOS（任意） | `setup_env.sh`（自动 CPU） |
