# AnyHand: A Large-Scale Synthetic Dataset for RGB(-D) Hand Pose Estimation

[Chen Si](https://chen-si-cs.github.io)<sup>1</sup> &emsp; [Yulin Liu](https://liuyulinn.github.io/)<sup>1</sup> &emsp; [Bo Ai](https://albertboai.com/)<sup>1</sup> &emsp; [Jianwen Xie](http://www.stat.ucla.edu/~jxie/)<sup>2</sup> &emsp; [Rolandos Alexandros Potamias](https://rolpotamias.github.io)<sup>3</sup> &emsp; [Chuanxia Zheng](https://physicalvision.github.io/people/~chuanxia)<sup>4</sup> &emsp; [Hao Su](https://cseweb.ucsd.edu/~haosu/)<sup>1</sup>

<sup>1</sup>UC San Diego &emsp; <sup>2</sup>Lambda, Inc &emsp; <sup>3</sup>Imperial College London &emsp; <sup>4</sup>Nanyang Technological University

<a href='https://chen-si-cs.github.io/projects/AnyHand'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/XXXX.XXXXX'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/YOUR_HF_USERNAME/AnyHand'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-yellow'></a>
<a href='https://huggingface.co/spaces/YOUR_HF_USERNAME/AnyHand'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-orange'></a>
<a href='https://colab.research.google.com/YOUR_COLAB_LINK'><img src='https://colab.research.google.com/assets/colab-badge.svg'></a>

![teaser](assets/teaser.png)

---

## Overview

**AnyHand** is a large-scale synthetic RGB-D dataset for 3D hand pose estimation, containing **2.5M single-hand** and **4.1M hand-object interaction** images with full geometric annotations (RGB, depth, mask, 3D pose/shape, camera intrinsics).

This repository releases **fine-tuned checkpoints of [HaMeR](https://arxiv.org/abs/2312.05251) and [WiLoR](https://arxiv.org/abs/2409.12259) co-trained with AnyHand**, which achieve consistent improvements on standard benchmarks (FreiHAND, HO-3D) and better generalization to out-of-domain scenes. More components are coming — see the roadmap below.

---

## 🗺️ Roadmap

| Component | Status |
|---|---|
| **Fine-tuned HaMeR & WiLoR checkpoints** (RGB hand pose estimation) | ✅ Released |
| **AnyHandNet-D** (RGB-D hand pose estimation with depth fusion module) | 🔜 Coming soon |
| **AnyHand generation pipeline** (full dataset & training code) | 🔜 Coming soon |

---

## Part 1 — RGB Hand Pose Estimation (HaMeR & WiLoR + AnyHand)

We release improved checkpoints for both [HaMeR](https://github.com/geopavlakos/hamer) and [WiLoR](https://github.com/rolpotamias/WiLoR), co-trained with AnyHand. These are **drop-in replacements** for the original checkpoints — no architecture changes are needed.

### 1.1 Clone This Repo (with WiLoR Submodule)

The WiLoR codebase is included as a git submodule.

```bash
git clone --recurse-submodules https://github.com/chen-si-cs/AnyHand.git
cd AnyHand
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 1.2 Install Dependencies

```bash
conda create -n anyhand python=3.10 -y
conda activate anyhand
```

Install PyTorch (adjust CUDA version — see [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install WiLoR's dependencies:

```bash
pip install -r WiLoR/requirements.txt
```

### 1.3 Set Up MANO

WiLoR requires the MANO hand model, which must be downloaded manually due to its license.

1. Register and download from the [MANO website](https://mano.is.tue.mpg.de/)
2. Unzip and place the right hand model at:

```
AnyHand/
└── mano_data/
    └── MANO_RIGHT.pkl
```

> **Note:** By using MANO, you agree to the [MANO license terms](https://mano.is.tue.mpg.de/license.html).

### 1.4 Download AnyHand Checkpoints

The improved checkpoints and matching model configs are hosted on HuggingFace.

#### Option A — `wget` (quick)

```bash
mkdir -p pretrained_models

# AnyHand fine-tuned WiLoR checkpoint
wget https://huggingface.co/<YOUR_HF_USERNAME>/AnyHand/resolve/main/anyhand_wilor.ckpt \
     -O pretrained_models/anyhand_wilor.ckpt

# Matching WiLoR model config
wget https://huggingface.co/<YOUR_HF_USERNAME>/AnyHand/resolve/main/model_config.yaml \
     -O pretrained_models/model_config.yaml

# Hand detector (unchanged from original WiLoR)
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt \
     -O pretrained_models/detector.pt
```

#### Option B — `huggingface_hub` (Python, handles large files & retries)

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
import shutil, os

os.makedirs("pretrained_models", exist_ok=True)

for filename in ["anyhand_wilor.ckpt", "model_config.yaml"]:
    path = hf_hub_download(
        repo_id="<YOUR_HF_USERNAME>/AnyHand",
        filename=filename,
    )
    shutil.copy(path, f"pretrained_models/{filename}")
```

After downloading, `pretrained_models/` should look like:

```
pretrained_models/
├── anyhand_wilor.ckpt   ← AnyHand fine-tuned WiLoR checkpoint
├── model_config.yaml    ← matching model config
└── detector.pt          ← hand detector (from original WiLoR)
```

### 1.5 Run Inference

#### Demo on a folder of images

```bash
python WiLoR/demo.py \
    --img_folder demo_img \
    --out_folder demo_out \
    --checkpoint pretrained_models/anyhand_wilor.ckpt \
    --cfg pretrained_models/model_config.yaml \
    --save_mesh
```

#### Interactive Gradio demo

```bash
python WiLoR/gradio_demo.py \
    --checkpoint pretrained_models/anyhand_wilor.ckpt \
    --cfg pretrained_models/model_config.yaml
```

> **Note:** If the WiLoR scripts do not expose `--checkpoint` / `--cfg` flags, edit the `load_wilor(...)` call at the top of each script:
> ```python
> # Replace:
> model, model_cfg = load_wilor(
>     checkpoint_path='./pretrained_models/wilor_final.ckpt',
>     cfg_path='./pretrained_models/model_config.yaml'
> )
> # With:
> model, model_cfg = load_wilor(
>     checkpoint_path='../pretrained_models/anyhand_wilor.ckpt',
>     cfg_path='../pretrained_models/model_config.yaml'
> )
> ```

---

## Part 2 — RGB-D Hand Pose Estimation (AnyHandNet-D) 🔜

> **Coming soon.** We will release AnyHandNet-D, a lightweight depth fusion module that integrates into existing RGB-based models. Trained with AnyHand's aligned RGB-D data, it achieves superior performance on HO-3D without any fine-tuning on target data.

---

## Part 3 — AnyHand Generation Pipeline 🔜

> **Coming soon.** We will release the full AnyHand dataset, generation pipeline, and training code. The dataset includes 2.5M single-hand and 4.1M hand-object interaction images rendered with diverse hand poses, skin textures, lighting conditions, and backgrounds.

---


---

## License

- **AnyHand checkpoints**: [CC-BY-NC-ND](LICENSE)
- **WiLoR codebase**: [CC-BY-NC-ND](https://github.com/rolpotamias/WiLoR)
- **MANO model**: [MANO license](https://mano.is.tue.mpg.de/license.html)
- **Detector** (`detector.pt`): [Ultralytics license](https://github.com/ultralytics/ultralytics)

---

## Citation

If you find AnyHand useful, please cite our work and the baselines:

```bibtex
@misc{si2026anyhand,
  title         = {AnyHand: A Large-Scale Synthetic Dataset for RGB(-D) Hand Pose Estimation},
  author        = {Si, Chen and Liu, Yulin and Ai, Bo and Xie, Jianwen and
                   Potamias, Rolandos Alexandros and Zheng, Chuanxia and Su, Hao},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@misc{potamias2024wilor,
    title={WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild},
    author={Rolandos Alexandros Potamias and Jinglei Zhang and Jiankang Deng and Stefanos Zafeiriou},
    year={2024},
    eprint={2409.12259},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@inproceedings{pavlakos2024reconstructing,
    title={Reconstructing Hands in 3D with Transformers},
    author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
    booktitle={CVPR},
    year={2024}
}
```
