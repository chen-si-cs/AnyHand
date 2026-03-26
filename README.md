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
| **Fine-tuned HaMeR & WiLoR checkpoints** + unified `AnyHandPredictor` | ✅ Released |
| **AnyHandNet-D**  | 🔜 Coming soon |
| **AnyHand generation pipeline** | 🔜 Coming soon |
| **AnyHand dataset** | 🔜 Coming soon |

---

## Part 1 — RGB Hand Pose Estimation (HaMeR & WiLoR + AnyHand)

We release improved checkpoints for both [HaMeR](https://github.com/geopavlakos/hamer) and [WiLoR](https://github.com/rolpotamias/WiLoR), co-trained with AnyHand. These are **drop-in replacements** for the original checkpoints — no architecture changes are needed.

### 1.1 Clone This Repo (with Submodules)

Both WiLoR and HaMeR are included as git submodules.

```bash
git clone --recurse-submodules https://github.com/chen-si-cs/AnyHand.git
cd AnyHand
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

This populates WiLoR/ (WiLoR codebase) and third_party/hamer/ (HaMeR codebase).

### 1.2 Install Dependencies

```bash
conda create -n anyhand python=3.10 -y
conda activate anyhand
```

Install PyTorch (adjust CUDA version — see [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then run the preparation scripts for whichever backend(s) you need:
**WiLoR only** (recommended for most users):
```
bash scripts/prepare_wilor.sh
```

**HaMeR only**:
```
bash scripts/prepare_hamer.sh
```

**Both**:
```
bash scripts/prepare_wilor.sh
bash scripts/prepare_hamer.sh
```
Each script installs the corresponding Python package, downloads the
AnyHand checkpoint, and prints a checklist of remaining manual steps.

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

### 1.4 Download Checkpoints
The prepare scripts above handle this automatically once you fill in
your HuggingFace username. See the scripts for manual `wget` alternatives.
After running, your layout will be:
```
pretrained_models/
├── anyhand_wilor.ckpt          ← AnyHand fine-tuned WiLoR
├── model_config_wilor.yaml     ← WiLoR config
├── detector.pt                 ← YOLO hand detector (shared by both)
└── hamer_ckpts/
    └── checkpoints/
        ├── anyhand_hamer.ckpt  ← AnyHand fine-tuned HaMeR
        └── model_config.yaml   ← HaMeR config
```

### 1.5 Run Inference — Unified Predictor

We provide `AnyHandPredictor`, a single class that wraps both models behind one consistent API. It always uses WiLoR's YOLO hand detector for bbox detection, then dispatches to whichever reconstruction backbone you choose.

**Predict**

```python
from anyhand import AnyHandPredictor

# WiLoR backend (default)
predictor = AnyHandPredictor(backend='wilor')
hands = predictor.predict('path/to/image.jpg')

for hand in hands:
    print(f"{'Right' if hand.is_right else 'Left'} hand  score={hand.score:.2f}")
    print(f"  MANO pose  : {hand.mano_pose.shape}")    # (48,)
    print(f"  MANO shape : {hand.mano_shape.shape}")   # (10,)
    print(f"  Vertices   : {hand.vertices.shape}")     # (778, 3)
    print(f"  Keypoints3D: {hand.keypoints_3d.shape}") # (21, 3)
    print(f"  Keypoints2D: {hand.keypoints_2d.shape}") # (21, 2)
    print(f"  Cam translation : {hand.cam_t}")              # (3,)

# HaMeR backend
predictor = AnyHandPredictor(backend='hamer')
hands = predictor.predict('path/to/image.jpg')

# Both at once — same bboxes, two sets of predictions
# hand.backend == 'wilor' or 'hamer' tells you which is which
predictor = AnyHandPredictor(backend='both')
hands = predictor.predict('path/to/image.jpg')

# Batch of images
import cv2
imgs = [cv2.imread(p) for p in ['img1.jpg', 'img2.jpg']]
batch_results = predictor.predict(imgs)  # List[List[HandPrediction]]

# Override backend per call
hands = predictor.predict('photo.jpg', backend='wilor')
```

**Render mesh overlay**

```python
# Renders all detected hand meshes overlaid on the image.
# Returns a BGR uint8 numpy array ready for cv2.imwrite().
overlay = predictor.render_overlay('photo.jpg', hands)
cv2.imwrite('out.jpg', overlay)

# Custom mesh colour (float RGB in [0, 1])
overlay = predictor.render_overlay('photo.jpg', hands, mesh_color=(0.9, 0.4, 0.2))
```

**Save per-hand .obj meshes**

```python
# Saves <prefix>_0.obj, <prefix>_1.obj, … into out_dir/.
# Returns the list of absolute paths written.
paths = predictor.save_meshes(hands, out_dir='out/meshes', prefix='frame0042')
print(paths)
# ['…/out/meshes/frame0042_0.obj', '…/out/meshes/frame0042_1.obj']
```

**Project 3D points to 2D pixels**

```python
img = cv2.imread('photo.jpg')

for hand in hands:
    # Works on vertices (778, 3) or keypoints (21, 3) — any (N, 3) array
    kpts_2d = AnyHandPredictor.project_3d_to_2d(
        hand.keypoints_3d,
        hand.cam_t,
        hand.focal_length,
        img_size=(img.shape[1], img.shape[0]),
    )  # (21, 2) float32 pixel coordinates

    for x, y in kpts_2d.astype(int):
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)

cv2.imwrite('keypoints.jpg', img)
```

**Custom checkpoint paths**

```python
predictor = AnyHandPredictor(
    backend        = 'wilor',
    wilor_ckpt     = '/path/to/anyhand_wilor.ckpt',
    wilor_cfg      = '/path/to/model_config_wilor.yaml',
    detector_pt    = '/path/to/detector.pt',
    device         = 'cuda:0',
    det_conf       = 0.4,   # YOLO detection confidence threshold
    rescale_factor = 2.0,   # bbox padding factor
    batch_size     = 32,
)
```

**Command-line demo (WiLoR's original script, with AnyHand checkpoint)**

```bash
python WiLoR/demo.py \
    --img_folder demo_img \
    --out_folder demo_out \
    --checkpoint pretrained_models/anyhand_wilor.ckpt \
    --cfg        pretrained_models/model_config_wilor.yaml \
    --save_mesh
```

---

## Part 2 — RGB-D Hand Pose Estimation (AnyHandNet-D) 🔜

> **Coming soon.** 

---

## Part 3 — AnyHand Generation Pipeline 🔜

> **Coming soon.** 

---

## Part 4 — AnyHand Dataset 🔜

> **Coming soon.** 

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
