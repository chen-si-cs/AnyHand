"""
anyhand/predictor.py
====================
Unified hand pose predictor that:
  1. Runs WiLoR's YOLO hand detector to find bounding boxes.
  2. Dispatches to either AnyHand-WiLoR or AnyHand-HaMeR (or both)
     for 3D reconstruction.
  3. Returns structured HandPrediction objects.

Usage
-----
    from anyhand import AnyHandPredictor

    # Single backend
    predictor = AnyHandPredictor(backend='wilor')
    hands = predictor.predict('image.jpg')

    # Both backends, on a batch of pre-loaded BGR arrays
    predictor = AnyHandPredictor(backend='both')
    all_hands = predictor.predict([img_bgr_1, img_bgr_2])

    # Per-call backend override
    hands = predictor.predict(img_bgr, backend='hamer')
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve paths relative to the repo root (two levels above this file).
# ---------------------------------------------------------------------------
_THIS_DIR  = Path(__file__).resolve().parent          # anyhand/
_REPO_ROOT = _THIS_DIR.parent                         # AnyHand/


# ===========================================================================
# Output dataclass
# ===========================================================================

@dataclass
class HandPrediction:
    """
    All outputs for a single detected hand.

    Attributes
    ----------
    mano_pose : (48,) float32
        MANO pose parameters in axis-angle: 3 global orientation + 45 hand joints.
    mano_shape : (10,) float32
        MANO shape (beta) coefficients.
    vertices : (778, 3) float32
        MANO mesh vertices in camera-space metres.
    keypoints_3d : (21, 3) float32
        3D hand joints in camera-space metres.
    keypoints_2d : (21, 2) float32
        2D hand joints in original image pixel coordinates.
    cam_t : (3,) float32
        Camera translation [tx, ty, tz] in metres (full-image coordinate frame).
    focal_length : float
        Estimated focal length in pixels used for projection.
    bbox : (4,) float32
        Detected bounding box [x1, y1, x2, y2] in pixel coordinates.
    is_right : bool
        True if this is the right hand, False for left hand.
    score : float
        Detector confidence score.
    backend : str
        Which model produced this prediction: 'wilor' or 'hamer'.
    """
    mano_pose:    np.ndarray   # (48,)
    mano_shape:   np.ndarray   # (10,)
    vertices:     np.ndarray   # (778, 3)
    keypoints_3d: np.ndarray   # (21, 3)
    keypoints_2d: np.ndarray   # (21, 2)
    cam_t:        np.ndarray   # (3,)
    focal_length: float
    bbox:         np.ndarray   # (4,)
    is_right:     bool
    score:        float
    backend:      str


# ===========================================================================
# Main predictor
# ===========================================================================

class AnyHandPredictor:
    """
    Unified hand pose predictor using AnyHand fine-tuned checkpoints.

    Parameters
    ----------
    backend : 'wilor' | 'hamer' | 'both'
        Which reconstruction model(s) to load and use.
        - 'wilor'  — load only the WiLoR model (faster, lower memory).
        - 'hamer'  — load only the HaMeR model.
        - 'both'   — load both models; predict() returns predictions from
                     both, concatenated in the result list.
    device : str or None
        PyTorch device string. None = auto-detect CUDA, fall back to CPU.
    wilor_ckpt : str or None
        Path to anyhand_wilor.ckpt. None = use default location under
        pretrained_models/.
    wilor_cfg : str or None
        Path to model_config_wilor.yaml. None = use default location.
    hamer_ckpt : str or None
        Path to anyhand_hamer.ckpt. None = use default location under
        pretrained_models/hamer_ckpts/checkpoints/.
    detector_pt : str or None
        Path to WiLoR's detector.pt. None = use default location.
    det_conf : float
        YOLO detection confidence threshold (default 0.3).
    det_iou : float
        YOLO NMS IoU threshold (default 0.3).
    rescale_factor : float
        Padding factor applied around each detected bbox before cropping
        (default 2.0, matching original WiLoR/HaMeR demos).
    batch_size : int
        Number of hand crops processed per forward pass (default 16).
    """

    # ------------------------------------------------------------------
    # Default paths (relative to repo root)
    # ------------------------------------------------------------------
    _DEFAULT_WILOR_CKPT  = str(_REPO_ROOT / 'pretrained_models' / 'anyhand_wilor.ckpt')
    _DEFAULT_WILOR_CFG   = str(_REPO_ROOT / 'pretrained_models' / 'model_config_wilor.yaml')
    _DEFAULT_HAMER_CKPT  = str(_REPO_ROOT / 'pretrained_models' / 'hamer_ckpts' / 'checkpoints' / 'anyhand_hamer.ckpt')
    _DEFAULT_DETECTOR_PT = str(_REPO_ROOT / 'pretrained_models' / 'detector.pt')

    def __init__(
        self,
        backend: Literal['wilor', 'hamer', 'both'] = 'wilor',
        device: Optional[str] = None,
        wilor_ckpt: Optional[str] = None,
        wilor_cfg: Optional[str] = None,
        hamer_ckpt: Optional[str] = None,
        detector_pt: Optional[str] = None,
        det_conf: float = 0.3,
        det_iou: float = 0.3,
        rescale_factor: float = 2.0,
        batch_size: int = 16,
    ) -> None:
        if backend not in ('wilor', 'hamer', 'both'):
            raise ValueError(f"backend must be 'wilor', 'hamer', or 'both'. Got: {backend!r}")

        self.backend        = backend
        self.det_conf       = det_conf
        self.det_iou        = det_iou
        self.rescale_factor = rescale_factor
        self.batch_size     = batch_size

        # ---- device ----
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # ---- resolve paths ----
        self._wilor_ckpt  = wilor_ckpt  or self._DEFAULT_WILOR_CKPT
        self._wilor_cfg   = wilor_cfg   or self._DEFAULT_WILOR_CFG
        self._hamer_ckpt  = hamer_ckpt  or self._DEFAULT_HAMER_CKPT
        self._detector_pt = detector_pt or self._DEFAULT_DETECTOR_PT

        # ---- load components ----
        self._load_detector()

        self._wilor_model     = None
        self._wilor_model_cfg = None
        self._hamer_model     = None
        self._hamer_model_cfg = None

        if backend in ('wilor', 'both'):
            self._load_wilor()
        if backend in ('hamer', 'both'):
            self._load_hamer()

    # ==================================================================
    # Loading helpers
    # ==================================================================

    def _load_detector(self) -> None:
        """Load WiLoR's YOLO hand detector (used for all backends)."""
        from ultralytics import YOLO  # installed as part of WiLoR deps

        _check_file(self._detector_pt,
                    "Hand detector not found. Run:  bash scripts/prepare_wilor.sh")
        self._detector = YOLO(self._detector_pt).to(self.device)

    def _load_wilor(self) -> None:
        """Load the AnyHand-WiLoR model from the WiLoR submodule."""
        _ensure_on_path(_REPO_ROOT / 'WiLoR',
                        "WiLoR submodule not found. Run:  bash scripts/prepare_wilor.sh")
        try:
            from wilor.models.wilor import WiLoR  # noqa: F401 (import check)
        except ImportError as exc:
            raise ImportError(
                "WiLoR package is not installed. Run:  bash scripts/prepare_wilor.sh"
            ) from exc

        _check_file(self._wilor_ckpt,
                    "WiLoR checkpoint not found. Run:  bash scripts/prepare_wilor.sh")
        _check_file(self._wilor_cfg,
                    "WiLoR model config not found. Run:  bash scripts/prepare_wilor.sh")

        from omegaconf import OmegaConf
        from wilor.models.wilor import WiLoR

        model_cfg = OmegaConf.load(self._wilor_cfg)
        model = WiLoR.load_from_checkpoint(
            self._wilor_ckpt,
            strict=False,
            cfg=model_cfg,
        )
        model = model.to(self.device)
        model.eval()

        self._wilor_model     = model
        self._wilor_model_cfg = model_cfg

    def _load_hamer(self) -> None:
        """Load the AnyHand-HaMeR model from the HaMeR submodule."""
        _ensure_on_path(_REPO_ROOT / 'third_party' / 'hamer',
                        "HaMeR submodule not found. Run:  bash scripts/prepare_hamer.sh")
        try:
            from hamer.models.hamer import HAMER  # noqa: F401 (import check)
        except ImportError as exc:
            raise ImportError(
                "HaMeR package is not installed. Run:  bash scripts/prepare_hamer.sh"
            ) from exc

        _check_file(self._hamer_ckpt,
                    "HaMeR checkpoint not found. Run:  bash scripts/prepare_hamer.sh")

        # HaMeR's load_hamer() expects model_config.yaml in the SAME directory
        # as the checkpoint (see prepare_hamer.sh which places it there).
        ckpt_dir    = Path(self._hamer_ckpt).parent
        hamer_cfg_p = ckpt_dir / 'model_config.yaml'
        _check_file(str(hamer_cfg_p),
                    f"HaMeR model config not found at {hamer_cfg_p}. "
                    "Run:  bash scripts/prepare_hamer.sh")

        from omegaconf import OmegaConf
        from hamer.models.hamer import HAMER

        model_cfg = OmegaConf.load(str(hamer_cfg_p))
        model = HAMER.load_from_checkpoint(
            self._hamer_ckpt,
            strict=False,
            cfg=model_cfg,
        )
        model = model.to(self.device)
        model.eval()

        self._hamer_model     = model
        self._hamer_model_cfg = model_cfg

    # ==================================================================
    # Detection
    # ==================================================================

    def _detect_hands(
        self,
        img_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the YOLO hand detector on a single BGR image.

        Returns
        -------
        boxes    : (N, 4) float32  [x1, y1, x2, y2] in pixels
        is_right : (N,)   bool     True = right hand  (YOLO class 1)
        scores   : (N,)   float32  confidence scores
        """
        results = self._detector(
            img_bgr,
            verbose=False,
            conf=self.det_conf,
            iou=self.det_iou,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.array([], dtype=bool),
                np.array([], dtype=np.float32),
            )

        det      = results[0]
        boxes    = det.boxes.xyxy.cpu().numpy().astype(np.float32)  # (N, 4)
        cls      = det.boxes.cls.cpu().numpy().astype(int)          # 0=left, 1=right
        scores   = det.boxes.conf.cpu().numpy().astype(np.float32)
        is_right = (cls == 1)
        return boxes, is_right, scores

    # ==================================================================
    # Per-backend reconstruction
    # ==================================================================

    def _run_wilor(
        self,
        img_bgr: np.ndarray,
        boxes: np.ndarray,
        is_right: np.ndarray,
        scores: np.ndarray,
    ) -> List[HandPrediction]:
        from wilor.datasets.vitdet_dataset import ViTDetDataset

        model     = self._wilor_model
        model_cfg = self._wilor_model_cfg
        W, H      = img_bgr.shape[1], img_bgr.shape[0]
        img_size  = torch.tensor([W, H], dtype=torch.float32, device=self.device)

        scaled_focal = float(
            model_cfg.EXTRA.FOCAL_LENGTH
            / model_cfg.MODEL.IMAGE_SIZE
            * max(W, H)
        )

        dataset = ViTDetDataset(
            model_cfg,
            img_bgr,
            boxes,
            is_right,
            rescale_factor=self.rescale_factor,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return self._collect_predictions(
            loader, model, boxes, is_right, scores,
            img_size, scaled_focal, backend='wilor',
        )

    def _run_hamer(
        self,
        img_bgr: np.ndarray,
        boxes: np.ndarray,
        is_right: np.ndarray,
        scores: np.ndarray,
    ) -> List[HandPrediction]:
        from hamer.datasets.vitdet_dataset import ViTDetDataset

        model     = self._hamer_model
        model_cfg = self._hamer_model_cfg
        W, H      = img_bgr.shape[1], img_bgr.shape[0]
        img_size  = torch.tensor([W, H], dtype=torch.float32, device=self.device)

        scaled_focal = float(
            model_cfg.EXTRA.FOCAL_LENGTH
            / model_cfg.MODEL.IMAGE_SIZE
            * max(W, H)
        )

        dataset = ViTDetDataset(
            model_cfg,
            img_bgr,
            boxes,
            is_right,
            rescale_factor=self.rescale_factor,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return self._collect_predictions(
            loader, model, boxes, is_right, scores,
            img_size, scaled_focal, backend='hamer',
        )

    def _collect_predictions(
        self,
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        boxes: np.ndarray,
        is_right: np.ndarray,
        scores: np.ndarray,
        img_size: torch.Tensor,
        scaled_focal: float,
        backend: str,
    ) -> List[HandPrediction]:
        """
        Shared inference loop for both WiLoR and HaMeR.

        Both models share the same output dict schema, making this loop
        backend-agnostic.
        """
        preds: List[HandPrediction] = []
        hand_idx = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out   = model(batch)

                pred_cam      = out['pred_cam']           # (B, 3)  [s, tx, ty]
                pred_verts    = out['pred_vertices']      # (B, 778, 3)
                pred_kp3d     = out['pred_keypoints_3d']  # (B, 21,  3)
                pred_kp2d     = out['pred_keypoints_2d']  # (B, 21,  2)  normalised crop
                mano_params   = out['pred_mano_params']
                global_orient = mano_params['global_orient']   # (B, 1,  3, 3)
                hand_pose     = mano_params['hand_pose']        # (B, 15, 3, 3)
                betas         = mano_params['betas']            # (B, 10)

                box_center = batch['box_center'].float()  # (B, 2)
                box_size   = batch['box_size'].float()    # (B,)

                # Full-image camera translation
                cam_t_full = _cam_crop_to_full(
                    pred_cam, box_center, box_size,
                    img_size, scaled_focal,
                )  # (B, 3) on device

                B = pred_cam.shape[0]
                for i in range(B):
                    # ---- MANO pose: rot-mat → axis-angle ----
                    # global_orient[i]: (1, 3, 3)  →  (3,)
                    go_aa = _rotmat_to_aa(global_orient[i].cpu().numpy())     # (3,)
                    # hand_pose[i]:     (15, 3, 3) →  (45,)
                    hp_aa = _rotmat_to_aa(hand_pose[i].cpu().numpy())         # (45,)
                    mano_pose_aa = np.concatenate([go_aa, hp_aa], axis=0)     # (48,)

                    # ---- 2D keypoints: normalised crop → original pixels ----
                    kp2d_norm = pred_kp2d[i].cpu().numpy()                    # (21, 2) in [-1,1]
                    kp2d_px   = _kp2d_crop_to_full(
                        kp2d_norm,
                        box_center[i].cpu().numpy(),
                        float(box_size[i].cpu()),
                        model_size=int(batch['img'].shape[-1]),               # crop HxW (square)
                    )  # (21, 2) in pixel coords

                    preds.append(HandPrediction(
                        mano_pose    = mano_pose_aa.astype(np.float32),
                        mano_shape   = betas[i].cpu().numpy().astype(np.float32),
                        vertices     = pred_verts[i].cpu().numpy().astype(np.float32),
                        keypoints_3d = pred_kp3d[i].cpu().numpy().astype(np.float32),
                        keypoints_2d = kp2d_px.astype(np.float32),
                        cam_t        = cam_t_full[i].cpu().numpy().astype(np.float32),
                        focal_length = scaled_focal,
                        bbox         = boxes[hand_idx].copy(),
                        is_right     = bool(is_right[hand_idx]),
                        score        = float(scores[hand_idx]),
                        backend      = backend,
                    ))
                    hand_idx += 1

        return preds

    # ==================================================================
    # Public API
    # ==================================================================

    def predict(
        self,
        image: Union[np.ndarray, str, List[Union[np.ndarray, str]]],
        backend: Optional[Literal['wilor', 'hamer', 'both']] = None,
    ) -> Union[List[HandPrediction], List[List[HandPrediction]]]:
        """
        Run hand pose estimation on one or more images.

        Parameters
        ----------
        image : np.ndarray (H, W, 3) BGR  |  str filepath  |  list of either
            Input image(s). Numpy arrays must be in OpenCV BGR format.
        backend : 'wilor' | 'hamer' | 'both' | None
            Override the backend set at construction time for this call.
            None means use the instance-level default.

        Returns
        -------
        Single image  → List[HandPrediction]  (one entry per detected hand)
        List of images → List[List[HandPrediction]]

        Notes
        -----
        When backend='both', the result list contains predictions from WiLoR
        followed by predictions from HaMeR, sharing the same detected bboxes.
        Each HandPrediction.backend field tells you which model produced it.
        """
        backend = backend or self.backend

        # Validate that the requested backend has been loaded
        if backend in ('wilor', 'both') and self._wilor_model is None:
            raise RuntimeError(
                "WiLoR model not loaded. Re-instantiate with backend='wilor' or 'both'."
            )
        if backend in ('hamer', 'both') and self._hamer_model is None:
            raise RuntimeError(
                "HaMeR model not loaded. Re-instantiate with backend='hamer' or 'both'."
            )

        if isinstance(image, list):
            return [self._predict_single(img, backend) for img in image]
        return self._predict_single(image, backend)

    def _predict_single(
        self,
        image: Union[np.ndarray, str],
        backend: str,
    ) -> List[HandPrediction]:
        # ---- load image ----
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read image: {image}")
        elif isinstance(image, np.ndarray):
            img_bgr = image
        else:
            raise TypeError(f"image must be str, Path, or np.ndarray. Got {type(image)}")

        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(
                f"Expected a (H, W, 3) BGR array. Got shape: {img_bgr.shape}"
            )

        # ---- detect hands ----
        boxes, is_right, scores = self._detect_hands(img_bgr)
        if len(boxes) == 0:
            return []

        # ---- reconstruct ----
        results: List[HandPrediction] = []
        if backend in ('wilor', 'both'):
            results += self._run_wilor(img_bgr, boxes, is_right, scores)
        if backend in ('hamer', 'both'):
            results += self._run_hamer(img_bgr, boxes, is_right, scores)

        return results

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_wilor_loaded(self) -> bool:
        return self._wilor_model is not None

    @property
    def is_hamer_loaded(self) -> bool:
        return self._hamer_model is not None

    def __repr__(self) -> str:
        return (
            f"AnyHandPredictor("
            f"backend={self.backend!r}, "
            f"device={self.device}, "
            f"wilor={'✓' if self.is_wilor_loaded else '✗'}, "
            f"hamer={'✓' if self.is_hamer_loaded else '✗'}"
            f")"
        )


# ===========================================================================
# Geometry utilities (self-contained — no WiLoR/HaMeR imports needed)
# ===========================================================================

def _cam_crop_to_full(
    pred_cam: torch.Tensor,    # (B, 3)  [s, tx_crop, ty_crop]
    box_center: torch.Tensor,  # (B, 2)  [cx, cy] in pixels
    box_size: torch.Tensor,    # (B,)    square crop size in pixels
    img_size: torch.Tensor,    # (2,)    [W, H] in pixels  (or on same device)
    focal_length: float,
) -> torch.Tensor:             # (B, 3)  [tx, ty, tz] camera-space metres
    """
    Convert weak-perspective camera parameters from crop space to the
    full-image camera translation vector, following the same formula
    used internally by WiLoR and HaMeR.

    The weak-perspective model: x_img = f * (x_cam / z_cam)
    Given predicted [s, tx, ty] relative to the crop:
        tz = 2 * f / (s * crop_size)
        tx = tx_crop + (cx_box - cx_img) * 2 / (s * crop_size)
        ty = ty_crop + (cy_box - cy_img) * 2 / (s * crop_size)
    """
    s        = pred_cam[:, 0]           # (B,)
    tx_crop  = pred_cam[:, 1]           # (B,)
    ty_crop  = pred_cam[:, 2]           # (B,)

    cx_box   = box_center[:, 0]         # (B,)
    cy_box   = box_center[:, 1]         # (B,)

    # image centre (broadcast-friendly scalar)
    cx_img   = img_size[0] * 0.5
    cy_img   = img_size[1] * 0.5

    denom    = s * box_size + 1e-9      # (B,)  avoid division by zero

    tz       = 2.0 * focal_length / denom
    tx       = tx_crop + 2.0 * (cx_box - cx_img) / denom
    ty       = ty_crop + 2.0 * (cy_box - cy_img) / denom

    return torch.stack([tx, ty, tz], dim=1)  # (B, 3)


def _rotmat_to_aa(rotmat: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix (or stack of matrices) to axis-angle vectors.

    Parameters
    ----------
    rotmat : (..., 3, 3)

    Returns
    -------
    np.ndarray  shape (..., 3) flattened to 1-D
        e.g. (1, 3, 3) → (3,),  (15, 3, 3) → (45,)
    """
    from scipy.spatial.transform import Rotation

    shape   = rotmat.shape
    n_rots  = int(np.prod(shape[:-2]))                 # total number of matrices
    mats    = rotmat.reshape(n_rots, 3, 3)

    # Ensure valid rotation matrices (orthonormalise via SVD for robustness)
    U, _, Vt = np.linalg.svd(mats)
    mats_orth = U @ Vt
    # Fix reflections: det should be +1
    det_sign = np.linalg.det(mats_orth)[:, None, None]
    mats_orth[:, :, 2:] *= np.sign(det_sign)

    aa = Rotation.from_matrix(mats_orth).as_rotvec().astype(np.float32)  # (N, 3)
    return aa.reshape(-1)   # flatten: (N*3,)


def _kp2d_crop_to_full(
    kp2d_norm: np.ndarray,   # (21, 2) normalised in [-1, 1] (crop space)
    box_center: np.ndarray,  # (2,)  [cx, cy] in pixels
    box_size: float,         # scalar  crop size in pixels (square)
    model_size: int = 256,   # the ViT input resolution (typically 256)
) -> np.ndarray:             # (21, 2) pixel coords in original image
    """
    Map 2D keypoints from the model's normalised crop space back to
    pixel coordinates in the original (full) image.

    Both WiLoR and HaMeR normalise the crop to [-1, 1] before passing
    it into the ViT backbone, so this inverse mapping is:
        kp_px  = (kp_norm + 1) / 2 * model_size        # px in [0, model_size]
        kp_img = kp_px * (box_size / model_size) + (box_center - box_size / 2)
    which simplifies to:
        kp_img = (kp_norm + 1) / 2 * box_size + (box_center - box_size / 2)
    """
    # From [-1,1] to fraction of crop [0,1], then scale to crop pixel size
    kp = (kp2d_norm + 1.0) * 0.5 * box_size          # (21, 2) in crop pixels

    # Shift from crop-local to image-global
    kp[:, 0] += box_center[0] - box_size * 0.5
    kp[:, 1] += box_center[1] - box_size * 0.5

    return kp.astype(np.float32)


# ===========================================================================
# Internal helpers
# ===========================================================================

def _check_file(path: str, hint: str) -> None:
    """Raise FileNotFoundError with a helpful message if path doesn't exist."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"  → {hint}"
        )


def _ensure_on_path(pkg_dir: Path, hint: str) -> None:
    """
    Ensure that a source-installed package directory is on sys.path so that
    its top-level package can be imported.  If the directory doesn't exist,
    raises an informative ImportError.
    """
    if not pkg_dir.exists():
        raise ImportError(
            f"Package directory not found: {pkg_dir}\n"
            f"  → {hint}"
        )
    src = str(pkg_dir)
    if src not in sys.path:
        sys.path.insert(0, src)
