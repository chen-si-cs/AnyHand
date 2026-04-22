"""
anyhand/predictor.py
====================
Unified hand pose predictor that:
  1. Runs WiLoR's YOLO hand detector to find bounding boxes.
  2. Dispatches to either AnyHand-WiLoR or AnyHand-HaMeR (or both)
     for 3D reconstruction.
  3. Returns structured HandPrediction objects.
  4. Optionally renders mesh overlays or saves per-hand .obj files.

Usage
-----
    from anyhand import AnyHandPredictor

    # Basic prediction
    predictor = AnyHandPredictor(backend='wilor')
    hands = predictor.predict('image.jpg')

    # Render mesh overlay on the image (returns BGR uint8 numpy array)
    overlay = predictor.render_overlay('image.jpg', hands)
    cv2.imwrite('out.jpg', overlay)

    # Save per-hand .obj meshes to disk
    saved = predictor.save_meshes(hands, out_dir='meshes', prefix='frame0')

    # Project 3D vertices to 2D pixels (static helper)
    import cv2, numpy as np
    img = cv2.imread('image.jpg')
    for hand in hands:
        kpts2d = AnyHandPredictor.project_3d_to_2d(
            hand.vertices, hand.cam_t, hand.focal_length,
            img_size=(img.shape[1], img.shape[0]),
        )

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
from typing import List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve paths relative to the repo root (two levels above this file).
# ---------------------------------------------------------------------------
_THIS_DIR  = Path(__file__).resolve().parent          # anyhand/
_REPO_ROOT = _THIS_DIR.parent                         # AnyHand/

LIGHT_GOLD = (0.42, 0.02, 0.50)

from contextlib import contextmanager

@contextmanager
def _torch_load_trusted():
    """
    Temporarily force torch.load(weights_only=False).

    PyTorch 2.6 flipped the default to True, which refuses to unpickle
    arbitrary Python objects for safety. This breaks loaders that pickle
    non-tensor objects:
      - ultralytics' detector.pt pickles a full PoseModel.
      - Lightning checkpoints often contain OmegaConf DictConfig in hparams.

    We trust our own checkpoints (pinned HuggingFace repos). Use this
    context manager around load calls that don't expose a weights_only kwarg
    directly (e.g. ultralytics.YOLO).
    """
    orig = torch.load
    def patched(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return orig(*args, **kwargs)
    torch.load = patched
    try:
        yield
    finally:
        torch.load = orig


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
        NOTE: x-axis is already flipped for left hands so that all vertices
        are in the canonical right-hand coordinate frame matching the renderer.
    keypoints_3d : (21, 3) float32
        3D hand joints in camera-space metres (same flip convention as vertices).
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

        # ---- internal state ----
        self._wilor_model     = None
        self._wilor_model_cfg = None
        self._hamer_model     = None
        self._hamer_model_cfg = None
        self._renderer        = None   # lazy-loaded on first render call

        # ---- load components ----
        self._load_detector()

        if backend in ('wilor', 'both'):
            self._load_wilor()
        if backend in ('hamer', 'both'):
            self._load_hamer()

    # ==================================================================
    # Loading helpers
    # ==================================================================

    def _load_detector(self) -> None:
        """Load WiLoR's YOLO hand detector (used for all backends)."""
        from ultralytics import YOLO

        _check_file(self._detector_pt,
                    "Hand detector not found. Run:  bash scripts/prepare_wilor.sh")

        # ultralytics calls torch.load internally with no way to pass
        # weights_only=False, so we monkey-patch torch.load for the duration
        # of the constructor. See _torch_load_trusted() docstring.
        with _torch_load_trusted():
            self._detector = YOLO(self._detector_pt).to(self.device)

    def _load_wilor(self) -> None:
        """Load the AnyHand-WiLoR model from the WiLoR submodule."""
        _ensure_on_path(_REPO_ROOT / 'WiLoR',
                        "WiLoR submodule not found. Run:  bash scripts/prepare_wilor.sh")
        try:
            from wilor.models.wilor import WiLoR  # noqa: F401
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

        # Lightning's load_from_checkpoint exposes weights_only directly — use it.
        # The checkpoint's hparams contain OmegaConf DictConfig objects, which
        # weights_only=True refuses to unpickle on PyTorch 2.6+.
        model = WiLoR.load_from_checkpoint(
            self._wilor_ckpt,
            strict=False,
            cfg=model_cfg,
            weights_only=False,
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
            from hamer.models.hamer import HAMER  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "HaMeR package is not installed. Run:  bash scripts/prepare_hamer.sh"
            ) from exc

        _check_file(self._hamer_ckpt,
                    "HaMeR checkpoint not found. Run:  bash scripts/prepare_hamer.sh")

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

    def _ensure_renderer(self) -> None:
        """
        Lazy-load WiLoR's pyrender-based Renderer on first use.
        Requires WiLoR to be loaded (it provides the MANO face topology).
        """
        if self._renderer is not None:
            return
        if self._wilor_model is None:
            raise RuntimeError(
                "The renderer requires WiLoR to be loaded (it uses WiLoR's MANO face "
                "topology). Re-instantiate with backend='wilor' or 'both', or pass "
                "backend='wilor' to render_overlay() / save_meshes()."
            )
        from wilor.utils.renderer import Renderer
        self._renderer = Renderer(
            self._wilor_model_cfg,
            faces=self._wilor_model.mano.faces,
        )

    # ==================================================================
    # Detection
    # ==================================================================

    def _detect_hands(
        self,
        img_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

                # ---- camera: flip tx for left hands ----
                # demo.py lines 80-82: multiplier = (2*right-1); pred_cam[:,1] *= multiplier
                # This mirrors the weak-perspective tx so that the mesh lands on
                # the correct side of the image for left hands.
                multiplier    = (2 * batch['right'] - 1).float()   # (B,) +1 or -1
                pred_cam      = out['pred_cam'].clone()             # (B, 3)
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]

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
                )  # (B, 3)

                B = pred_cam.shape[0]
                for i in range(B):
                    # ---- MANO pose: rot-mat → axis-angle ----
                    go_aa        = _rotmat_to_aa(global_orient[i].cpu().numpy())  # (3,)
                    hp_aa        = _rotmat_to_aa(hand_pose[i].cpu().numpy())      # (45,)
                    mano_pose_aa = np.concatenate([go_aa, hp_aa], axis=0)         # (48,)

                    # ---- vertices & 3D joints: flip x for left hands ----
                    # demo.py lines 99-101: verts[:,0] = (2*is_right-1)*verts[:,0]
                    # Mirrors vertices into a canonical right-hand frame so that the
                    # renderer (which always renders a right hand) produces correct output.
                    is_right_i = float(batch['right'][i].item())
                    flip        = float(2 * is_right_i - 1)          # +1 right, -1 left

                    verts_np    = pred_verts[i].cpu().numpy().astype(np.float32)
                    joints_np   = pred_kp3d[i].cpu().numpy().astype(np.float32)
                    verts_np[:, 0]  = flip * verts_np[:, 0]
                    joints_np[:, 0] = flip * joints_np[:, 0]

                    # ---- 2D keypoints: normalised crop → original pixels ----
                    kp2d_norm = pred_kp2d[i].cpu().numpy()
                    kp2d_px   = _kp2d_crop_to_full(
                        kp2d_norm,
                        box_center[i].cpu().numpy(),
                        float(box_size[i].cpu()),
                        model_size=int(batch['img'].shape[-1]),
                    )

                    preds.append(HandPrediction(
                        mano_pose    = mano_pose_aa.astype(np.float32),
                        mano_shape   = betas[i].cpu().numpy().astype(np.float32),
                        vertices     = verts_np,
                        keypoints_3d = joints_np,
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
    # Public API — prediction
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
        Each HandPrediction.backend tells you which model produced it.
        """
        backend = backend or self.backend

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
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read image: {image}")
        elif isinstance(image, np.ndarray):
            img_bgr = image
        else:
            raise TypeError(f"image must be str, Path, or np.ndarray. Got {type(image)}")

        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) BGR array. Got shape: {img_bgr.shape}")

        boxes, is_right, scores = self._detect_hands(img_bgr)
        if len(boxes) == 0:
            return []

        results: List[HandPrediction] = []
        if backend in ('wilor', 'both'):
            results += self._run_wilor(img_bgr, boxes, is_right, scores)
        if backend in ('hamer', 'both'):
            results += self._run_hamer(img_bgr, boxes, is_right, scores)

        return results

    # ==================================================================
    # Public API — visualisation
    # ==================================================================

    def render_overlay(
        self,
        image: Union[np.ndarray, str],
        hands: List[HandPrediction],
        mesh_color: Tuple[float, float, float] = LIGHT_GOLD,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        Render MANO mesh overlays for all detected hands onto the input image.

        Uses WiLoR's pyrender-based Renderer. The Renderer is lazy-loaded on
        the first call and cached for subsequent calls.

        Parameters
        ----------
        image : np.ndarray (H, W, 3) BGR  |  str filepath
            The original input image.
        hands : List[HandPrediction]
            Output of predict().  Can be empty — returns the original image.
        mesh_color : (R, G, B) float tuple in [0, 1]
            Colour of the rendered mesh faces. Default is WiLoR's light purple.
        bg_color : (R, G, B) float tuple in [0, 1]
            Background colour for the renderer scene.

        Returns
        -------
        np.ndarray  (H, W, 3)  uint8  BGR
            Image with mesh overlaid via alpha compositing.

        Example
        -------
        >>> overlay = predictor.render_overlay('photo.jpg', hands)
        >>> cv2.imwrite('out.jpg', overlay)
        """
        # ---- load image ----
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read image: {image}")
        else:
            img_bgr = image

        if not hands:
            return img_bgr.copy()

        self._ensure_renderer()

        H, W  = img_bgr.shape[:2]
        img_size = torch.tensor([W, H], dtype=torch.float32)

        # Collect per-hand data — renderer expects lists
        all_verts   = [h.vertices  for h in hands]
        all_cam_t   = [h.cam_t     for h in hands]
        all_is_right = [h.is_right for h in hands]

        # focal_length may differ per hand (e.g. if images differ in size);
        # use the first hand's value — they're all computed from the same image.
        focal_length = hands[0].focal_length

        misc_args = dict(
            mesh_base_color = mesh_color,
            scene_bg_color  = bg_color,
            focal_length    = focal_length,
        )

        # render_rgba_multiple returns (H, W, 4) float32 RGBA in [0, 1]
        cam_view = self._renderer.render_rgba_multiple(
            all_verts,
            cam_t      = all_cam_t,
            render_res = img_size,
            is_right   = all_is_right,
            **misc_args,
        )

        # Alpha-composite mesh over original image (RGB space)
        # demo.py lines 128-130
        img_rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, [0,1]
        alpha   = cam_view[:, :, 3:]                                 # (H, W, 1)
        overlay = img_rgb * (1 - alpha) + cam_view[:, :, :3] * alpha

        # Return BGR uint8 to stay consistent with OpenCV conventions
        return (overlay[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)

    def save_meshes(
        self,
        hands: List[HandPrediction],
        out_dir: str,
        prefix: str = 'hand',
        mesh_color: Tuple[float, float, float] = LIGHT_GOLD,
    ) -> List[str]:
        """
        Save each detected hand mesh as a separate .obj file.

        Uses WiLoR's Renderer.vertices_to_trimesh() internally, so the
        exported meshes include vertex colours.

        Parameters
        ----------
        hands : List[HandPrediction]
            Output of predict().
        out_dir : str
            Directory where .obj files will be written (created if needed).
        prefix : str
            Filename prefix.  Files are named  <prefix>_<n>.obj
            where n is the 0-based hand index.
        mesh_color : (R, G, B) float tuple in [0, 1]
            Vertex colour for the mesh.

        Returns
        -------
        List[str]
            Absolute paths of every saved .obj file.

        Example
        -------
        >>> paths = predictor.save_meshes(hands, 'out/meshes', prefix='frame0042')
        >>> print(paths)
        ['out/meshes/frame0042_0.obj', 'out/meshes/frame0042_1.obj']
        """
        if not hands:
            return []

        self._ensure_renderer()
        os.makedirs(out_dir, exist_ok=True)

        saved_paths: List[str] = []
        for n, hand in enumerate(hands):
            out_path = os.path.join(out_dir, f'{prefix}_{n}.obj')
            tmesh = self._renderer.vertices_to_trimesh(
                hand.vertices,
                hand.cam_t.copy(),
                mesh_color,
                is_right=hand.is_right,
            )
            tmesh.export(out_path)
            saved_paths.append(os.path.abspath(out_path))

        return saved_paths

    # ==================================================================
    # Public API — static geometry helpers
    # ==================================================================

    @staticmethod
    def project_3d_to_2d(
        points: np.ndarray,
        cam_t: np.ndarray,
        focal_length: float,
        img_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Project 3D camera-space points onto the full image plane.

        Implements the standard pinhole projection with principal point at
        the image centre, matching demo.py's project_full_img().

        Parameters
        ----------
        points : (N, 3) float32
            3D points in camera space (e.g. hand.vertices or hand.keypoints_3d).
        cam_t : (3,) float32
            Camera translation from HandPrediction.cam_t.
        focal_length : float
            Focal length in pixels from HandPrediction.focal_length.
        img_size : (W, H) int tuple
            Width and height of the original image in pixels.

        Returns
        -------
        np.ndarray  (N, 2)  float32
            2D pixel coordinates in the original image.

        Example
        -------
        >>> kpts2d = AnyHandPredictor.project_3d_to_2d(
        ...     hand.vertices, hand.cam_t, hand.focal_length,
        ...     img_size=(img.shape[1], img.shape[0]),
        ... )
        """
        W, H = img_size
        cx, cy = W / 2.0, H / 2.0

        # Translate into camera frame
        pts = points + cam_t[None, :]            # (N, 3)

        # Perspective division
        pts_norm = pts / pts[:, 2:3]             # (N, 3), z=1

        # Apply intrinsic matrix  K = diag(f, f, 1) with principal point (cx, cy)
        u = focal_length * pts_norm[:, 0] + cx   # (N,)
        v = focal_length * pts_norm[:, 1] + cy   # (N,)

        return np.stack([u, v], axis=1).astype(np.float32)  # (N, 2)

    # ==================================================================
    # Convenience properties
    # ==================================================================

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
    img_size: torch.Tensor,    # (2,)    [W, H] in pixels
    focal_length: float,
) -> torch.Tensor:             # (B, 3)  [tx, ty, tz] camera-space metres
    """
    Convert weak-perspective camera parameters from crop space to the
    full-image camera translation vector.

    The weak-perspective model: x_img = f * (x_cam / z_cam)
    Given predicted [s, tx, ty] relative to the crop:
        tz = 2 * f / (s * crop_size)
        tx = tx_crop + (cx_box - cx_img) * 2 / (s * crop_size)
        ty = ty_crop + (cy_box - cy_img) * 2 / (s * crop_size)
    """
    s       = pred_cam[:, 0]
    tx_crop = pred_cam[:, 1]
    ty_crop = pred_cam[:, 2]
    cx_box  = box_center[:, 0]
    cy_box  = box_center[:, 1]
    cx_img  = img_size[0] * 0.5
    cy_img  = img_size[1] * 0.5
    denom   = s * box_size + 1e-9
    tz      = 2.0 * focal_length / denom
    tx      = tx_crop + 2.0 * (cx_box - cx_img) / denom
    ty      = ty_crop + 2.0 * (cy_box - cy_img) / denom
    return torch.stack([tx, ty, tz], dim=1)


def _rotmat_to_aa(rotmat: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix (or stack) to axis-angle vectors, flattened.
    (..., 3, 3)  →  (N*3,)
    """
    from scipy.spatial.transform import Rotation

    n_rots   = int(np.prod(rotmat.shape[:-2]))
    mats     = rotmat.reshape(n_rots, 3, 3)
    U, _, Vt = np.linalg.svd(mats)
    mats_orth = U @ Vt
    det_sign  = np.linalg.det(mats_orth)[:, None, None]
    mats_orth[:, :, 2:] *= np.sign(det_sign)
    return Rotation.from_matrix(mats_orth).as_rotvec().astype(np.float32).reshape(-1)


def _kp2d_crop_to_full(
    kp2d_norm: np.ndarray,   # (21, 2) in [-1, 1]
    box_center: np.ndarray,  # (2,)
    box_size: float,
    model_size: int = 256,
) -> np.ndarray:             # (21, 2) pixel coords
    kp = (kp2d_norm + 1.0) * 0.5 * box_size
    kp[:, 0] += box_center[0] - box_size * 0.5
    kp[:, 1] += box_center[1] - box_size * 0.5
    return kp.astype(np.float32)


# ===========================================================================
# Internal helpers
# ===========================================================================

def _check_file(path: str, hint: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n  → {hint}"
        )


def _ensure_on_path(pkg_dir: Path, hint: str) -> None:
    if not pkg_dir.exists():
        raise ImportError(
            f"Package directory not found: {pkg_dir}\n  → {hint}"
        )
    src = str(pkg_dir)
    if src not in sys.path:
        sys.path.insert(0, src)