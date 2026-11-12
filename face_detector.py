"""face_detector.py —— MTCNN 人脸检测 + 关键点对齐 + 裁剪，并提供共用的图像格式工具。"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
ImageLike = Union[str, os.PathLike, Image.Image, np.ndarray]
DEFAULT_IMAGE_SIZE, DEFAULT_MARGIN, DEFAULT_MIN_FACE_SIZE = 160, 20, 40
_CJK_FONTS = (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
              "/System/Library/Fonts/PingFang.ttc")
# ArcFace 标准 5 点参考坐标（112x112），按其布局做相似变换即可消除旋转/缩放/平移差异
_ALIGN_REF_112 = np.array([
    [30.2946, 51.6963],   # 左眼
    [65.5318, 51.5014],   # 右眼
    [48.0252, 71.7366],   # 鼻尖
    [33.5493, 92.3655],   # 左嘴角
    [62.7299, 92.2041],   # 右嘴角
], dtype=np.float64)
def _similarity_transform(src, dst):
    """最小二乘求 2D 相似变换（旋转+等比缩放+平移），返回 2x3 矩阵：src -> dst。"""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = src.shape[0]
    A = np.zeros((2 * n, 4)); b = np.zeros(2 * n)
    for i in range(n):
        sx, sy = src[i]; dx, dy = dst[i]
        A[2*i]     = [sx, -sy, 1.0, 0.0]
        A[2*i+1]   = [sy,  sx, 0.0, 1.0]
        b[2*i]     = dx; b[2*i+1] = dy
    a, bb, tx, ty = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([[a, -bb, tx], [bb, a, ty]], dtype=np.float64)
def _align_by_landmarks(pil, landmarks, size):
    """按 5 点相似变换把整图重采样到 size×size（PIL 需要逆映射，故对 M 求逆）。"""
    ref = _ALIGN_REF_112 * (size / 112.0)
    M = _similarity_transform(np.asarray(landmarks[:5], dtype=np.float64), ref)
    inv = np.linalg.inv(np.vstack([M, [0.0, 0.0, 1.0]]))
    a, b, c = inv[0, 0], inv[0, 1], inv[0, 2]
    d, e, f = inv[1, 0], inv[1, 1], inv[1, 2]
    aligned = pil.transform((size, size), Image.AFFINE, (a, b, c, d, e, f),
                            resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    return np.asarray(aligned, dtype=np.uint8)
def to_rgb_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8)
def to_rgb_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, np.ndarray):
        return Image.fromarray(np.ascontiguousarray(to_rgb_array(image))).convert("RGB")
    return Image.open(image).convert("RGB")
def resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred is None or str(preferred).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(preferred).lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("选择了 GPU（CUDA）但检测不到可用设备，请改选 CPU 或安装 CUDA 版 PyTorch。")
    return torch.device(preferred)
@dataclass
class Face:
    box: np.ndarray
    prob: float
    landmarks: Optional[np.ndarray] = None
    @property
    def area(self) -> float:
        return max(0.0, float(self.box[2] - self.box[0])) * max(0.0, float(self.box[3] - self.box[1]))
class FaceDetector:
    """MTCNN 封装：检测 -> 关键点对齐 -> 裁成 160x160。"""
    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE, margin: int = DEFAULT_MARGIN,
                 min_face_size: int = DEFAULT_MIN_FACE_SIZE,
                 thresholds: Sequence[float] = (0.6, 0.7, 0.7), factor: float = 0.709,
                 device: Optional[str] = None, align: bool = True, keep_all: bool = True) -> None:
        self.image_size, self.margin = int(image_size), int(margin)
        self.min_face_size, self.align, self.keep_all = int(min_face_size), bool(align), bool(keep_all)
        self.device = resolve_device(device)
        self.mtcnn = MTCNN(image_size=self.image_size, margin=self.margin,
                           min_face_size=self.min_face_size,
                           thresholds=[float(t) for t in thresholds], factor=float(factor),
                           keep_all=self.keep_all, select_largest=not self.keep_all,
                           post_process=False, device=self.device)
    @staticmethod
    def _clip(box, width: int, height: int) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in box]
        return np.array([max(0.0, min(x1, width - 1.0)), max(0.0, min(y1, height - 1.0)),
                         max(0.0, min(x2, width - 1.0)), max(0.0, min(y2, height - 1.0))],
                        dtype=np.float32)
    def detect(self, image: ImageLike) -> List[Face]:
        pil = to_rgb_pil(image)
        boxes, probs, landmarks = self.mtcnn.detect(pil, landmarks=True)
        faces: List[Face] = []
        if boxes is None:
            return faces
        for i, box in enumerate(boxes):
            if box is None:
                continue
            lm = landmarks[i] if landmarks is not None else None
            faces.append(Face(self._clip(box, pil.width, pil.height),
                              float(probs[i]) if probs is not None and probs[i] is not None else 0.0,
                              None if lm is None else np.asarray(lm, dtype=np.float32)))
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces
    def crop(self, image: ImageLike, face: Face, image_size: Optional[int] = None,
             margin: Optional[int] = None) -> np.ndarray:
        size = int(image_size or self.image_size)
        pil = to_rgb_pil(image)
        # 首选：5 点齐全时用相似变换对齐（同时消除旋转、缩放、平移差异）
        if self.align and face.landmarks is not None and len(face.landmarks) >= 5:
            pts = np.asarray(face.landmarks[:5], dtype=np.float64)
            if np.all(np.isfinite(pts)):
                try:
                    return _align_by_landmarks(pil, pts, size)
                except np.linalg.LinAlgError:   # 关键点退化（共线）时退回 box 裁剪
                    pass
        # 退化路径：关键点不足或不可用，按 box 外扩后裁正方形
        pad = int(self.margin if margin is None else margin)
        box = np.asarray(face.box, dtype=np.float32)
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        side = max(4.0, max(float(box[2] - box[0]), float(box[3] - box[1])) + 2.0 * pad)
        crop_img = pil.crop((int(round(cx - side / 2)), int(round(cy - side / 2)),
                             int(round(cx + side / 2)), int(round(cy + side / 2))))
        w, h = crop_img.size
        if w != h:                          # 黑边补齐成正方形，避免形变
            canvas = Image.new("RGB", (max(w, h),) * 2, (0, 0, 0))
            canvas.paste(crop_img, ((max(w, h) - w) // 2, (max(w, h) - h) // 2))
            crop_img = canvas
        if crop_img.size != (size, size):   # 3) 缩放到模型输入尺寸
            crop_img = crop_img.resize((size, size), resample=Image.BILINEAR)
        return np.asarray(crop_img, dtype=np.uint8)
    def detect_and_crop(self, image: ImageLike, max_faces: Optional[int] = None):
        faces = self.detect(image)
        if max_faces is not None:
            faces = faces[: int(max_faces)]
        return faces, [self.crop(image, f) for f in faces]
    def draw(self, image, faces, unknown_flags=None, width: int = 3) -> Image.Image:
        canvas = to_rgb_pil(image).copy()
        draw = ImageDraw.Draw(canvas)
        for idx, face in enumerate(faces):
            unknown = bool(unknown_flags[idx]) if unknown_flags is not None and idx < len(unknown_flags) else False
            x1, y1, x2, y2 = [float(v) for v in face.box]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64) if unknown else (0, 255, 0), width=width)
        return canvas
