"""face_detector.py —— MTCNN 人脸检测 + 关键点对齐 + 裁剪，并提供共用的图像格式工具。"""
from __future__ import annotations
import math
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
        pad = int(self.margin if margin is None else margin)
        pil, box = to_rgb_pil(image), np.asarray(face.box, dtype=np.float32)
        if self.align and face.landmarks is not None and len(face.landmarks) >= 2:
            (lx, ly), (rx, ry) = face.landmarks[0], face.landmarks[1]
            angle = math.degrees(math.atan2(float(ry - ly), float(rx - lx)))
            if abs(angle) >= 0.5:          # 小于 0.5 度不值得旋转
                cx0, cy0 = pil.width / 2.0, pil.height / 2.0
                pil = pil.rotate(angle, resample=Image.BILINEAR, center=(cx0, cy0),
                                 fillcolor=(0, 0, 0))
                bx, by = (box[0] + box[2]) / 2.0 - cx0, (box[1] + box[3]) / 2.0 - cy0
                t = math.radians(angle)
                nx, ny = math.cos(t) * bx + math.sin(t) * by, -math.sin(t) * bx + math.cos(t) * by
                hw, hh = (box[2] - box[0]) / 2.0, (box[3] - box[1]) / 2.0
                box = self._clip([cx0 + nx - hw, cy0 + ny - hh, cx0 + nx + hw, cy0 + ny + hh],
                                 pil.width, pil.height)
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        side = max(4.0, max(float(box[2] - box[0]), float(box[3] - box[1])) + 2.0 * pad)
        crop = pil.crop((int(round(cx - side / 2)), int(round(cy - side / 2)),
                         int(round(cx + side / 2)), int(round(cy + side / 2))))
        w, h = crop.size
        if w != h:                          # 黑边补齐成正方形，避免形变
            canvas = Image.new("RGB", (max(w, h),) * 2, (0, 0, 0))
            canvas.paste(crop, ((max(w, h) - w) // 2, (max(w, h) - h) // 2))
            crop = canvas
        if crop.size != (size, size):       # 3) 缩放到模型输入尺寸
            crop = crop.resize((size, size), resample=Image.BILINEAR)
        return np.asarray(crop, dtype=np.uint8)
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
