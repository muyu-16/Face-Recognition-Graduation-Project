"""feature_extractor.py —— 预训练骨干提取 512 维人脸特征。
两个骨干都实现（作业字面要求「骨干网络使用 ResNet18」）：
  inception_resnet_v1 : facenet-pytorch，VGGFace2 人脸预训练（默认，效果好）
  resnet18            : torchvision，ImageNet 通用预训练（符合字面要求，区分能力弱）
特征空间不通用：换骨干后必须重新注册人脸库。
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from face_detector import DEFAULT_IMAGE_SIZE, resolve_device, to_rgb_array
from similarity import l2_normalize
# 权重缓存固定放项目内 .cache/torch（~/.cache/torch 在不少企业电脑上不可写，WinError 5）；显式设了 TORCH_HOME 则尊重之
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parent / ".cache" / "torch"))
FACENET_MEAN, FACENET_STD = 127.5, 128.0
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
class FeatureExtractor:
    """把 160x160 人脸图编码成 512 维特征向量。"""
    SUPPORTED_BACKBONES = ("inception_resnet_v1", "resnet18")
    def __init__(self, backbone: str = "inception_resnet_v1",
                 pretrained: Union[str, bool, None] = "vggface2",
                 device: Optional[str] = None, l2_normalize: bool = True,
                 image_size: int = DEFAULT_IMAGE_SIZE) -> None:
        backbone = str(backbone).lower().strip()
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"不支持的骨干网络：{backbone}，可选 {self.SUPPORTED_BACKBONES}")
        self.backbone, self.pretrained = backbone, pretrained
        self.image_size, self.l2_normalize = int(image_size), bool(l2_normalize)
        self.device = resolve_device(device)
        self.model = self._build(backbone, pretrained).eval().to(self.device)
        with torch.no_grad():       # 探测输出维度，避免硬编码
            dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
            self.embedding_dim = int(self.model(dummy).shape[-1])
    @staticmethod
    def _is_pretrained(pretrained) -> bool:
        if pretrained is None or pretrained is False:
            return False
        return not (isinstance(pretrained, str) and pretrained.lower() in ("none", "false", "random", ""))
    def _build(self, backbone: str, pretrained) -> nn.Module:
        load_pretrained = self._is_pretrained(pretrained)
        try:
            if backbone == "inception_resnet_v1":
                from facenet_pytorch import InceptionResnetV1
                weight = pretrained if isinstance(pretrained, str) and load_pretrained else None
                return InceptionResnetV1(pretrained=weight, classify=False)
            from torchvision import models
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1
                                    if load_pretrained else None)
            model.fc = nn.Identity()     # 去掉分类头，输出 512 维池化特征
            return model
        except Exception as exc:
            from pathlib import Path as _P
            cache = os.environ.get("TORCH_HOME", str(_P.home() / ".cache" / "torch"))
            raise RuntimeError(
                f"预训练模型加载失败：{exc}\n当前权重缓存目录：{cache}\n"
                "首次运行需联网下载（InceptionResnetV1 约 110MB / ResNet18 约 45MB）；\n"
                "也可手动下载到上述目录的 checkpoints/ 下，或设置 TORCH_HOME 换缓存位置。") from exc
    def preprocess(self, faces) -> torch.Tensor:
        if isinstance(faces, np.ndarray) and faces.ndim == 3:
            faces = [faces]
        if len(faces) == 0:
            return torch.zeros(0, 3, self.image_size, self.image_size, device=self.device)
        x = torch.from_numpy(np.stack([to_rgb_array(f) for f in faces], axis=0))
        x = x.permute(0, 3, 1, 2).float()
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        if self.backbone == "inception_resnet_v1":
            x = (x - FACENET_MEAN) / FACENET_STD
        else:
            mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
            x = (x / 255.0 - mean) / std
        return x.to(self.device)
    @torch.no_grad()
    def extract_batch(self, faces: Sequence[np.ndarray], batch_size: int = 32,
                      progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        if len(faces) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        outputs: List[np.ndarray] = []
        for start in range(0, len(faces), int(batch_size)):
            chunk = faces[start:start + int(batch_size)]
            feat = self.model(self.preprocess(chunk)).detach().cpu().numpy().astype(np.float32)
            outputs.append(l2_normalize(feat, axis=-1) if self.l2_normalize else feat)
            if progress_callback is not None:
                progress_callback(min(start + len(chunk), len(faces)), len(faces))
        return np.concatenate(outputs, axis=0)
    def extract(self, face: np.ndarray) -> np.ndarray:
        return self.extract_batch([face])[0]
    def describe(self) -> Dict[str, object]:
        return {"骨干网络": self.backbone, "预训练权重": str(self.pretrained),
                "特征维度": self.embedding_dim, "输入尺寸": f"{self.image_size}x{self.image_size}",
                "是否 L2 归一化": self.l2_normalize, "运行设备": str(self.device)}
    def __repr__(self) -> str:
        return f"FeatureExtractor(backbone='{self.backbone}', dim={self.embedding_dim}, device='{self.device}')"
