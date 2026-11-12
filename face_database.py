"""face_database.py —— 人脸库：特征增删查 + pickle 持久化（原子写盘 + 自动备份）。"""
from __future__ import annotations
import os
import pickle
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image
from similarity import l2_normalize
SCHEMA_VERSION = 1
DEFAULT_DB_PATH, DEFAULT_FACE_DIR = Path("face_db.pkl"), Path("registered_faces")
_ILLEGAL = re.compile(r'[\\/:*?"<>|\r\n\t]+')
def sanitize_name(name: str) -> str:
    # 首尾空白与句点必须一起去：分两步做时 ". . ." 会残留 " . "（Windows 也不允许目录名以空格/句点结尾）
    return _ILLEGAL.sub("_", str(name)).strip().strip(" .")
def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
class FaceDatabase:
    """人脸库：特征的增删查与本地 pickle 持久化。"""
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH,
                 face_dir: Path | str = DEFAULT_FACE_DIR, auto_load: bool = True,
                 save_face_images: bool = True, l2_normalize: bool = True) -> None:
        self.db_path, self.face_dir = Path(db_path), Path(face_dir)
        self.save_face_images, self.l2_normalize = bool(save_face_images), bool(l2_normalize)
        self.people: Dict[str, dict] = {}
        self.meta: Dict[str, object] = {}
        self.last_error: Optional[str] = None
        self.created_at = self.updated_at = _now()
        if auto_load:
            self.load()
    def load(self, force: bool = False) -> "FaceDatabase":
        if not self.db_path.exists():
            self.people, self.meta = {}, {}
            return self
        if self.people and not force:
            return self
        try:
            with open(self.db_path, "rb") as f:
                payload = pickle.load(f)
            self.people = payload.get("people", {}) or {}
            self.meta = payload.get("meta", {}) or {}
            self.created_at = payload.get("created_at", _now())
            self.updated_at = payload.get("updated_at", _now())
        except Exception as exc:
            self.last_error = f"人脸库读取失败（{exc}），已按空库启动。"
            self.people, self.meta = {}, {}
        return self
    def save(self) -> Path:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        payload = {"schema_version": SCHEMA_VERSION, "created_at": self.created_at,
                   "updated_at": self.updated_at, "meta": self.meta, "people": self.people}
        if self.db_path.exists():
            try:
                shutil.copy2(self.db_path, self.db_path.with_name(self.db_path.stem + ".bak.pkl"))
            except Exception:
                pass
        tmp = self.db_path.with_name(self.db_path.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.db_path)
        return self.db_path
    def export_bytes(self) -> bytes:
        return pickle.dumps({"schema_version": SCHEMA_VERSION, "created_at": self.created_at,
                             "updated_at": self.updated_at, "meta": self.meta,
                             "people": self.people}, protocol=pickle.HIGHEST_PROTOCOL)
    def add_person(self, name: str, embeddings: np.ndarray,
                   face_images: Optional[Sequence[np.ndarray]] = None, overwrite: bool = False,
                   backbone: Optional[str] = None, append: bool = True) -> dict:
        safe = sanitize_name(name)
        emb = np.asarray(embeddings, dtype=np.float32)
        if not safe or emb.size == 0:
            raise ValueError("姓名不能为空，且至少要有一张包含人脸的照片")
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if not np.all(np.isfinite(emb)):
            raise ValueError("特征向量中出现 NaN/Inf，请检查输入图片")
        emb = l2_normalize(emb, axis=1) if self.l2_normalize else emb
        if self.embedding_dim is not None and self.embedding_dim != emb.shape[1]:
            raise ValueError(f"特征维度不一致：人脸库 {self.embedding_dim} 维，本次 {emb.shape[1]} 维。"
                             f"更换骨干网络后请先清空人脸库再重新注册。")
        if safe in self.people and not overwrite and not append:
            raise ValueError(f"『{safe}』已存在；如需覆盖请勾选覆盖选项。")
        old = self.people.get(safe, {})
        created = old.get("created_at", _now())
        old_emb = np.asarray(old.get("embeddings", np.zeros((0, emb.shape[1]), np.float32)))
        paths: List[str] = list(old.get("image_paths", []))
        if overwrite:
            new_emb, paths, start = emb, [], 0
        else:
            new_emb = np.concatenate([old_emb, emb], axis=0) if old_emb.size else emb
            start = len(paths)
        if face_images is not None and self.save_face_images:
            try:
                person_dir = self.face_dir / safe
                if overwrite and person_dir.exists():
                    shutil.rmtree(person_dir, ignore_errors=True)
                person_dir.mkdir(parents=True, exist_ok=True)
                for offset, img in enumerate(face_images):
                    path = person_dir / f"{start + offset:04d}.png"
                    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(path)
                    paths.append(str(path))
                self.meta.pop("image_save_error", None)
            except Exception as exc:      # 存图只是可视化辅助，不能影响特征入库
                self.meta["image_save_error"] = f"人脸小图保存失败（特征已入库）：{exc}"
        self.people[safe] = {"embeddings": np.ascontiguousarray(new_emb, dtype=np.float32),
                             "image_paths": paths, "created_at": created, "updated_at": _now(),
                             "backbone": backbone or self.meta.get("backbone")}
        self.meta["backbone"] = backbone or self.meta.get("backbone")
        self.meta["embedding_dim"] = int(new_emb.shape[1])
        self._sync_counts()
        self.save()
        return {"姓名": safe, "图片数": int(new_emb.shape[0]), "注册时间": created,
                "更新时间": self.people[safe]["updated_at"]}
    def _sync_counts(self) -> None:
        """把 meta 里的人数与特征数刷新为与 people 实际内容一致（增删人员后必须调用）。"""
        self.meta.update(num_people=self.num_people, num_embeddings=self.num_embeddings)
    def delete_person(self, name: str) -> bool:
        safe = sanitize_name(name)
        if safe not in self.people:
            return False
        del self.people[safe]
        person_dir = self.face_dir / safe
        if person_dir.exists():
            shutil.rmtree(person_dir, ignore_errors=True)
        self._sync_counts()
        self.save()
        return True
    def rename_person(self, old_name: str, new_name: str) -> bool:
        old_safe, new_safe = sanitize_name(old_name), sanitize_name(new_name)
        if old_safe not in self.people or not new_safe:
            return False
        if new_safe in self.people and new_safe != old_safe:
            raise ValueError(f"『{new_safe}』已存在，无法重名。")
        if new_safe == old_safe:      # 改成同一个名字：幂等成功，不动内存/图片/人脸库
            return True
        record = self.people.pop(old_safe)
        old_dir, new_dir = self.face_dir / old_safe, self.face_dir / new_safe
        if old_dir.exists():
            if new_dir.exists():
                shutil.rmtree(new_dir, ignore_errors=True)
            old_dir.rename(new_dir)
            record["image_paths"] = [str(new_dir / Path(p).name) if p else ""
                                     for p in record.get("image_paths", [])]
        record["updated_at"] = _now()
        self.people[new_safe] = record
        self.save()
        return True
    @property
    def embedding_dim(self) -> Optional[int]:
        for record in self.people.values():
            emb = np.asarray(record.get("embeddings"))
            if emb.size:
                return int(emb.shape[1])
        return None
    @property
    def backbone(self) -> Optional[str]:
        return self.meta.get("backbone")
    def get_embeddings(self, name: str) -> Optional[np.ndarray]:
        record = self.people.get(sanitize_name(name))
        return None if record is None else np.asarray(record["embeddings"], dtype=np.float32)
    def list_people(self) -> List[dict]:
        rows = [{"姓名": n, "图片数": int(len(r.get("embeddings", []))),
                 "注册时间": r.get("created_at", ""), "更新时间": r.get("updated_at", "")}
                for n, r in self.people.items()]
        return sorted(rows, key=lambda row: (row["注册时间"], row["姓名"]))
    def names(self) -> List[str]:
        return [row["姓名"] for row in self.list_people()]
    def get_gallery(self) -> Tuple[List[str], np.ndarray]:
        names: List[str] = []
        feats: List[np.ndarray] = []
        for name in self.names():
            emb = self.get_embeddings(name)
            if emb is None or emb.size == 0:
                continue
            names += [name] * len(emb)
            feats.append(emb)
        if not feats:
            return [], np.zeros((0, self.embedding_dim or 512), dtype=np.float32)
        return names, np.vstack(feats).astype(np.float32)
    @property
    def num_people(self) -> int:
        return len(self.people)
    @property
    def num_embeddings(self) -> int:
        return int(sum(len(np.asarray(v.get("embeddings", []))) for v in self.people.values()))
    def stats(self) -> Dict[str, object]:
        return {"注册人数": self.num_people, "人脸特征总数": self.num_embeddings,
                "特征维度": self.embedding_dim, "骨干网络": self.backbone,
                "文件大小(KB)": (round(self.db_path.stat().st_size / 1024, 1)
                                if self.db_path.exists() else 0.0),
                "最后更新": self.updated_at}
    def get_meta(self, key: str, default: object = None) -> object:
        return self.meta.get(key, default)
