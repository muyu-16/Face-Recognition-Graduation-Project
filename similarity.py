"""similarity.py —— 余弦相似度、1:N 识别与阈值评估。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
DEFAULT_THRESHOLD = 0.6     # 低于该值判为陌生人
UNKNOWN_LABEL = "陌生人"
def l2_normalize(x, axis=-1, eps=1e-10):
    """L2 归一化：把向量缩放到长度 1，避免零向量除零。"""
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)
def cosine_similarity(a, b, eps=1e-10) -> float:
    """两个向量的余弦相似度，范围 [-1, 1]。"""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        raise ValueError(f"特征维度不一致：{a.shape} vs {b.shape}，换过骨干网络请重新注册")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom < eps else float(np.dot(a, b) / denom)
def cosine_similarity_matrix(query, gallery) -> np.ndarray:
    """一个查询向量 vs 整个人脸库矩阵，返回 (N,) 相似度（矩阵乘法加速）。"""
    q = np.asarray(query, dtype=np.float32).ravel()
    g = np.asarray(gallery, dtype=np.float32)
    if g.ndim == 1:
        g = g.reshape(1, -1)
    if g.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if g.shape[1] != q.shape[0]:
        raise ValueError(f"维度不匹配：查询 {q.shape[0]}，人脸库 {g.shape[1]}，换骨干后请重新注册")
    q = q / max(float(np.linalg.norm(q)), 1e-10)
    g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-10)
    return (g @ q).astype(np.float32)
@dataclass
class MatchResult:
    """一次识别的完整结果（界面直接展示）。"""
    is_known: bool
    name: str
    similarity: float
    threshold: float
    top_k: List[Tuple[str, float]] = field(default_factory=list)
    all_scores: Optional[np.ndarray] = None
    matched_index: int = -1
    @property
    def margin(self) -> float:
        """第一名与第二名的分差，越大越可靠。"""
        return float("nan") if len(self.top_k) < 2 else float(self.top_k[0][1] - self.top_k[1][1])
def identify(query_embedding, names: Sequence[str], gallery: np.ndarray,
             threshold: float = DEFAULT_THRESHOLD, top_k: int = 5,
             aggregate: str = "max", mean_topk: int = 3) -> MatchResult:
    """比对一张人脸特征：max 取最像的一张，mean_topk 取前 N 张平均。"""
    names = list(names)
    scores = cosine_similarity_matrix(query_embedding, gallery)
    if scores.size == 0 or not names:
        return MatchResult(False, UNKNOWN_LABEL, 0.0, float(threshold), [], scores, -1)
    per_person: Dict[str, List[int]] = {}
    for idx, name in enumerate(names):
        per_person.setdefault(name, []).append(idx)
    ranked: List[Tuple[str, float, int]] = []
    for name, idxs in per_person.items():
        sims = [float(scores[i]) for i in idxs]
        if aggregate == "mean_topk":
            k = max(1, min(int(mean_topk), len(sims)))
            score = float(np.mean(sorted(sims, reverse=True)[:k]))
        else:
            score = max(sims)
        ranked.append((name, score, idxs[int(np.argmax(sims))]))
    ranked.sort(key=lambda item: item[1], reverse=True)
    best_name, best_score, best_idx = ranked[0]
    known = bool(best_score >= float(threshold))
    return MatchResult(known, best_name if known else UNKNOWN_LABEL, best_score,
                       float(threshold),
                       [(n, float(s)) for n, s, _ in ranked[:max(1, int(top_k))]],
                       scores, best_idx if known else -1)
def _pairwise(embeddings, names) -> Tuple[np.ndarray, np.ndarray]:
    """所有特征两两余弦相似度（上三角）及每对是否同人。"""
    emb = np.asarray(embeddings, dtype=np.float32)
    if len(emb) < 2:
        return np.zeros(0), np.zeros(0, dtype=bool)
    unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)
    upper = np.triu_indices(len(emb), k=1)
    name_arr = np.asarray(list(names))
    return (unit @ unit.T)[upper].astype(np.float64), name_arr[upper[0]] == name_arr[upper[1]]
def _stat(values) -> Dict[str, float]:
    if values.size == 0:
        return {"数量": 0, "均值": float("nan"), "最小": float("nan"), "最大": float("nan")}
    return {"数量": int(values.size), "均值": float(np.mean(values)),
            "最小": float(np.min(values)), "最大": float(np.max(values))}
def similarity_stats(embeddings, names) -> Dict[str, float]:
    """类内（同人）/ 类间（异人）相似度分布 + 建议阈值。"""
    sims, same = _pairwise(embeddings, names)
    intra, inter = _stat(sims[same]), _stat(sims[~same])
    suggestion = float("nan")
    if intra["数量"] and inter["数量"]:
        suggestion = float((intra["最小"] + inter["最大"]) / 2.0)
    return {"类内相似度": intra, "类间相似度": inter, "建议阈值": suggestion}
def find_best_threshold(embeddings, names, steps: int = 199) -> Dict[str, float]:
    """在 [0,1] 上扫描阈值，返回成对判断准确率最高的阈值。"""
    sims, same = _pairwise(embeddings, names)
    if sims.size == 0:
        return {"最佳阈值": DEFAULT_THRESHOLD, "最佳准确率": float("nan")}
    same_sims, diff_sims = np.sort(sims[same]), np.sort(sims[~same])
    thresholds = np.linspace(0.0, 1.0, int(steps))
    hit = ((same_sims.size - np.searchsorted(same_sims, thresholds, side="left"))
           + np.searchsorted(diff_sims, thresholds, side="left"))
    accs = hit / float(sims.size)
    best = int(np.argmax(accs))
    return {"最佳阈值": float(thresholds[best]), "最佳准确率": float(accs[best])}
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    v = rng.normal(size=512).astype(np.float32)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-5
    names, feats = [], []
    for person in ["张三", "李四", "王五"]:
        base = rng.normal(size=512).astype(np.float32); base /= np.linalg.norm(base)
        for _ in range(3):
            names.append(person)
            feats.append(l2_normalize(base + 0.15 * rng.normal(size=512).astype(np.float32)))
    gallery = np.stack(feats)
    print("识别结果：", identify(gallery[0], names, gallery, threshold=0.6, top_k=3))
    print("陌生人结果：", identify(l2_normalize(rng.normal(size=512).astype(np.float32)),
                                 names, gallery, threshold=0.6))
    print("相似度统计：", similarity_stats(gallery, names))
    print("最佳阈值扫描：", find_best_threshold(gallery, names))
