"""app.py —— 基于预训练模型的人脸识别 Demo（Streamlit 主界面）。
流程：MTCNN 检测 + 对齐裁剪 160x160 -> 骨干提 512 维特征并 L2 归一化
      -> 与库内特征算余弦相似度 -> 最高分 >= 阈值判本人，否则「陌生人」。
运行：streamlit run app.py
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

def _check_env() -> None:
    """缺库 / 启动方式不对时给中文提示，而不是一屏 traceback（必须在第三方 import 之前）。"""
    import importlib.util
    import sys
    need = [("numpy", "numpy"), ("pandas", "pandas"), ("PIL", "Pillow"), ("streamlit", "streamlit"),
            ("torch", "torch"), ("torchvision", "torchvision"), ("facenet_pytorch", "facenet-pytorch")]
    missing = [pkg for mod, pkg in need if importlib.util.find_spec(mod) is None]
    if missing:
        print("=" * 72)
        print(f"❌ 当前 Python（{sys.executable}）缺少依赖：" + "、".join(missing))
        print("   请执行：pip install -r requirements.txt   或双击 start.bat")
        print("=" * 72)
        raise SystemExit(1)
    try:
        from streamlit.runtime import exists
        if not exists():
            print("⚠️  这是 Streamlit 应用，请用：streamlit run app.py（或双击 start.bat）")
            raise SystemExit(1)
    except ImportError:
        pass
_check_env()
import numpy as np
import pandas as pd
import streamlit as st
from face_database import FaceDatabase, sanitize_name
from face_detector import FaceDetector, to_rgb_pil
from feature_extractor import FeatureExtractor
from similarity import DEFAULT_THRESHOLD, find_best_threshold, identify, similarity_stats
st.set_page_config(page_title="基于预训练模型的人脸识别 Demo", page_icon="🧑",
                   layout="wide", initial_sidebar_state="expanded")
APP_TITLE = "基于预训练模型的人脸识别 Demo"
DB_PATH, FACE_DIR = Path("face_db.pkl"), Path("registered_faces")
MIN_REGISTER_IMAGES = 5          # 作业要求：每人至少 5 张人脸
BACKBONES = {"inception_resnet_v1": "InceptionResnetV1（face 预训练，推荐）",
             "resnet18": "ResNet18（ImageNet 预训练，按作业字面要求）"}
AGGREGATES = {"max": "取最大相似度（推荐）", "mean_topk": "取 Top-N 平均（更稳）"}
@st.cache_resource(show_spinner=False)
def load_detector(device: str, min_face_size: int, margin: int, align: bool) -> FaceDetector:
    return FaceDetector(image_size=160, margin=margin, min_face_size=min_face_size,
                        device=device, align=align, keep_all=True)
@st.cache_resource(show_spinner=False)
def load_extractor(backbone: str, device: str) -> FeatureExtractor:
    return FeatureExtractor(backbone=backbone, device=device,
                            pretrained="vggface2" if backbone == "inception_resnet_v1" else True)
@st.cache_resource(show_spinner=False)
def load_database(db_path: str, face_dir: str) -> FaceDatabase:
    return FaceDatabase(db_path=Path(db_path), face_dir=Path(face_dir), auto_load=True)
def run_recognition(image, detector: FaceDetector, extractor: FeatureExtractor,
                    db: FaceDatabase, cfg: Dict[str, object]) -> Optional[dict]:
    """检测 + 提特征 + 与库比对，返回结果（不渲染）。"""
    faces, crops = detector.detect_and_crop(image, max_faces=int(cfg["max_faces"]))
    if not faces:
        return None
    embeddings = extractor.extract_batch(crops)
    names, gallery = db.get_gallery()
    results = [identify(emb, names, gallery, threshold=float(cfg["threshold"]),
                        top_k=int(cfg["top_k"]), aggregate=str(cfg["aggregate"]),
                        mean_topk=int(cfg["mean_topk"])) for emb in embeddings]
    canvas = detector.draw(image, faces, unknown_flags=[not r.is_known for r in results])
    return {"faces": faces, "results": results, "canvas": canvas}
def render_sidebar() -> Dict[str, object]:
    with st.sidebar:
        st.title("⚙️ 参数设置")
        backbone = st.selectbox("特征提取骨干网络", list(BACKBONES), format_func=BACKBONES.get,
                                help="切换骨干后特征空间不同，需要重新注册人脸库。")
        device = st.selectbox("运行设备", ["auto", "cpu", "cuda"],
                              help="auto = 有显卡自动用显卡。CPU 也能跑，只是慢一些。")
        st.divider()
        threshold = st.slider("相似度阈值（低于该值判为陌生人）", 0.0, 1.0,
                              float(DEFAULT_THRESHOLD), 0.01,
                              help="调高更严格（本人可能被拒），调低更宽松（可能误认）。")
        top_k = st.slider("展示 Top-K 候选", 1, 10, 5, 1)
        aggregate = st.radio("同一人多张注册图的汇总方式", list(AGGREGATES),
                             format_func=AGGREGATES.get)
        mean_topk = st.slider("mean_topk 的 N", 1, 5, 3, 1, disabled=(aggregate != "mean_topk"))
        min_face_size = st.slider("最小人脸尺寸(px)", 20, 200, 40, 5)
        margin = st.slider("裁剪边距(px)", 0, 80, 20, 5)
        st.divider()
        if st.button("🔄 重新加载人脸库", width="stretch"):
            load_database.clear()
            st.rerun()
        st.caption("首次运行会自动下载预训练权重，需要联网。")
    return {"backbone": backbone, "device": device, "threshold": float(threshold), "top_k": int(top_k),
            "aggregate": aggregate, "mean_topk": int(mean_topk), "min_face_size": int(min_face_size),
            "margin": int(margin), "align": True, "max_faces": 3}
def render_register_tab(cfg: Dict[str, object], detector: FaceDetector,
                        extractor: FeatureExtractor, db: FaceDatabase) -> None:
    """人员注册：输入姓名，上传至少 5 张人脸图，提特征入库。"""
    st.subheader("🧑‍🎓 人员注册")
    st.markdown(f"流程：**输入姓名 → 上传至少 {MIN_REGISTER_IMAGES} 张人脸图片 → MTCNN 检测裁剪 → "
                "提取 512 维特征 → 保存到人脸库**")
    col_name, col_upload = st.columns([1, 1])
    with col_name:
        name = st.text_input("① 姓名", placeholder="例如：张三", max_chars=32)
        overwrite = st.checkbox("覆盖该人已有特征（重新注册）", value=False)
    with col_upload:
        files = st.file_uploader(f"② 人脸图片（同一人不同角度，至少 {MIN_REGISTER_IMAGES} 张）",
                                 type=["jpg", "jpeg", "png", "bmp", "webp"],
                                 accept_multiple_files=True, key="reg_uploader")
    candidates = [(f.name, to_rgb_pil(f)) for f in (files or [])]
    st.divider()
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        do_register = st.button("🚀 提取特征并注册", type="primary", width="stretch",
                                disabled=not (name and len(candidates) >= MIN_REGISTER_IMAGES))
    with col_info:
        if not name:
            st.warning("请先填写姓名")
        elif len(candidates) < MIN_REGISTER_IMAGES:
            st.warning(f"当前 {len(candidates)} 张，作业要求每人至少 {MIN_REGISTER_IMAGES} 张")
        else:
            st.success(f"已准备 {len(candidates)} 张图片，可以开始注册")
    if not do_register:
        return
    progress = st.progress(0.0, text="开始处理……")
    crops, embeddings = [], []
    for idx, (label, image) in enumerate(candidates, start=1):
        progress.progress((idx - 1) / len(candidates), text=f"正在处理 {label}（{idx}/{len(candidates)}）")
        faces = detector.detect(image)
        if not faces:
            continue
        crop = detector.crop(image, faces[0])
        crops.append(crop)
        embeddings.append(extractor.extract(crop))
    progress.progress(1.0, text="处理完成")
    if len(embeddings) < MIN_REGISTER_IMAGES:
        st.error(f"❌ 只有 {len(embeddings)} 张成功提取特征，少于要求的 {MIN_REGISTER_IMAGES} 张，本次未注册。建议换正面、清晰、光照均匀的照片后重试。")
    else:
        try:
            info = db.add_person(name=name, embeddings=np.stack(embeddings, axis=0),
                                 face_images=crops, overwrite=overwrite, backbone=cfg["backbone"])
            st.success(f"🎉 注册成功！『{info['姓名']}』当前共 {info['图片数']} 条人脸特征")
            st.balloons()
        except Exception as err:
            st.error(f"注册失败：{err}")
def render_recognize_tab(cfg: Dict[str, object], detector: FaceDetector,
                         extractor: FeatureExtractor, db: FaceDatabase) -> None:
    """检测 -> 提特征 -> 与库比对 -> 输出姓名和相似度 / 陌生人。"""
    st.subheader("🔍 人脸识别")
    threshold = float(cfg["threshold"])
    st.markdown(f"当前阈值 **{threshold:.2f}**：最高相似度 ≥ 阈值判为已知人员，否则输出**陌生人**。")
    if db.num_people == 0:
        st.warning("人脸库还是空的，请先到「人员注册」标签页注册至少一个人。")
        return
    names, gallery = db.get_gallery()
    st.caption(f"人脸库：{db.num_people} 人 / {len(names)} 条特征 / 特征维度 {gallery.shape[1]} / "
               f"骨干网络 {db.backbone or cfg['backbone']}")
    if db.embedding_dim is not None and db.embedding_dim != extractor.embedding_dim:
        st.error(f"⚠️ 人脸库维度（{db.embedding_dim}）与当前骨干输出（{extractor.embedding_dim}）不一致，请重新注册人脸库。")
        return
    uploaded = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                key="rec_uploader")
    if uploaded is None:
        st.info("请上传图片，识别会自动开始。")
        return
    image = to_rgb_pil(uploaded)
    with st.spinner("正在检测人脸并提取特征……"):
        outcome = run_recognition(image, detector, extractor, db, cfg)
    if outcome is None:
        st.image(image, caption="未检测到人脸", width="stretch")
        st.error("❌ 没有检测到人脸。请换一张更清晰、人脸更大的图片，或把侧边栏「最小人脸尺寸」调小后重试。")
        return
    main = outcome["results"][0]
    col_img, col_res = st.columns([1.1, 1])
    with col_img:
        st.image(outcome["canvas"], caption="识别结果（绿框=已识别，红框=陌生人）", width="stretch")
    with col_res:
        if main.is_known:
            st.success(f"✅ 识别结果：**{main.name}**\n\n相似度 **{main.similarity:.4f}** ≥ {threshold:.2f}")
        else:
            st.error(f"🚫 识别结果：**陌生人**\n\n最高相似度 **{main.similarity:.4f}** < {threshold:.2f}")
def render_database_tab(cfg: Dict[str, object], db: FaceDatabase) -> None:
    """查看人员、删除、重命名、导出、阈值分析。"""
    st.subheader("🗂️ 人脸库管理")
    stats = db.stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("注册人数", stats["注册人数"])
    col2.metric("人脸特征总数", stats["人脸特征总数"])
    col3.metric("特征维度", stats["特征维度"] or "-")
    col4.metric("文件大小(KB)", stats["文件大小(KB)"])
    if db.last_error or db.get_meta("image_save_error"):
        st.warning(str(db.last_error or db.get_meta("image_save_error")))
    if db.num_people == 0:
        st.info("人脸库为空。请先到「人员注册」标签页注册人员。")
        return
    st.dataframe(pd.DataFrame(db.list_people()), width="stretch", hide_index=True)
    st.divider()
    col_del, col_ren = st.columns(2)
    with col_del:
        st.markdown("#### 🗑️ 删除指定人员")
        target = st.selectbox("选择要删除的人员", db.names(), key="del_target")
        confirm = st.checkbox("我确认删除该人员的全部特征与图片", key="del_confirm")
        if st.button("删除该人员", type="primary", disabled=(not confirm or not target)):
            try:
                if db.delete_person(target):
                    st.success(f"已删除『{target}』")
                    st.rerun()
                else:
                    st.error("删除失败：未找到")
            except Exception as err:
                st.error(f"删除失败：{err}")
    with col_ren:
        st.markdown("#### ✏️ 重命名")
        old_name = st.selectbox("选择人员", db.names(), key="ren_old")
        new_name = st.text_input("新姓名", key="ren_new")
        if st.button("重命名", disabled=not new_name):
            try:
                if db.rename_person(old_name, new_name):
                    st.success(f"已改名为『{sanitize_name(new_name)}』")
                    st.rerun()
                else:
                    st.error(f"重命名失败：未找到『{old_name}』或新姓名不合法")
            except Exception as err:
                st.error(f"重命名失败：{err}")
    st.download_button("💾 下载 face_db.pkl", data=db.export_bytes(), file_name="face_db.pkl",
                       mime="application/octet-stream",
                       help="拷到另一台机器即可复用人脸库（骨干网络需一致）。")
    st.divider()
    st.markdown("#### 📈 阈值分析（基于已注册样本自动统计）")
    names, gallery = db.get_gallery()
    if len(names) < 4 or len(set(names)) < 2:
        st.caption("至少注册 2 个人、合计 4 条以上特征才能做类内/类间相似度统计。")
        return
    stats_sim, best = similarity_stats(gallery, names), find_best_threshold(gallery, names)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("类内相似度均值", f"{stats_sim['类内相似度']['均值']:.4f}")
    col_b.metric("类间相似度均值", f"{stats_sim['类间相似度']['均值']:.4f}")
    col_c.metric("建议阈值", f"{stats_sim['建议阈值']:.3f}")
    st.caption(f"当前阈值 {cfg['threshold']:.2f}；扫描最佳阈值约 {best['最佳阈值']:.3f}（成对判断准确率 {best['最佳准确率'] * 100:.1f}%）。")
def render_about_tab(detector: FaceDetector, extractor: FeatureExtractor, db: FaceDatabase) -> None:
    st.subheader("📖 原理与工程说明（答辩速查）")
    st.markdown("""
**流程**：MTCNN 检测人脸与 5 个关键点 → 按双眼连线转正并裁成 160×160 → 预训练骨干输出 512 维特征并 L2 归一化（余弦相似度 = 向量点积）→ 与库内特征比对、按人取最高分 → 最高分 ≥ 阈值输出姓名，否则「陌生人」。

**阈值**：调高更严格、调低更宽松；FaceNet 系列常用 0.6~0.75，本 Demo 默认 0.6。

**骨干**：`inception_resnet_v1`（VGGFace2 人脸预训练，默认，准确率高）与 `resnet18`（ImageNet 通用预训练，符合作业「ResNet18 骨干」字面要求）都实现；两者特征空间不通用，换骨干必须重新注册人脸库。
    """)
    st.json({"检测器": str(detector.device), "特征提取器": extractor.describe(), "人脸库": db.stats()})


def main() -> None:
    cfg = render_sidebar()
    st.title("🧑 " + APP_TITLE)
    st.caption("人工智能视觉课程大作业 · 选题二：基于预训练模型的人脸识别 Demo ｜ MTCNN + InceptionResnetV1(ResNet18) + 余弦相似度 + Streamlit")
    try:
        detector = load_detector(cfg["device"], cfg["min_face_size"], cfg["margin"], cfg["align"])
        extractor = load_extractor(cfg["backbone"], cfg["device"])
    except Exception as err:
        st.error(f"❌ 模型初始化失败：{err}")
        st.stop()
    db = load_database(str(DB_PATH), str(FACE_DIR))
    st.caption(f"已注册 {db.num_people} 人 / {db.num_embeddings} 条特征 ｜ 阈值 {cfg['threshold']:.2f} ｜ 设备 {extractor.device}")
    tabs = st.tabs(["🧑‍🎓 人员注册", "🔍 人脸识别", "🗂️ 人脸库管理", "📖 原理与说明"])
    with tabs[0]:
        render_register_tab(cfg, detector, extractor, db)
    with tabs[1]:
        render_recognize_tab(cfg, detector, extractor, db)
    with tabs[2]:
        render_database_tab(cfg, db)
    with tabs[3]:
        render_about_tab(detector, extractor, db)
    st.caption(f"运行：streamlit run app.py ｜ 人脸库：{DB_PATH.resolve()}")
if __name__ == "__main__":
    main()
