# 基于预训练模型的人脸识别 Demo

> 人工智能视觉课程大作业 · **选题二：基于预训练模型的人脸识别 Demo**
>
> 技术路线：**MTCNN 检测与对齐 → 预训练骨干提特征 → 余弦相似度比对 → 阈值判定（含陌生人拒识）→ Streamlit 界面**

项目地址：<https://gitee.com/muyu1213/face_demo>

---

## 一、功能与作业要求对照

| 作业要求 | 实现情况 | 对应位置 |
| --- | --- | --- |
| 人员注册：输入姓名，上传 **≥5 张**人脸图，MTCNN 检测裁剪，提特征入库 | ✅ 不足 5 张直接拒绝注册并说明原因 | `app.py` →「人员注册」 |
| 单人识别：图片 → 检测 → 提特征 → 余弦相似度 → 输出姓名与相似度 | ✅ 支持多张人脸逐张识别，可按人取最大 / Top-N 平均 | `app.py` →「人脸识别」 |
| 陌生人判断：最高相似度低于阈值输出「陌生人」 | ✅ 阈值默认 **0.6**，侧边栏可实时拖动 | `similarity.py` → `identify()` |
| 人脸库管理：查看已注册人员、删除指定人员 | ✅ 另附重命名、导出、阈值分析 | `app.py` →「人脸库管理」 |
| 简单界面：注册 / 识别 / 查看结果 / 管理库 | ✅ Streamlit 四个标签页 | `app.py` |
| 阈值可调 | ✅ 侧边栏滑块 + 基于库内样本的自动阈值扫描 | `app.py` / `similarity.py` |
| 数据持久化 | ✅ pickle 存特征与元数据，原子写入 + `.bak` 备份 | `face_database.py` |
| 骨干网络 | ✅ `inception_resnet_v1`（VGGFace2）与 `resnet18`（ImageNet）双实现 | `feature_extractor.py` |
| 关键步骤中文注释 | ✅ 每个模块开头有原理说明 | 全部 `.py` |

---

## 二、环境要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Windows 10/11（推荐双击 `start.bat`）；macOS / Linux 走手动步骤 |
| Python | **3.10 ~ 3.12**（推荐 3.11；3.13+ 的 PyTorch 轮子可能还没发布） |
| 内存 / 磁盘 | ≥ 8 GB 内存；磁盘预留 **4 GB**（`.venv` 约 2.5 GB + 权重 150 MB） |
| 网络 | **首次运行需联网**：装依赖 1~2 GB、下载模型权重约 150 MB |
| 显卡 | 不需要；默认 CPU 推理，有 NVIDIA 显卡会自动用 GPU（需 CUDA 版 PyTorch） |

---

## 三、安装与启动

### 方式 A：双击 `start.bat`（推荐，唯一入口）

它自动判断环境并按 4 步走（窗口显示 `[1/4]`~`[4/4]`）：

| 步骤 | 做什么 | 首次耗时 |
| --- | --- | --- |
| `[1/4]` | 找可用的 `uv` / Python（3.12 → 3.11 → 3.10 依次尝试） | 秒 |
| `[2/4]` | 建虚拟环境 `.venv`；已存在则体检，坏了自动重建 | 秒 |
| `[3/4]` | 装 `requirements.txt`（清华镜像优先，失败回退官方源；依赖齐全时走快通道） | 5~20 分钟 |
| `[4/4]` | 启动 Streamlit 并打开 <http://localhost:8501> | 秒 |

- 结束后**这个黑窗口不要关**，它就是服务器；关窗口或 `Ctrl+C` 即停止。
- 模型权重由程序首次用到时自动下载，**没有单独的预下载步骤**。
- 可选参数：`start.bat --setup`（强制重查依赖）、`--no-browser`（不自动开浏览器）、`--dry-run`（只看会做什么）。
- 常见故障：`'uv' 不是内部或外部命令` → 装 uv 后**重开命令行**；`error: externally-managed-environment` → 别加 `--break-system-packages`，用 `uv venv` + `uv pip install`；端口 8501 被占用 → `python -m streamlit run app.py --server.port 8502`。

### 方式 B：手动（macOS / Linux 或不想用 .bat）

```bash
python -m venv .venv
.venv\Scripts\activate                       # Windows；macOS/Linux 用 source .venv/bin/activate
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m streamlit run app.py
```

镜像源备选：清华 `https://pypi.tuna.tsinghua.edu.cn/simple`（默认）｜阿里云 `https://mirrors.aliyun.com/pypi/simple`｜中科大 `https://mirrors.ustc.edu.cn/pypi/simple`

---

## 四、使用流程

1. **人员注册** —— 填姓名 → 上传同一人的 ≥5 张不同角度照片 → 点「提取特征并注册」。不足 5 张会拦下并提示。
2. **人脸识别** —— 上传一张图片，自动检测并比对；绿框=已识别、红框=陌生人，下方显示姓名与相似度。
3. **人脸库管理** —— 查看人数/特征数/维度/文件大小，删除或重命名指定人员，下载 `face_db.pkl`，查看类内/类间相似度与建议阈值。
4. **原理与说明** —— 答辩速查页，一屏讲清流程、阈值取值与双骨干差异。

侧边栏可调参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| 特征提取骨干网络 | `inception_resnet_v1` | 换成 `resnet18` 必须**重新注册**（特征空间不通用） |
| 运行设备 | `auto` | 有显卡自动用显卡 |
| 相似度阈值 | `0.6` | 调高更严格（本人可能被拒），调低更宽松（可能误认） |
| 展示 Top-K 候选 | `5` | 识别页候选人数 |
| 汇总方式 / mean_topk 的 N | `max` / `3` | 同一人多张注册图取最大分或 Top-N 平均 |
| 最小人脸尺寸 / 裁剪边距 | `40px` / `20px` | 漏检就调小最小尺寸 |

---

## 五、文件结构

| 文件 | 作用 |
| --- | --- |
| **`start.bat`** | Windows 一键启动（唯一入口）：检测环境 → 建 venv → 装依赖 → 起服务 |
| `app.py` | Streamlit 主界面：4 个标签页 + 侧边栏参数 + 环境自检 |
| `face_detector.py` | MTCNN 检测、关键点对齐、160×160 裁剪、结果画框；兼图片格式转换 |
| `feature_extractor.py` | 双骨干特征提取，输出 L2 归一化 512 维向量 |
| `face_database.py` | 人脸库持久化：pickle 原子写入 + `.bak` 备份，增删改查、统计、导出 |
| `similarity.py` | 余弦相似度、1:N 检索、陌生人判定、阈值统计 |
| `requirements.txt` | 依赖清单 |
| `README.md` | 本文件 |

运行时生成（已在 `.gitignore`，不提交仓库）：

| 路径 | 内容 |
| --- | --- |
| `.venv/` | 虚拟环境（约 2.5 GB） |
| `.cache/torch/` | 模型权重缓存（约 150 MB），`feature_extractor.py` 固定指向此处，避免 `~/.cache` 不可写 |
| `face_db.pkl` / `face_db.bak.pkl` | 人脸库（特征 + 元数据）与自动备份，含隐私数据 |
| `registered_faces/` | 注册时保存的人脸小图（`<姓名>/<序号>.png`），含隐私数据 |

---

## 六、从 Gitee 获取与更新

```bash
git clone https://gitee.com/muyu1213/face_demo.git
cd face_demo
start.bat            # Windows 一键启动；依赖有变化会自动补装
```

更新：`git pull` 后重新运行 `start.bat`。
