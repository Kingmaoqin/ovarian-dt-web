# Stage-1B & Stage-1C 技术报告

> **项目**：BIM → Medical Digital Twin (MDBIMDT)
> **模块**：Stage-1B 多器官语义表面图谱构建 / Stage-1C LLM 语义渲染先验生成
> **脚本**：`scripts/stage1b_build_semantic_graph.py` / `scripts/stage1c_llm_semantic_priors.py`
> **输出**：`data/processed/stage1b_semantic_model/` / `data/processed/stage1c_llm_priors/`

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [数据集说明](#2-数据集说明)
3. [Stage-1B：多器官语义图谱构建](#3-stage-1b多器官语义图谱构建)
   - 3.1 [Phase A：强度与形状统计](#31-phase-a强度与形状统计)
   - 3.2 [Phase B：解剖拓扑模板](#32-phase-b解剖拓扑模板)
   - 3.3 [Phase C：逐病例语义表面图](#33-phase-c逐病例语义表面图)
   - 3.4 [可视化产物](#34-可视化产物)
   - 3.5 [交付物汇总](#35-交付物汇总)
4. [Stage-1C：LLM 语义渲染先验生成](#4-stage-1cllm-语义渲染先验生成)
   - 4.1 [输入与输出](#41-输入与输出)
   - 4.2 [LLM 调用机制](#42-llm-调用机制)
   - 4.3 [规则回退（Mock 模式）](#43-规则回退mock-模式)
   - 4.4 [参数约束与 Clamp 处理](#44-参数约束与-clamp-处理)
   - 4.5 [交付物汇总](#45-交付物汇总)
5. [两个 Stage 的接口关系](#5-两个-stage-的接口关系)
6. [当前局限与后续计划](#6-当前局限与后续计划)
7. [运行方式](#7-运行方式)

---

## 1. 背景与动机

本项目的目标是将建筑行业的 **BIM（Building Information Model）** 思路引入腹腔手术场景，为外科医生提供一套实时更新的 **医学数字孪生（Medical Digital Twin）**。在这套三路径架构（BIM 快速路径 / 可变形配准 / nnUNet 重分割）中，Stage-1B 和 Stage-1C 承担 **"术前知识准备"** 职责：

- **Stage-1B** 回答：*每种器官在 CT 影像中长什么样（亮度、体积、形状、空间邻接）？*
- **Stage-1C** 回答：*在三维渲染展示时，每种器官应以怎样的视觉参数显示？*

这两个模块的产出是后续所有动态追踪模块的"解剖先验知识库"，类似于外科医生术前阅读图谱——让系统在手术开始之前就"知道"正常人体腹部的解剖规律。

---

## 2. 数据集说明

### 2.1 KiTS23（原始 CT 影像来源）

| 属性 | 内容 |
|------|------|
| 全称 | Kidney Tumor Segmentation Challenge 2023 |
| 原始标注 | 肾脏 / 肾肿瘤 / 肾囊肿（3 类） |
| **Stage-1B 使用方式** | **仅使用 CT 影像文件**（`imaging.nii.gz`），不使用原始肿瘤标注 |
| 病例数 | 489 例完整腹部 CT 扫描 |
| 视野范围 | 胸腔至骨盆，覆盖绝大多数腹部器官 |
| 路径 | `data/raw/KiTS23/kits23/dataset/case_XXXXX/imaging.nii.gz` |

> **关键区分**：Stage-1B 不使用 KiTS23 的肿瘤标注。KiTS23 仅作为高质量腹部 CT 扫描的来源，其完整视野使得 TotalSegmentator 能够分割出 110+ 类结构。

### 2.2 TotalSegmentator 伪标签（主要输入）

| 属性 | 内容 |
|------|------|
| 工具版本 | TotalSegmentator v2.12.0（6 个子模型，104 类） |
| 生成方式 | 对 489 例 KiTS23 CT 图像逐例运行 TotalSegmentator |
| 输出格式 | 每个 case 目录下 117 个 `.nii.gz` 文件，每个文件为一类结构的二值分割掩膜 |
| 总标签类别 | 110 类（部分 case 因视野截断，部分类别为空） |
| 路径 | `data/processed/totalseg_labels/case_XXXXX/<label_name>.nii.gz` |
| 标签涵盖 | 肝脏、脾脏、肾脏、主动脉、各段椎体、肺叶、肋骨、肌肉等全身结构 |

### 2.3 实际使用规模（Stage-1B 默认参数）

```
--n_sample 20   # 随机抽取 20 例用于形状统计（Phase A）
--n_topo   0    # 使用文献解剖先验，跳过数据驱动拓扑（Phase B）
--n_graph  2    # 为 2 例构建完整语义图（Phase C）
```

---

## 3. Stage-1B：多器官语义图谱构建

脚本执行三个顺序阶段（Phase A → B → C），加可视化和摘要输出。

### 3.1 Phase A：强度与形状统计

#### 3.1.1 目标

对 110 类解剖结构，估计每类的：
- **CT 亮度先验**：均值 $\mu_k$、标准差 $\sigma_k$（Hounsfield Unit，HU）
- **形状先验**：平均体积 $\bar{V}_k$、球形度 $\Psi_k$
- **强度容差阈值** $\theta_k$：用于下游 BIM 快速路径的边界吸附判断

#### 3.1.2 两种运行模式

**快速模式（默认，`collect_intensity_stats_fast`）**

只加载分割掩膜（不加载 CT 灰度），HU 值直接取文献先验常数：

```
对每个采样 case_dir：
  对每个 label.nii.gz：
    加载二值掩膜 M ∈ {0,1}^{H×W×D}
    读取体素间距 (dx, dy, dz)（来自 NIfTI 头部，不解压体数据）
    n_vox = Σ M
    volume = n_vox × dx × dy × dz   [单位: mm³]
    HU 值 → 查文献表 LITERATURE_HU[label]
```

优点：每例处理时间约 0.5s，不需要 CT 解压（节约 ~4s/case）。
缺点：HU 值非真实测量，为近似值。

**精确模式（`--use_ct` 参数，`collect_intensity_stats`）**

加载 CT 灰度体数据，逐像素计算 HU 统计：

对第 $i$ 个体素（属于类别 $k$），其 HU 值为 $h_i$。对 $N_k$ 个采样病例的 $n_k$ 个体素：

**均值**：

$$\mu_k = \frac{1}{n_k} \sum_{i=1}^{n_k} h_i$$

- $\mu_k$：类别 $k$ 的 CT 平均亮度（HU 均值）
- $n_k$：类别 $k$ 在全部采样病例中的总体素数
- $h_i$：第 $i$ 个属于类别 $k$ 的体素的 HU 值（范围约 -1000 至 +3000）

**标准差（在线累积公式，避免两次遍历）**：

$$\sigma_k = \sqrt{\frac{\sum_{i=1}^{n_k} h_i^2}{n_k} - \mu_k^2}$$

- $\sigma_k$：类别 $k$ 的 HU 标准差，反映该器官内部灰度的异质性
- $\sum h_i^2$：在遍历体素时同步累积，无需二次读取 CT 数据

代码中对应：
```python
acc["sum_hu"]    += float(hu_vals.sum())       # 累加 Σh_i
acc["sum_sq_hu"] += float((hu_vals ** 2).sum()) # 累加 Σh_i²
# 最终:
mean_hu = acc["sum_hu"] / n_hu
std_hu  = sqrt(max(acc["sum_sq_hu"] / n_hu - mean_hu**2, 0))
```

#### 3.1.3 强度容差阈值 $\theta_k$

$$\theta_k = 2\sigma_k$$

- $\theta_k$：类别 $k$ 的 HU 容差窗口（半宽度）
- 含义：在 BIM 快速路径（Stage-3c）中，判断一个体素是否属于类别 $k$ 时，允许其 HU 值在 $[\mu_k - \theta_k,\ \mu_k + \theta_k]$ 范围内。选取 $2\sigma$ 而非 $1\sigma$ 是为了覆盖约 95% 的正常组织像素，减少假阴性。

#### 3.1.4 体积统计

对每个病例 $j$ 中类别 $k$ 的掩膜：

$$V_k^{(j)} = n_{k,j} \cdot \prod_{d \in \{x,y,z\}} \Delta d$$

- $V_k^{(j)}$：病例 $j$ 中类别 $k$ 的体积（mm³）
- $n_{k,j}$：病例 $j$ 中类别 $k$ 的体素数量
- $\Delta x, \Delta y, \Delta z$：体素的三维物理尺寸（毫米），从 NIfTI 头部读取

跨 $N_k$ 个病例的平均体积和标准差：

$$\bar{V}_k = \frac{1}{N_k}\sum_{j=1}^{N_k} V_k^{(j)}, \quad s_{V,k} = \sqrt{\frac{1}{N_k}\sum_{j=1}^{N_k}(V_k^{(j)} - \bar{V}_k)^2}$$

- $\bar{V}_k$：类别 $k$ 的跨病例平均体积（mm³）
- $s_{V,k}$：跨病例体积标准差，反映器官大小的个体差异

#### 3.1.5 球形度（Sphericity）

对于每个器官掩膜，首先用 **Marching Cubes 算法** 提取等值面三角网格，再计算球形度：

$$\Psi = \frac{\pi^{1/3} \cdot (6V)^{2/3}}{A}$$

- $\Psi \in (0, 1]$：球形度，$\Psi = 1$ 代表完美球形，$\Psi \to 0$ 代表极度不规则形状
- $V$：器官的真实体积（mm³），由体素计数 × 体素体积得到
- $A$：三角网格的表面积（mm²），由 `skimage.measure.mesh_surface_area(verts, faces)` 计算
- $\pi^{1/3}(6V)^{2/3}$：与同体积球体的表面积，球体的表面积 $= \pi^{1/3}(6V)^{2/3}$

**Marching Cubes 原理简述**：
将体素掩膜视为标量场，在值 0.5 处插值提取等值面。对每个 $2\times2\times2$ 体素立方体，根据 8 个顶点的 0/1 状态（共 $2^8=256$ 种构型，归并后 15 种基本构型），生成 0~5 个三角面片。体素间距 $(dx,dy,dz)$ 传入 `spacing` 参数，使顶点坐标直接以 mm 为单位。

> **当前状态**：由于球形度计算耗时（每个掩膜需运行 Marching Cubes），默认关闭（`--compute_sphericity` 为可选参数）。对于 `n_vox > 50,000` 的大器官（肝脏、小肠、结肠等），计算开销尤高，代码中直接跳过，导致这些器官球形度报告为 0（已在图表中用文献填补值修正）。

---

### 3.2 Phase B：解剖拓扑模板

#### 3.2.1 目标

构建 $\mathcal{G}_{topo} = (\mathcal{V}, \mathcal{E})$，其中：
- 节点 $v \in \mathcal{V}$：解剖结构类别
- 边 $(v_a, v_b) \in \mathcal{E}$：两类结构在空间上紧密相邻（有物理接触）
- 边属性：出现频率 $f_{ab}$（在多少比例的病例中存在此邻接）、平均接触体素数 $c_{ab}$

#### 3.2.2 文献先验模式（`--n_topo 0`，当前使用）

直接编码来自解剖学文献的 44 条标准邻接关系，例如：

```
胆囊 ↔ 肝脏    (胆囊附着于肝脏下缘)
主动脉 ↔ 下腔静脉  (主动脉与下腔静脉平行伴行)
脾脏 ↔ 胃      (脾脏位于胃大弯左侧)
```

每条边的 `freq_ratio = 1.0`，`mean_contact_vox = 100`（占位符）。

#### 3.2.3 数据驱动模式（`--n_topo N`，N > 0）

当指定实际处理的病例数时，对每个 case 执行以下接触检测算法。

**算法：两阶段流式接触检测**

**第一阶段（Pass 1）：轻量 Bounding Box 扫描**

对 case 目录下每个 `label.nii.gz`，只提取非零区域的轴对齐包围盒（AABB）：

$$\text{bbox}(M) = \left(\min_{(i,j,k): M_{ijk}=1}(i,j,k),\ \max_{(i,j,k): M_{ijk}=1}(i,j,k)\right)$$

- $M \in \{0,1\}^{H\times W\times D}$：类别 $k$ 的二值分割掩膜
- AABB 由两个三维角点 $(lo_x, lo_y, lo_z)$ 和 $(hi_x, hi_y, hi_z)$ 表示
- 此阶段不保留掩膜数据，内存占用极低（仅 6 个整数/类别）

**AABB 快速过滤（剪枝）**：

对候选对 $(a, b)$，当且仅当满足以下条件时，才值得进行下一步精确检测：

$$\forall d \in \{x,y,z\}: lo_d^a - m \leq hi_d^b \;\wedge\; lo_d^b - m \leq hi_d^a$$

- $m = 2 \times \text{dilation\_iter} + 3$：扩张操作的最大影响半径，作为安全边距
- 该公式等价于：两个 AABB 在加上边距 $m$ 后仍有重叠
- 不满足此条件的对 $(a,b)$ 直接跳过，无需加载掩膜

**第二阶段（Pass 2）：裁剪区域精确接触检测**

对通过 AABB 过滤的候选对 $(a, b)$：

1. 计算共享裁剪区域（两个 AABB 的并集 + 边距 $m$）：

$$\text{crop\_lo} = \max\left(\min(lo^a, lo^b) - m,\ \mathbf{0}\right)$$
$$\text{crop\_hi} = \min\left(\max(hi^a, hi^b) + m,\ \text{vol\_shape}\right)$$

- $\text{crop\_lo}, \text{crop\_hi}$：裁剪区域的起始和终止坐标（体素单位）
- 对小器官而言，裁剪区域远小于完整体积（$512^3$ → 约 $50^3$），是加速的关键

2. 在裁剪区域内对掩膜 $a$ 做二值膨胀：

$$M_a^{\text{dil}} = M_a^{\text{crop}} \oplus B_r$$

- $\oplus$：二值膨胀运算（Minkowski sum）
- $B_r$：半径为 $r = \text{dilation\_iter}$ 的球形结构元素（`skimage.morphology.ball(r)`）
- 膨胀的目的：将器官 $a$ 的边界向外扩展若干体素，使其能"接触"到距离较近但未直接相邻的器官 $b$

3. 计算接触体素数：

$$c_{ab} = \left| M_a^{\text{dil}} \cap M_b^{\text{crop}} \right| = \sum_{(i,j,k)} \left[ M_a^{\text{dil}}(i,j,k) \wedge M_b^{\text{crop}}(i,j,k) \right]$$

- $c_{ab}$：膨胀后器官 $a$ 与器官 $b$ 的重叠体素数量
- 若 $c_{ab} \geq \text{min\_contact\_vox}$（默认 5），则认为 $a$ 和 $b$ 相邻，写入边

4. 跨多个病例聚合，计算频率：

$$f_{ab} = \frac{\text{count}_{ab}}{N_{\text{topo}}}$$

- $f_{ab}$：类别 $a$ 与类别 $b$ 在 $N_{\text{topo}}$ 个病例中相邻的比例
- $\bar{c}_{ab} = \text{mean}(\{c_{ab}^{(j)}\})$：平均接触体素数

**算法复杂度分析**

| 步骤 | 朴素方法 | 本实现 |
|------|---------|--------|
| 候选对数量 | $O(L^2)$（$L \approx 110$，$\sim 6000$ 对） | AABB 过滤后约 $100\sim200$ 对 |
| 膨胀操作的体积 | $H \times W \times D \approx 512^3$ | $\sim 50^3$（裁剪后） |
| 综合加速比 | 基准 | $\sim 100\times$ |

---

### 3.3 Phase C：逐病例语义表面图

#### 3.3.1 目标

对每个 CT 病例构建一个语义图 $\mathcal{G}_{\text{case}} = (\mathcal{V}_{\text{case}}, \mathcal{E}_{\text{case}})$，其中每个节点携带来自 Phase A 的语义属性，每条边携带拓扑关系。

#### 3.3.2 节点属性计算

对每个类别 $k$ 的二值掩膜 $M_k$：

**质心坐标（体素空间转物理空间）**：

$$\mathbf{c}_k^{\text{vox}} = \frac{1}{n_k}\sum_{(i,j,k): M_{ijk}=1}\begin{pmatrix}i\\j\\k\end{pmatrix}$$

$$\mathbf{c}_k^{\text{mm}} = A_{\text{affine}} \cdot \begin{pmatrix}\mathbf{c}_k^{\text{vox}} \\ 1\end{pmatrix}$$

- $\mathbf{c}_k^{\text{vox}} \in \mathbb{R}^3$：体素空间中的质心坐标
- $A_{\text{affine}} \in \mathbb{R}^{4\times4}$：NIfTI 仿射矩阵，将体素坐标映射到 mm 空间
- $\mathbf{c}_k^{\text{mm}} \in \mathbb{R}^3$：物理空间中的质心坐标（单位：mm），供后续 BIM 快速路径使用

最终节点结构：

```json
{
  "label": "liver",
  "organ_system": "parenchymal",
  "n_voxels": 234567,
  "volume_mm3": 1872456.0,
  "centroid_mm": [-45.2, -120.3, 88.7],
  "mean_hu": 60.0,
  "std_hu": 25.0,
  "theta_k": 50.0
}
```

#### 3.3.3 边属性

从 `topology_templates.json` 中筛选本 case 实际存在的类别对，写入边列表：

```json
{"node_a": "liver", "node_b": "gallbladder", "contact_voxels": 100}
```

---

### 3.4 可视化产物

| 文件名 | 类型 | 内容 |
|--------|------|------|
| `intensity_stats_heatmap.png` | 条形图 + 箱线图 | 36 类命名器官的 HU 均值±std（左）；7 个系统的 HU 范围（右） |
| `topology_graph.png` | 网络图 | 44 条解剖邻接边，按系统分区布局，边宽=接触面积 |
| `shape_stats.png` | 双条形图 | 前 30 类器官的平均体积（左）和球形度（右），文献填充值用斜线标注 |
| `sample_case_3d_organs.png` | 3D 多视角渲染 | case_00000 的 8 类器官三角网格，6 个视角，每器官独立颜色 |

---

### 3.5 交付物汇总

```
data/processed/stage1b_semantic_model/
├── structure_priors/
│   ├── intensity_stats.json       # 110 类结构的 HU 均值/std/θ_k
│   ├── shape_stats.json           # 110 类结构的体积/球形度
│   └── topology_templates.json   # 44 条解剖邻接边（文献先验）
├── semantic_surface_graph/
│   ├── case_00000_semantic_graph.json   # 含节点语义属性 + 邻接边
│   └── case_00001_semantic_graph.json
├── figures/
│   ├── intensity_stats_heatmap.png
│   ├── topology_graph.png
│   ├── shape_stats.png
│   └── sample_case_3d_organs.png
└── summary.md
```

**`intensity_stats.json` 字段说明**：

| 字段 | 类型 | 含义 |
|------|------|------|
| `mean_hu` | float | HU 均值 $\mu_k$ |
| `std_hu` | float | HU 标准差 $\sigma_k$ |
| `theta_k` | float | 强度容差 $\theta_k = 2\sigma_k$ |
| `n_voxels_total` | int | 全部采样病例的累计体素数 |
| `n_cases` | int | 该类别出现的病例数 |
| `organ_system` | str | 所属系统（parenchymal/vascular/等） |
| `hu_source` | str | `"literature"` 或 `"measured"` |

---

## 4. Stage-1C：LLM 语义渲染先验生成

### 4.1 输入与输出

**输入**：`stage1b_semantic_model/structure_priors/intensity_stats.json`

从中提取每类器官的以下字段，构建 LLM 的输入载荷：

```json
{
  "label":       "liver",
  "count":       83739,       // n_voxels_total（体素总数，反映覆盖率）
  "mean_hu":     60.0,        // CT 平均亮度
  "std_hu":      25.0,        // CT 灰度异质性
  "organ_system":"parenchymal"
}
```

**输出**：`priors.json`，为 60 类器官各自生成三个渲染参数：

| 参数 | 范围 | 含义 |
|------|------|------|
| `sat` | [0.70, 1.40] | 颜色饱和度增益，控制渲染颜色的鲜艳程度 |
| `val` | [0.70, 1.30] | 亮度增益，控制器官表面的整体明暗 |
| `smooth` | [0.00, 0.98] | 类内平滑权重，控制同类器官内相邻点的颜色混合强度 |

---

### 4.2 LLM 调用机制

#### 4.2.1 架构设计

```
intensity_stats.json
       ↓
load_classes_from_intensity_stats()   # 读取 ≤ max_labels 类
       ↓
build_user_payload()                  # 序列化为 JSON 字符串
       ↓
call_llm()  →  HTTP POST /v1/chat/completions
       ↓
parse_llm_response()                  # 解析 + 验证 + 填充缺失
       ↓
priors.json
```

#### 4.2.2 LLM 服务端

使用本地 **vLLM** 部署，兼容 OpenAI Chat Completions API：

```
POST http://localhost:8080/v1/chat/completions
```

请求载荷关键字段：

| 字段 | 值 | 含义 |
|------|---|------|
| `model` | `NousResearch/Llama-2-7b-chat-hf` | 本地模型标识符 |
| `max_tokens` | 1200 | 最大输出 token 数（约 60 类 × 约 18 token/类） |
| `temperature` | 0.15 | 低温度，减少随机性，使输出偏向确定性 |
| `top_p` | 0.9 | Nucleus sampling 阈值 |

#### 4.2.3 System Prompt 设计

Prompt 以枚举规则的形式约束 LLM 的输出空间，共 10 条规则（节选关键规则）：

| 规则编号 | 条件 | sat | val | smooth |
|----------|------|-----|-----|--------|
| 4 | 实质器官（HU ≈ 40~60） | ~1.0~1.1 | ~1.0~1.1 | — |
| 5 | 血管（HU ≈ 50，std 低） | ~1.1 | — | 0.85~0.95 |
| 6 | 骨骼（HU > 400） | — | 1.1~1.2 | 0.3~0.5 |
| 7 | 含气结构（HU < -500） | 0.75~0.90 | — | — |
| 8 | 稀疏类别（count < 100,000） | → 向 1.0 收缩 | → 向 1.0 收缩 | → 向 0.7 收缩 |
| 9 | std_hu > 100（高异质性） | — | — | 0.3~0.5 |

**输出格式约束**（Prompt 中明确要求 LLM 只输出纯 JSON）：

```json
{
  "priors": [
    {"label": "liver", "sat": 1.05, "val": 1.00, "smooth": 0.80},
    ...
  ],
  "global_default": {"sat": 1.0, "val": 1.0, "smooth": 0.7}
}
```

#### 4.2.4 响应解析与鲁棒性处理

`parse_llm_response()` 按以下顺序处理 LLM 原始文本：

1. 剥除 Markdown 代码栏（` ```json ... ``` `）
2. 尝试直接 `json.loads()`
3. 失败则用正则表达式提取第一个 `{...}` 块后重试
4. 对每个先验条目调用 `validate_prior()`，将数值截断到合法范围（见 4.4 节）
5. 对 LLM 遗漏的类别，用 `global_default` 补充

**重试策略**：最多重试 2 次（`max_retries=2`），间隔 2 秒；若 LLM 服务不可达，立即回退到规则模式。

---

### 4.3 规则回退（Mock 模式）

当 LLM 服务不可用时（`--mock` 参数或网络连接失败），使用 `rule_based_priors()` 函数生成确定性参数。

#### 4.3.1 规则优先级（按顺序匹配）

```
IF mean_hu > 300          → 骨骼密质  (sat=1.05, val=1.15, smooth=0.40)
ELIF mean_hu < -400       → 肺/气腔   (sat=0.75, val=0.88, smooth=0.60)
ELIF std_hu > 100         → 高异质组织 (sat=1.00, val=0.98, smooth=0.40)
ELIF organ_system == "vascular" OR 标签在血管列表中
                          → 血管      (sat=1.12, val=1.05, smooth=0.90)
ELIF organ_system == "parenchymal"
                          → 实质器官  (sat=1.05, val=1.00, smooth=0.80)
ELIF organ_system == "cardiac"
                          → 心脏      (sat=1.08, val=1.02, smooth=0.75)
ELIF organ_system == "gi_tract"
                          → 消化道    (sat=1.02, val=0.97, smooth=0.72)
ELIF organ_system == "lung"
                          → 肺（非含气）(sat=0.78, val=0.90, smooth=0.62)
ELIF organ_system == "skeletal"
                          → 骨骼      (sat=0.95, val=1.12, smooth=0.42)
ELSE                      → 其他软组织 (sat=0.98, val=1.02, smooth=0.65)
```

#### 4.3.2 稀疏类别收缩（Shrinkage）

对于体素总数 $< 100{,}000$ 的稀疏类别（在数据集中出现很少、统计不可靠），以 $\alpha = 0.5$ 做线性收缩，将参数向默认值拉近：

$$\text{sat}' = \alpha \cdot 1.0 + (1-\alpha) \cdot \text{sat}$$
$$\text{val}' = \alpha \cdot 1.0 + (1-\alpha) \cdot \text{val}$$
$$\text{smooth}' = \alpha \cdot 0.7 + (1-\alpha) \cdot \text{smooth}$$

- $\alpha = 0.5$：收缩强度，稀疏类别最终参数是规则值与默认值的等权均值
- 默认值 $(\text{sat}_0, \text{val}_0, \text{smooth}_0) = (1.0, 1.0, 0.7)$：无任何颜色增强的"中性"渲染

---

### 4.4 参数约束与 Clamp 处理

所有参数（无论来自 LLM 还是规则）都经过范围截断，防止数值越界导致渲染异常：

$$\text{sat} = \text{clamp}(\text{sat},\ 0.70,\ 1.40)$$
$$\text{val} = \text{clamp}(\text{val},\ 0.70,\ 1.30)$$
$$\text{smooth} = \text{clamp}(\text{smooth},\ 0.00,\ 0.98)$$

其中：

$$\text{clamp}(x, lo, hi) = \max(lo, \min(hi, x))$$

- `smooth` 上界设为 0.98 而非 1.0，是为了保留最小的类内差异，避免整个器官被渲染成纯色平面

---

### 4.5 交付物汇总

```
data/processed/stage1c_llm_priors/
├── priors.json                  # 60 类器官的 sat/val/smooth 参数
├── figures/
│   ├── llm_priors_bar.png       # 热图：organ × {sat, val, smooth}，按系统分组
│   ├── llm_priors_scatter.png   # 散点图：sat vs smooth（左），HU vs val（右）
│   └── llm_vs_default.png       # 棒糖偏差图：各类别与默认值的偏差量
└── summary.md
```

**`priors.json` 结构**：

```json
{
  "priors": [
    {"label": "aorta",  "sat": 1.12, "val": 1.05, "smooth": 0.90},
    {"label": "liver",  "sat": 1.05, "val": 1.00, "smooth": 0.80},
    ...
  ],
  "global_default": {"sat": 1.0, "val": 1.0, "smooth": 0.7},
  "source": "rule_based_mock",
  "n_classes": 60
}
```

---

## 5. 两个 Stage 的接口关系

```
                  KiTS23 CT 影像 (489 cases)
                         ↓
          TotalSegmentator 伪标签 (117 labels/case)
                         ↓
                   ┌─────────────┐
                   │  Stage-1B   │
                   │  Phase A    │ → intensity_stats.json
                   │  Phase B    │ → topology_templates.json
                   │  Phase C    │ → case_XXXXX_semantic_graph.json
                   └──────┬──────┘
                          │ intensity_stats.json
                          ↓
                   ┌─────────────┐
                   │  Stage-1C   │
                   │ (LLM/规则)  │ → priors.json
                   └──────┬──────┘
                          │
         ┌────────────────┼──────────────────┐
         ↓                ↓                  ↓
    centroid_mm       theta_k          sat/val/smooth
    + topology     (BIM快速路径       (三维可视化
    (Stage-3c       边界吸附判断)       渲染参数)
    在线追踪)
```

### 下游使用方式

| 下游模块 | 使用的字段 | 作用 |
|----------|----------|------|
| Stage-3c BIM 快速路径 | `centroid_mm`, `theta_k` | 器官位置锚点 + HU 边界吸附阈值 |
| Stage-3c 异常检测（未实现） | `topology_templates` | 器官位置不符合邻接关系时触发重分割 |
| 3D 可视化渲染 | `sat`, `val`, `smooth` | 控制器官颜色鲜艳度、亮度和表面平滑度 |
| 形变约束（未实现） | `mean_volume_mm3`, `std_volume_mm3` | 体积变化超出先验范围时报警 |

---

## 6. 当前局限与后续计划

| 局限 | 原因 | 改进方向 |
|------|------|---------|
| HU 统计均为文献先验，非 CT 实测 | 快速模式跳过 CT 加载（~4s/case） | 在充足时间预算下运行 `--use_ct`，基于全部 489 例实测 |
| 球形度对大器官（肝、肠、结肠）为 0 | Marching Cubes 在 >50,000 体素时默认跳过 | 用降采样后计算，或补充文献球形度值 |
| Stage-1C 当前为规则模式 | LLM 服务未启动 | 启动 `vllm_server.py` 后去掉 `--mock` 参数 |
| 拓扑约束未接入 Stage-3c | 设计时间优先 | 在 Stage-3c `fast_bim_propagate()` 中增加邻接一致性检验 |
| 语义粒度为整个器官 | 当前架构所限 | 细化到器官亚区域（肝左/右叶，肾皮/髓质） |
| Stage-1B 仅对 2 个 case 构建语义图 | `--n_graph 2` 参数 | 批量推广到全部 489 例，建立完整图谱库 |

---

## 7. 运行方式

### 7.1 运行 Stage-1B

```bash
# 快速模式（文献先验 HU，无需 CT，约 2~5 分钟）
conda run -n MDPC python scripts/stage1b_build_semantic_graph.py \
  --totalseg_dir data/processed/totalseg_labels \
  --ct_dir       data/raw/KiTS23/kits23/dataset \
  --out_dir      data/processed/stage1b_semantic_model \
  --n_sample 20  \    # 用于形状统计的病例数
  --n_topo   0   \    # 0 = 文献拓扑，>0 = 数据驱动拓扑
  --n_graph  2        # 构建完整语义图的病例数

# 精确 HU 模式（需加载 CT，约 30~60 分钟）
conda run -n MDPC python scripts/stage1b_build_semantic_graph.py \
  --use_ct --n_sample 50 --n_topo 10 --n_graph 10

# 断点续跑（跳过已完成的 Phase A/B）
conda run -n MDPC python scripts/stage1b_build_semantic_graph.py \
  --resume --n_graph 50
```

### 7.2 运行 Stage-1C

```bash
# Mock 模式（无需 LLM 服务，立即完成）
conda run -n MDPC python scripts/stage1c_llm_semantic_priors.py \
  --stats_json data/processed/stage1b_semantic_model/structure_priors/intensity_stats.json \
  --out_dir    data/processed/stage1c_llm_priors \
  --mock

# LLM 模式（先启动 vLLM 服务）
conda run -n llm python /home/xqin5/llm/vllm_server.py \
  --model NousResearch/Llama-2-7b-chat-hf --port 8080 &

conda run -n MDPC python scripts/stage1c_llm_semantic_priors.py \
  --stats_json data/processed/stage1b_semantic_model/structure_priors/intensity_stats.json \
  --out_dir    data/processed/stage1c_llm_priors \
  --endpoint   http://localhost:8080 \
  --model      NousResearch/Llama-2-7b-chat-hf \
  --max_labels 60
```

### 7.3 依赖环境

```
Conda env: MDPC
  - Python 3.10+
  - nibabel       # NIfTI 文件读取
  - numpy         # 数值计算
  - scikit-image  # marching_cubes, binary_dilation
  - matplotlib    # 可视化
  - networkx      # 图布局（可选）
  - requests      # LLM HTTP 调用（可选）

Conda env: llm（Stage-1C LLM 模式）
  - vllm          # 本地 LLM 推理服务
  - fastapi       # HTTP 接口
```

---

*报告生成时间：2026-03-08*
*覆盖脚本版本：`stage1b_build_semantic_graph.py` / `stage1c_llm_semantic_priors.py`（MDBIMDT 项目，Stage-4 架构）*
