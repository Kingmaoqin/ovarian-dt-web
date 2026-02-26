# MDBIMDT 全阶段中文技术解说

> **BIM → 医学数字孪生：结构感知增量映射**
> 本文档结合现有实现代码，逐阶段解说数学原理、输入输出、模型架构与当前完成状态。

---

## 目录

1. [系统总览：三路径架构](#0-系统总览三路径架构)
2. [Stage-0：BIM 基线复现](#1-stage-0bim-基线复现)
3. [Stage-1：医学结构模型构建](#2-stage-1医学结构模型构建)
4. [Stage-1B：语义分割训练（Medical BIM 文件生成）](#3-stage-1b语义分割训练medical-bim-文件生成)
5. [BIM 算子迁移：O_index / O_assoc / O_proj](#4-bim-算子迁移o_index--o_assoc--o_proj)
6. [Stage-2：观测流与增量事件定义](#5-stage-2观测流与增量事件定义)
7. [Stage-3A：离线 Otsu 基线分割](#6-stage-3a离线-otsu-基线分割)
8. [Stage-3B：3D UNet 深度学习分割](#7-stage-3b3d-unet-深度学习分割)
9. [Stage-3C：术中实时三路径追踪](#8-stage-3c术中实时三路径追踪)
10. [Stage-4：实时手术导航与展示](#9-stage-4实时手术导航与展示)
11. [符号表](#10-符号表)

---

## 0. 系统总览：三路径架构

### 核心比喻

本项目的核心思想来自 BIM（建筑信息模型）领域：
**"建筑师事先给出每面墙的位置 → 施工监测时只更新发生变化的局部"**
迁移到手术场景就是：
**"nnUNet 离线给出每个器官的位置 → 手术中只实时更新位移/形变的部分"**

| 概念 | BIM 域 | 医学域 |
|------|--------|--------|
| 语义先验 | 建筑师在 BIM 文件中定义的墙/柱/门 | nnUNet 从 CT 推理出的器官/病灶/血管 Mask |
| 在线高频更新 | BIM O_index + O_assoc + O_proj 算子（< 50ms/帧） | 同一套算子，替换"投影目标"后直接复用 |
| 结构先验载体 | IFC 文件 | 多类语义 Mask + Morton 空间索引（"Medical BIM 文件"） |
| 关键差异 | 建筑构件是**刚体** | 器官在呼吸/手术下**持续形变**（需要第二条路径） |

### 三路径决策逻辑

手术中每来一帧观测 $I_t$，系统计算两个指标：

$$\text{drift\_mm}_k = \|\mathbf{c}_k(t) - \mathbf{c}_k(t-1)\|_2$$

$$\text{vol\_change}_k = \frac{|V_k(t) - V_k(t-1)|}{V_k(t-1)}$$

其中 $\mathbf{c}_k(t)$ 为第 $k$ 个解剖结构在 $t$ 时刻的质心坐标（mm），$V_k(t)$ 为体积（mm³）。

三路径切换规则：

$$\text{path} = \begin{cases}
\text{路径1（BIM 快速路径）} & \text{if } \text{drift\_mm} < \delta_{\text{fast}} \\
\text{路径2（形变配准恢复）} & \text{if } \delta_{\text{fast}} \leq \text{drift\_mm} < \delta_{\text{deform}} \\
\text{路径3（nnUNet 重分割）} & \text{if } \text{vol\_change} > 0.3 \\
\text{路径2（兜底）} & \text{otherwise}
\end{cases}$$

推荐阈值：$\delta_{\text{fast}} = 3\,\text{mm}$，$\delta_{\text{deform}} = 10\,\text{mm}$。

**触发频率估计**（按呼吸运动统计）：
- 路径1：~60% 的帧（平静呼吸，小运动）
- 路径2：~38% 的帧（呼吸深幅 / 医生操作）
- 路径3：< 2% 的帧（切除术后体积骤变）

---

## 1. Stage-0：BIM 基线复现

### 目标与地位

Stage-0 是整个项目的**系统性对照组**，证明"BIM 算子在建筑场景已验证可用"。
脚本：[`scripts/stage0_bim_baseline.py`](scripts/stage0_bim_baseline.py)

### 四个算子阶段

BIM baseline 按以下顺序依次调用 `scan2bim_stage_runner.py` 的四个 stage：

```
prep  →  index  →  assoc  →  proj
```

每个 stage 可选四种 impl：`cpu` / `morton_ordered` / `gpu_generic` / `gpu_bimaware`

#### Stage `prep`（预处理）

**输入**：点云文件（`.ply` / `.laz`），含 $(x, y, z)$ 坐标和可选颜色 $(r, g, b)$
**输出**：体素化降采样点集 $P = \{p_i\}_{i=1}^N$，表面代理体素集合 $\{\text{proxy}_k\}_{k=1}^K$

降采样方法：以体素尺寸 $\ell$（默认 0.05m）将点云分格，每格取均值：

$$p_i^{\text{voxel}} = \frac{1}{|G_i|}\sum_{p \in G_i} p, \quad G_i = \{p : \lfloor p / \ell \rfloor = \text{key}_i\}$$

#### Stage `index`（Morton 空间索引）

**输入**：体素化点集 $P$，表面代理体素集合 $\{\text{proxy}_k\}$
**输出**：Morton 哈希索引 $H$，每个输入点的候选表面集合 $\mathcal{A}(p_i)$

**核心函数**：`_expand_bits21` + `compute_morton_codes`

**Morton 编码**（Z-曲线三维空间填充）：

对体素坐标 $(v_x, v_y, v_z)$（均为 21 bit 整数，对应最大 $2^{21}$ 格分辨率），展开位后交叉拼接：

$$\text{expand}(v) = \sum_{i=0}^{20} \text{bit}_i(v) \cdot 2^{3i}$$

最终 Morton 编码（63 bit）：

$$\text{morton}(v_x, v_y, v_z) = \text{expand}(v_x) \,|\, (\text{expand}(v_y) \ll 1) \,|\, (\text{expand}(v_z) \ll 2)$$

**意义**：空间相邻的体素在 Morton 编码中的数值也相近，GPU kernel 访问时内存连续，L2 缓存命中率高。

**27-邻域候选压缩**：

对每个点 $p_i$，只考虑其 $3 \times 3 \times 3 = 27$ 个相邻体素格子内的表面代理，忽略全局其他所有表面：

$$\mathcal{A}(p_i) = \{p_i + \Delta : \Delta \in \{-1, 0, 1\}^3\} \cap \{\text{proxy}_k\}$$

**关键指标**：`selected_ratio`，即实际参与 assoc 计算的点比例（基线约 8.4%），说明 92% 的全局搜索被省略。

#### Stage `assoc`（局部关联）

**输入**：每个点 $p_i$ 及其候选表面集合 $\mathcal{A}(p_i)$
**输出**：点到表面的关联映射 $N(S_k) = \{p_i : p_i \in \mathcal{A}(p_i),\, d(p_i, S_k) < \varepsilon\}$

过滤条件：欧氏距离阈值 $\varepsilon$（默认 0.1m）

**内存跳跃分析**（`_jump_stats`）：

对每个表面按关联点数统计"内存跳跃"，即从一个候选点跳到下一候选点的内存地址步长均值。Morton 排序后跳跃小，验证内存访问局部性。

$$\text{jump\_mean} = \frac{1}{|N(S_k)| - 1} \sum_{i=1}^{|N(S_k)|-1} |\text{addr}(p_{i+1}) - \text{addr}(p_i)|$$

#### Stage `proj`（纹理投影）

**输入**：每个表面 $S_k$ 的关联点集 $N(S_k)$，每点的颜色 $(r_i, g_i, b_i)$
**输出**：每个表面的加权平均纹理 $T(S_k)$

$$T(S_k) = \frac{\sum_{p_i \in N(S_k)} w_i \cdot \mathbf{c}_i}{\sum_{p_i \in N(S_k)} w_i}, \quad w_i = \exp\!\left(-\frac{d(p_i, S_k)^2}{\sigma^2}\right)$$

其中 $\mathbf{c}_i = (r_i, g_i, b_i)$ 为点的颜色，$\sigma$ 为距离衰减带宽（默认 0.05m）。

**kernel fusion 内存分析**：

GPU 单次 fused kernel 的显存需求：

$$M_{\text{fused}} = M_P + M_T$$

两次独立 kernel 的需求：

$$M_{\text{sep}} = M_P + 2 \cdot M_{\text{assoc}} + M_T$$

其中 $M_P$=点云数据，$M_T$=纹理缓冲，$M_{\text{assoc}}$=关联索引中间结果。Fusion 可节省约 $2 M_{\text{assoc}}$ 显存（对大规模场景约省 30-50%）。

### 输入输出汇总

```
输入：
  --dataset-zip   .zip 内含点云（.ply/.laz）
  --surface-count 表面代理数量 K（默认 50）
  --voxel-size    体素尺寸 ℓ（默认 0.05m）
  --max-points    最大点数 N

输出：runs/stage0/
  ├── metrics.json         ← selected_ratio, avg_candidates, runtime_sec
  ├── summary.md           ← 各 impl 对比表
  └── summary.csv
```

### 当前状态

✅ **已完成**，可复现 4 种 impl（cpu/morton_ordered/gpu_generic/gpu_bimaware）对比结果。

---

## 2. Stage-1：医学结构模型构建

### 目标

把 BIM 中"建筑师给定的表面网格"类比操作在医学 CT 上实现：
从**分割标签 Mask** → **器官表面网格** → **patch 拓扑图** → **局部索引**。

脚本：[`scripts/stage1_build_surfaces.py`](scripts/stage1_build_surfaces.py)

### 数据

当前使用 MSD Task09_Spleen（41 例，单器官）：
- `imagesTr/spleen_{id}.nii.gz`：CT 体数据，形状 $(D \times H \times W)$，单位：体素
- `labelsTr/spleen_{id}.nii.gz`：二值分割标签，1=脾脏，0=背景

### 核心数学：Marching Cubes 表面重建

**输入**：二值 Mask $M \in \{0,1\}^{D \times H \times W}$，体素间距 $\Delta = (\Delta_d, \Delta_h, \Delta_w)$（mm）

**目标**：提取 $M$ 的等值面（$M=1$ 区域的外表面）

Marching Cubes 将 Mask 视为三维标量场，在每个 $2 \times 2 \times 2$ 体素块（"Marching Cube"）内判断：

$$\text{sign\_code} = \sum_{i=0}^{7} \mathbf{1}[M(\mathbf{v}_i) > 0.5] \cdot 2^i \in \{0, 1, \ldots, 255\}$$

通过查表得到局部三角面片顶点插值位置，连接后得到全局表面网格：

$$\mathcal{M} = (\mathbf{V}, \mathbf{F}), \quad \mathbf{V} \in \mathbb{R}^{N_v \times 3}\ (\text{体素坐标}),\ \mathbf{F} \in \mathbb{Z}^{N_f \times 3}\ (\text{三角面片})$$

**体素 → 世界坐标转换**（NIfTI 仿射矩阵 $\mathbf{A} \in \mathbb{R}^{4 \times 4}$）：

$$\mathbf{v}_{\text{world}} = \mathbf{A} \cdot \begin{bmatrix} v_x \\ v_y \\ v_z \\ 1 \end{bmatrix}, \quad \mathbf{v}_{\text{world}} \in \mathbb{R}^3\ (\text{mm})$$

代码实现（[`stage1_build_surfaces.py:17`](scripts/stage1_build_surfaces.py#L17)）：
```python
verts_voxel, faces, _, _ = measure.marching_cubes(mask_array, level=0.5)
affine = nib.load(label_path).affine
verts_world = (affine[:3, :3] @ verts_voxel.T + affine[:3, 3:]).T  # 体素→mm
```

### Patch 拓扑图构建

**目的**：把连续表面网格离散化为图结构，便于局部索引和并行更新。

**方法**：以 $l_{\text{patch}} = 20\,\text{mm}$ 为格子尺寸，将世界坐标顶点分组：

$$\text{key}(v) = \left\lfloor \frac{\mathbf{v}_{\text{world}} - \mathbf{v}_{\min}}{l_{\text{patch}}} \right\rfloor \in \mathbb{Z}^3$$

同一 key 的顶点归为一个**patch 节点**，patch 间若共享三角面片则连边：

$$E = \{(u, v) : \exists \text{三角面片} (a, b, c),\ \text{key}(a) = \text{key}_u,\ \text{key}(b) = \text{key}_v,\ u \neq v\}$$

每个节点记录（[`stage1_build_surfaces.py:51`](scripts/stage1_build_surfaces.py#L51)）：
- `patch_key`：三维格子坐标 $(k_x, k_y, k_z)$
- `centroid_mm`：该 patch 内所有顶点的均值坐标
- `bbox_min_mm / bbox_max_mm`：包围盒（用于 O_index 候选筛选）

### 输入输出汇总

```
输入：
  Task09_Spleen/
    imagesTr/*.nii.gz   CT 体数据（体素）
    labelsTr/*.nii.gz   二值 Mask（体素）

输出：data/processed/stage1/
  ├── meshes/{case_id}_spleen.ply        ← 表面网格（世界坐标，mm）
  ├── surface_graph.json                 ← 节点列表 + 边列表
  └── cases_summary.json                 ← 每例统计（patch 数、边数、度均值）
```

典型规模（Task09_Spleen，41 例均值）：
- 节点数：306 个 patch
- 边数：902 条邻接边
- 每 patch 覆盖 ~50-200 个网格顶点

### 当前状态

✅ **已完成**（单器官脾脏），产物可复现。
**局限**：仅处理单器官 + 依赖手工标注 → 需 Stage-1B 扩展到多器官自动分割。

---

## 3. Stage-1B：语义分割训练（Medical BIM 文件生成）

> **这是当前最关键的待完成环节。**
> BIM 文件里的语义（墙/柱/门）是建筑师直接定义的；医学语义（器官/病灶/血管）必须由模型学出来。
> 没有这一步，就没有"Medical BIM 文件"，三路径架构无法启动。

### 模型架构：3D nnUNet（自适应 U-Net）

**网络输入**：

$$I \in \mathbb{R}^{1 \times D \times H \times W}$$

其中 $D, H, W$ 为深度/高度/宽度体素数（nnUNet 自动配置 patch size，典型值 $128 \times 128 \times 128$）。

**编码器**（5 层下采样）：

每层由两个卷积块组成：

$$\mathbf{F}_l = \text{LeakyReLU}(\text{BN}(\text{Conv3D}_{3^3}(\mathbf{F}_{l-1})))$$

$$\mathbf{F}_{l}^{\text{down}} = \text{AvgPool}_{2^3}(\mathbf{F}_l) \quad \text{或} \quad \text{Conv3D}_{\text{stride=2}}$$

- 输入通道：$C_0 = 1$
- 逐层通道数：$32 \to 64 \to 128 \to 256 \to 320$（nnUNet 默认上限 320）
- 最深特征图：$\mathbf{F}_5 \in \mathbb{R}^{320 \times d \times h \times w}$，$d = D/32$

**解码器**（5 层上采样 + 跳跃连接）：

$$\mathbf{G}_l = \text{Conv}(\text{Cat}[\text{Upsample}(\mathbf{G}_{l+1}),\ \mathbf{F}_l])$$

其中 $\text{Cat}$ 表示通道维度拼接（跳跃连接）。

**输出头**：

$$\mathbf{P} = \text{Softmax}\!\left(\text{Conv3D}_{1^3}(\mathbf{G}_1)\right) \in \mathbb{R}^{(K+1) \times D \times H \times W}$$

其中 $K+1$ 个通道对应 $K$ 个解剖结构类 + 1 个背景类。预测 Mask：

$$\hat{M}_k(v) = \mathbf{1}\!\left[k = \arg\max_{k'} P_{k'}(v)\right]$$

### 损失函数

$$\mathcal{L} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{CE}}$$

**Dice Loss**（处理类别不平衡，小器官不被大背景淹没）：

$$\mathcal{L}_{\text{Dice}} = -\frac{1}{K}\sum_{k=1}^{K} \underbrace{\frac{2\sum_{v} p_k(v) \cdot g_k(v)}{\sum_{v} p_k(v) + \sum_{v} g_k(v)}}_{\text{Dice}(k)}$$

其中：
- $p_k(v) = P_k(v)$：体素 $v$ 属于类别 $k$ 的预测概率（Softmax 输出）
- $g_k(v) \in \{0, 1\}$：体素 $v$ 是否属于类别 $k$ 的真值标签
- 分子 $2\sum_v p_k g_k$：两倍交集体积
- 分母 $\sum_v p_k + \sum_v g_k$：预测 + 真值体积之和

**Cross-Entropy Loss**（提供每体素梯度信号）：

$$\mathcal{L}_{\text{CE}} = -\frac{1}{K} \sum_{k=1}^{K} \sum_{v} g_k(v) \log p_k(v)$$

**为什么联合使用**：Dice Loss 对小器官更敏感但梯度稀疏；CE Loss 梯度稳定但受样本不平衡影响。两者互补，实践中比单独使用任意一个收敛更快、精度更高。

### 滑动窗口推理（Sliding Window Inference）

训练时用随机裁剪 patch（如 $128^3$），推理时输入整个 CT，通过滑动窗口：

$$\hat{P}(v) = \frac{\sum_{\text{window }w \ni v} G_w(v) \cdot \hat{P}_w(v)}{\sum_{\text{window }w \ni v} G_w(v)}$$

其中 $G_w(v) = \exp\!\left(-\frac{\|v - c_w\|^2}{2\sigma_w^2}\right)$ 为高斯权重（中央权重高，边缘权重低），用于平滑重叠区域的预测。

**参数**：窗口大小 $128^3$，步长 64（50% 重叠），$\sigma_w = 0.125 \times 128 = 16$ 体素。

### 结构先验统计（O_assoc 参数来源）

训练完成后，在全部训练样本上统计每类结构的强度分布：

$$\mu_k = \frac{1}{|\mathcal{V}_k|} \sum_{(v, s) \in \mathcal{D}} \sum_{v' \in M_k^{(s)}} I^{(s)}(v')$$

$$\sigma_k^2 = \frac{1}{|\mathcal{V}_k|} \sum_{(v, s) \in \mathcal{D}} \sum_{v' \in M_k^{(s)}} \left(I^{(s)}(v') - \mu_k\right)^2$$

其中 $\mathcal{D}$ 为训练集所有样本，$I^{(s)}$ 为第 $s$ 个样本的 CT 强度（HU），$M_k^{(s)}$ 为对应的类别 $k$ Mask。

这两个统计量 $(\mu_k, \sigma_k)$ 保存为 `intensity_stats.json`，后续 O_assoc 的强度门控阈值直接取 $\theta_k = 2\sigma_k$。

### 构建多类拓扑图

从预测 Mask 判断结构间的解剖邻接关系，通过膨胀腐蚀检测接触：

```python
# 对结构 k1 的 Mask 膨胀 d 个体素，若与 k2 重叠则添加邻接边
dilated_k1 = binary_dilation(M_k1, iterations=d)
contact = np.count_nonzero(dilated_k1 & M_k2)
if contact > min_contact_voxels:
    graph.add_edge(k1, k2, contact_area=contact)
```

层级结构（三层，类比 BIM IFC 层级）：

```
Level 0：全身容器（胸腔、腹腔）
Level 1：器官（肝、脾、肾、胰腺…）
Level 2：病灶（肝肿瘤、肾肿瘤…）← 包含于 Level 1
Level 3：血管（门静脉、肾动脉…）← 与 Level 1 邻接
```

### 训练数据与预计时间

| 数据集 | 类别数 $K$ | 用途 |
|--------|-----------|------|
| KiTS23（489例 CT，已下载） | 3（肾/肿瘤/囊肿） | 主训练集（病灶+器官） |
| TotalSegmentator 推理结果（正在生成） | 117 类 | 作为多器官伪标签（无需额外标注） |
| BTCV（下载中，~8GB） | 13 | 验证集 |

当前状态：**TotalSegmentator 正在跑 KiTS23 489例**（4 GPU 并行，预计 3-4 小时完成）。
完成后即可以 TotalSegmentator 输出作为 Stage-1B 多类伪标签，不需要单独训练初始化即可直接使用。

### 输入输出汇总

```
输入：
  CT 体数据 I ∈ R^{D×H×W}（HU 强度）

输出：data/processed/stage1b_semantic_model/
  ├── checkpoints/best_model.pth           ← nnUNet 最优权重
  ├── predictions/{case_id}_semantics.nii.gz ← K 类语义 Mask
  ├── structure_priors/
  │   ├── intensity_stats.json             ← {k: {mu, sigma}} 强度先验
  │   ├── shape_stats.json                 ← {k: {vol_mean, vol_std, sphericity}}
  │   └── topology_templates.json         ← 结构间邻接图模板
  └── semantic_surface_graph/
      └── {case_id}_semantic_graph.json    ← 多层 SurfaceGraph
```

### 验收标准

| 指标 | 目标值 |
|------|--------|
| 器官类 Dice | ≥ 0.85 |
| 病灶类 Dice | ≥ 0.70 |
| 推理时延（单例滑窗） | ≤ 30s（离线可接受） |
| 强度先验覆盖 | 所有 $K$ 类均有 $(\mu_k, \sigma_k)$ |

---

## 4. BIM 算子迁移：O_index / O_assoc / O_proj

> Stage-1B 完成后，医学域的"BIM 文件"就绪，三个核心算子可从建筑域直接迁移。
> 以下说明各算子的医学化改造（改动量极小，共约 16 行代码变化）。

### 4.1 $\mathcal{O}_{\text{index}}$：从 27-邻域到语义 ROI

**BIM 原版**：所有点统一用 27-邻域候选压缩，selected_ratio ≈ 8.4%

**医学版**：对每个结构 $S_k$ 单独计算**自适应 ROI 半径** $m_k$：

$$\text{ROI}_k = \{v \in I_t : d(v, M_k) \leq m_k\}$$

$$m_k = \text{clip}\!\left(\frac{\text{drift\_mm}_k}{2} + 4,\ 4\,\text{mm},\ 18\,\text{mm}\right)$$

**物理含义**：
- $m_k$ 的下限 4mm：确保即便结构静止也有足够候选范围
- $m_k$ 的上限 18mm：防止 ROI 过大导致性能退化
- $\text{drift\_mm}_k / 2$：将上一帧漂移量纳入 ROI 动态扩展，大运动时自动放大

**代码改动**：约 5 行（原 27-邻域循环参数改为按结构分组调用）。

**Morton 编码**：直接复用 `_expand_bits21` + `compute_morton_codes`（零改动），对医学体素同样适用——这两个函数只关心体素整数坐标，与数据域无关。

### 4.2 $\mathcal{O}_{\text{assoc}}$：从点面距离到体素-结构关联

**BIM 原版**：仅用几何距离过滤 $d(p_i, S_k) < \varepsilon$

**医学版**：在几何过滤后追加**强度门控**（新增 3 行代码）：

$$N_{\text{med}}(S_k, t) = \{v \in \text{ROI}_k :\ \underbrace{d(v,\ \text{mesh}_k) < \varepsilon}_{\text{几何过滤（BIM原版）}}\ \wedge\ \underbrace{|I_t(v) - \mu_k| < 2\sigma_k}_{\text{强度门控（新增）}}\}$$

其中：
- $d(v, \text{mesh}_k)$：体素 $v$ 到解剖表面网格 $\text{mesh}_k$ 的最短距离（mm）
- $I_t(v)$：体素 $v$ 在 $t$ 时刻的 CT 强度（HU）
- $\mu_k, \sigma_k$：结构 $k$ 的强度先验均值和标准差（来自 `intensity_stats.json`）
- $2\sigma_k$：$\pm 2$ 标准差覆盖约 95% 的正常强度范围

**物理含义**：肝脏的典型强度范围是 50-70 HU，血液是 30-45 HU。强度门控可以排除 CT 伪影和邻接组织的误关联，从而提高追踪精度。

**跨模态情形**（术中超声 vs 术前 CT）：将欧氏距离替换为互信息（MI）：

$$\mathcal{L}_{\text{sim}}^{\text{CT-US}} = -\text{MI}(I_t^{\text{US}}, I_{t_0}^{\text{CT}}) = -\sum_{a,b} p(a,b) \log\frac{p(a,b)}{p(a)p(b)}$$

### 4.3 $\mathcal{O}_{\text{proj}}$：从颜色投影到状态贝叶斯融合

**BIM 原版**：将 RGB 颜色做距离加权均值 $T(S_k) = \frac{\sum w_i c_i}{\sum w_i}$

**医学版**：将投影目标替换为**解剖状态向量**（约 8 行代码改动）：

$$\mathbf{s}_k = \begin{bmatrix} c_k^x \\ c_k^y \\ c_k^z \\ V_k \\ \mu_k^{\text{obs}} \\ \sigma_k^{\text{obs}} \\ \text{conf}_k \end{bmatrix}$$

其中：$c_k^{x,y,z}$ 为质心坐标（mm），$V_k$ 为体积（mm³），$\mu_k^{\text{obs}}, \sigma_k^{\text{obs}}$ 为当前观测强度统计，$\text{conf}_k \in [0,1]$ 为追踪置信度。

**时序更新（指数平滑贝叶斯融合）**：

$$\mathbf{s}_k(t) = (1 - \alpha_k) \cdot \mathbf{s}_k(t-1) + \alpha_k \cdot \hat{\mathbf{s}}_k(t)$$

融合权重 $\alpha_k$（自适应，基于事件严重度）：

$$\alpha_k = 1 - \exp(-\text{event\_score}_k), \quad \text{event\_score}_k \in [0, 1]$$

- $\text{event\_score}_k \approx 0$（小变化）→ $\alpha_k \approx 0$：保持历史状态（惯性强）
- $\text{event\_score}_k \approx 1$（大变化）→ $\alpha_k \approx 0.63$：快速跟随新观测

**权重公式不变**（直接复用 BIM proj 公式）：

$$w_i = \exp\!\left(-\frac{d(v_i, \text{mesh}_k)^2}{\sigma^2}\right), \quad \hat{\mathbf{s}}_k(t) = \frac{\sum_{v_i \in N_{\text{med}}} w_i \cdot \phi(v_i)}{\sum_{v_i \in N_{\text{med}}} w_i}$$

其中 $\phi(v_i) = (x_i, y_i, z_i, I_t(v_i), ...)$ 为体素特征向量。

### 4.4 代码复用量汇总

| 函数/模块 | 迁移方式 | 代码改动 |
|-----------|----------|---------|
| `_expand_bits21` | 零改动直接复用 | 0 行 |
| `compute_morton_codes` | 零改动 | 0 行 |
| `pack_voxel_keys` | 零改动 | 0 行 |
| `apply_morton_sort` | 零改动 | 0 行 |
| `_jump_stats` | 零改动（验证内存访问） | 0 行 |
| 27-邻域候选压缩 | 参数替换（代理体素来源） | ~5 行 |
| `stage_assoc` 几何过滤 | 追加强度门控 | +3 行 |
| `stage_proj` 加权聚合 | 替换投影目标（RGB→状态向量） | ~8 行 |
| Kernel fusion 分析框架 | 直接搬运 | 0 行（分析） |
| SoA 内存布局 | 直接复用（体素 SoA） | 0 行 |

> `scan2bim_stage_runner.py` 中超过 **60%** 的代码与域无关，可零或微改动迁移。

---

## 5. Stage-2：观测流与增量事件定义

### 目标

在没有真实术中数据的条件下，**合成**一套伪在线两帧观测流 $(t_0, t_1)$，定义三类增量事件，为在线追踪的事件驱动架构建立输入层。

脚本：[`scripts/stage2_build_observation_stream.py`](scripts/stage2_build_observation_stream.py)

### 合成观测流的构造方法

**基准帧 $t_0$（真实 CT）**：

$$\mathbf{O}_{t_0} = \{I_{t_0},\ M_{t_0},\ P_{t_0}^{\text{surface}},\ \text{ROI}_{t_0}\}$$

- $I_{t_0} \in \mathbb{R}^{D \times H \times W}$：原始 CT crop（在 Mask 包围盒 + 20mm margin 区域）
- $M_{t_0} \in \{0,1\}^{D \times H \times W}$：器官/病灶 Mask
- $P_{t_0}^{\text{surface}}$：Marching Cubes 提取的表面点云（用于 O_assoc 距离计算）

**增量帧 $t_1$（合成变化）**：

在 $t_0$ 基础上施加三类合成扰动：

$$I_{t_1} = I_{t_0} + \Delta I^{\text{intensity}} + \text{noise}(\sigma_n)$$

$$M_{t_1} = \text{translate}(M_{t_0},\ \delta_{\text{shift}}) \oplus \Delta M^{\text{morph}}$$

具体参数（从 `stage2_build_observation_stream.py` 的 `CaseArtifacts`）：
- `drift_mm`：表面质心位移距离（合成），范围 2-8mm
- `changed_voxel_ratio`：Mask 变化体素占比
- `mean_intensity_delta`：平均强度变化（HU）
- `hr_delta`：合成心率（生理参数），用于 physiology 事件

### 三类增量事件定义

事件通过阈值比较触发：

| 事件类型 | 触发条件 | 数学表达 |
|----------|----------|---------|
| `geometry_drift` | 表面质心位移过大 | $\text{drift\_mm} > \theta_d$（默认 3mm） |
| `intensity_change` | 强度分布异常 | $|\Delta\mu_{\text{obs}}| > 2\sigma_k$ |
| `physiology_priority_trigger` | 生理参数突变 | $|\Delta \text{HR}| > \theta_{\text{HR}}$（默认 15 bpm） |

**事件评分**（综合三类事件的加权归一化分数）：

$$\text{event\_score}_k = \text{clip}\!\left(\frac{w_g \cdot \text{drift\_mm} + w_i \cdot |\Delta\mu_k| / \sigma_k + w_p \cdot |\Delta\text{HR}|}{\text{scale}},\ 0,\ 1\right)$$

权重默认：$w_g = 0.5,\ w_i = 0.3,\ w_p = 0.2$。

该分数直接作为 $\alpha_k$ 的计算输入（见 O_proj 部分）。

### 输入输出汇总

```
输入：
  Stage-1 输出的 surface_graph.json + meshes/*.ply
  Task09_Spleen imagesTr/*.nii.gz + labelsTr/*.nii.gz

输出：data/processed/stage2/
  ├── observation_stream.jsonl   ← 每例 t0/t1 帧的元数据 + 统计量
  ├── event_log.jsonl            ← 每例触发的事件列表
  ├── cases_metrics.csv          ← drift_mm, changed_ratio, intensity_delta
  └── {case_id}/
      ├── t0_volume.nii.gz       ← t0 CT crop
      ├── t1_volume.nii.gz       ← t1 合成 CT
      ├── t0_points.ply          ← t0 表面点云
      └── t1_points.ply          ← t1 合成表面点云
```

### 当前状态

✅ **已完成**，41 例均生成 t0/t1 两帧，每例可触发 3 类事件。

---

## 6. Stage-3A：离线 Otsu 基线分割

### 目标

用最简单的阈值方法建立精度下界（天花板低，仅作对照，不是最终目标）。

脚本：[`scripts/stage3_incremental_segmentation.py`](scripts/stage3_incremental_segmentation.py)

### Otsu 阈值分割

**原理**：在 CT 强度直方图上找最大化类间方差的阈值 $\tau^*$：

$$\tau^* = \arg\max_\tau \left[\omega_0(\tau)\,\omega_1(\tau)\,(\mu_0(\tau) - \mu_1(\tau))^2\right]$$

其中：
- $\omega_0(\tau) = \sum_{v < \tau} p(v)$：低于阈值的体素概率（背景权重）
- $\omega_1(\tau) = 1 - \omega_0(\tau)$：高于阈值的体素概率（器官权重）
- $\mu_0(\tau), \mu_1(\tau)$：两类强度均值
- $(\mu_0 - \mu_1)^2$：类间均值距离

**双路径实现**：
1. **全局路径**：在整个 crop 上计算 $\tau^*$，直接阈值化
2. **增量路径**：用 $t_0$ Mask 限定 ROI，在 ROI 内重新计算 $\tau^*$，用于 $t_1$ 的变化区域

**已知局限**：Otsu 对 CT 的软组织分割效果差（软组织强度重叠严重），Dice 约 0.40-0.56，这是方法本身的天花板，不代表框架局限。

### 当前状态

✅ **已完成**，Dice 约 0.40-0.56，输出在 `data/processed/stage3_otsu/`。

---

## 7. Stage-3B：3D UNet 深度学习分割

### 目标

用轻量 3D UNet（基于 MONAI 框架）建立深度学习精度基线，目标 Dice ≥ 0.85。

脚本：[`scripts/stage3b_unet_segmentation.py`](scripts/stage3b_unet_segmentation.py)

### 模型架构（MONAI UNet）

**代码定义**（[`stage3b_unet_segmentation.py:43`](scripts/stage3b_unet_segmentation.py#L43)）：

```python
from monai.networks.nets import UNet
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256),    # 编码器通道数
    strides=(2, 2, 2),              # 每层下采样步长
    num_res_units=2,                # 每层残差块数量
)
```

**网络结构**：

```
输入 I ∈ R^{1×D×H×W}
  ↓ Conv3D(1→32) + Residual×2
  ↓ Stride=2 下采样
  ↓ Conv3D(32→64) + Residual×2
  ↓ Stride=2 下采样
  ↓ Conv3D(64→128) + Residual×2
  ↓ Stride=2 下采样
Bottleneck: Conv3D(128→256)
  ↑ Upsample + Skip + Conv3D(256→128)
  ↑ Upsample + Skip + Conv3D(128→64)
  ↑ Upsample + Skip + Conv3D(64→32)
  ↓ Conv3D(32→1) + Sigmoid
输出 P ∈ R^{1×D×H×W}（逐体素肿瘤概率）
```

输出单通道表示前景（器官/病灶）概率，阈值 0.5 转为二值 Mask。

### 训练细节

**数据**：Stage-2 输出的 41 例 crop（$t_0$ + $t_1$ 两帧，合计 82 个样本）

**训练/验证划分**：80%/20% 随机划分

**损失函数**：BCE-Dice 联合损失

$$\mathcal{L}_{\text{3B}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_v\!\left[g(v)\log p(v) + (1-g(v))\log(1-p(v))\right]$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_v p(v) g(v)}{\sum_v p(v) + \sum_v g(v) + \varepsilon}, \quad \varepsilon = 10^{-5}\ \text{（防零除）}$$

**优化器**：Adam，学习率 $3 \times 10^{-4}$，训练 10 epoch

**目标 patch 大小**：自动对齐到 Stage-2 输出尺寸（约 $80 \times 80 \times 80$ 体素）

### 增量推理（O_proj 的近似实现）

**全局推理**（full inference）：直接对整个 $t_1$ crop 运行 UNet 前向：

$$\hat{M}_{t_1}^{\text{full}} = (p_{\theta}(I_{t_1}) > 0.5)$$

**增量推理**（incremental inference）：用 $t_0$ 的预测 Mask 限定 ROI，仅在 ROI 内运行 UNet：

$$\text{ROI} = \{v : d(v, \hat{M}_{t_0}) \leq r_{\text{expand}}\} \cup \hat{M}_{t_0}$$

$$\hat{M}_{t_1}^{\text{incr}} = \hat{M}_{t_0} \odot \mathbf{1}[\text{ROI}]$$

增量回退门控（当 ROI 与体积比异常时降级为全局推理）：
- 条件1：$V(\text{ROI}) / V(\text{crop}) > 0.8$（ROI 过大，索引无效）
- 条件2：$V(\hat{M}_{t_1}^{\text{incr}}) / V(\hat{M}_{t_0}) \notin [0.5, 2.0]$（体积突变，需重推理）

**Dice 指标**（$v$ 遍历所有体素）：

$$\text{Dice} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}} = \frac{2\sum_v \hat{g}(v) g(v)}{\sum_v \hat{g}(v) + \sum_v g(v)}$$

**HD95**（95th 分位数豪斯多夫距离，衡量边界对齐）：

$$\text{HD95} = \max(\text{directed\_HD95}(\hat{M}, M),\ \text{directed\_HD95}(M, \hat{M}))$$

$$\text{directed\_HD95}(A, B) = \text{percentile}_{95}\!\left(\{d(a, B) : a \in \partial A\}\right)$$

其中 $\partial A$ 为 $A$ 的表面体素集合，$d(a, B)$ 为点 $a$ 到集合 $B$ 的最小距离。

### 当前结果（2026-02-26）

| 评测模式 | Dice（均值） | 说明 |
|---------|------------|------|
| t1 全局推理 | 0.747 | 对 3 个测试例（spleen_10/12/13）均值 |
| t1 增量推理 | 0.735 | 增量 ROI 约束，效率提升 |

**待提升**：引入 Stage-1B 多结构先验 + 更长训练 → 目标 Dice 0.85+。

### 输入输出汇总

```
输入：
  data/processed/stage2/{case_id}/t*.nii.gz （crop 体数据）

输出：data/processed/stage3b_unet_mdpc_v2/
  ├── model_best.pth                 ← 最优 checkpoint（按 val Dice）
  ├── train_log.csv                  ← 每 epoch 的 train_loss, val_dice
  ├── metrics_per_case.csv           ← per-case Dice/HD95（full+incr）
  ├── 3d_vis_{case_id}_*.png         ← GT vs Pred 3D 表面对比图
  └── summary.md
```

---

## 8. Stage-3C：术中实时三路径追踪

### 目标

把 Stage-3B 的离线分割能力，通过三路径架构变成**手术中逐帧实时状态维护**。

脚本：[`scripts/stage3c_online_tracking.py`](scripts/stage3c_online_tracking.py)

### 合成时序帧（伪在线流）

**输入帧生成**（[`stage3c_online_tracking.py:69`](scripts/stage3c_online_tracking.py#L69) `blend_frames`函数）：

在 $t_0$ 和 $t_1$ 之间插值生成 $N=10$ 帧，并叠加两种运动：

**1. 呼吸运动**（周期正弦）：

$$\mathbf{b}(t) = A_{\text{breath}} \cdot \begin{bmatrix} \sin(2\pi\alpha) \\ 0.5\sin(2\pi\alpha + 0.8) \\ 0 \end{bmatrix}, \quad \alpha = \frac{t}{N-1} \in [0, 1]$$

其中 $A_{\text{breath}} = 2$ 体素（约 2-4mm），相位偏移 0.8 rad 模拟 x/y 轴的轻微异相。

**2. 手术操作冲击**（单帧阶跃）：

$$\mathbf{s}(t) = \begin{cases} (6, -1.8, 0)\ \text{体素} & t = t_{\text{shock}} = 0.7(N-1) \\ \mathbf{0} & \text{otherwise} \end{cases}$$

总位移（体素坐标，取整）：

$$\text{shift}(t) = \text{round}\!\left(\alpha \cdot (\mathbf{c}_1 - \mathbf{c}_0) + \mathbf{b}(t) + \mathbf{s}(t)\right)$$

第 $t$ 帧图像：

$$I_t = (1 - \alpha) \cdot I_{t_0} + \alpha \cdot I_{t_1}$$

对应 Mask：$M_t = \text{translate}(M_{t_0},\ \text{shift}(t))$（刚性平移近似）。

### 三路径调度实现

**每帧处理流程**：

```python
drift_mm = ‖c(t) - c(t-1)‖₂ × vox_spacing_mm

if drift_mm < δ_fast (3mm):
    # 路径1：BIM 快速路径（增量推理）
    pred_t = infer_incremental(model, I_t, prior_mask=state.mask)

elif drift_mm < δ_deform (10mm):
    # 路径2：形变配准恢复（平移近似）
    shift_vox = round(drift_mm / spacing)
    shifted_mask = shift_mask_no_wrap(state.mask, shift_vox)
    pred_t = infer_incremental(model, I_t, prior_mask=shifted_mask)

else:
    # 路径3：全局重分割
    pred_t = infer_full(model, I_t)
```

**状态更新**（O_proj 的简化实现）：

$$\mathbf{c}(t) = (1 - \alpha) \cdot \mathbf{c}(t-1) + \alpha \cdot \mathbf{c}^{\text{pred}}(t)$$

$$V(t) = |M^{\text{pred}}(t)|_{\text{voxels}} \times \prod_i \Delta_i$$

其中 $\alpha = 0.3$（快速路径），$\alpha = 0.7$（形变路径），$\alpha = 1.0$（重分割路径）。

### 安全裕量计算

在每帧 O_proj 之后附加，基于距离变换（EDT）：

$$d_{\text{vessel}}(t) = \text{EDT}(\sim M_{\text{vessel}})\big|_{\mathbf{c}_{\text{lesion}}(t)}$$

其中：
- $\sim M_{\text{vessel}}$：血管 Mask 的补集（背景）
- $\text{EDT}(\cdot)$：欧氏距离变换（每个背景体素到最近血管边界的距离，mm）
- $|_{\mathbf{c}_{\text{lesion}}(t)}$：在病灶质心处查表，O(1) 查询

安全裕量分级：

$$\text{alert}(t) = \begin{cases} \texttt{SAFE} & d_{\text{vessel}}(t) > 5\,\text{mm} \\ \texttt{WARNING} & 2 < d_{\text{vessel}}(t) \leq 5\,\text{mm} \\ \texttt{CRITICAL} & d_{\text{vessel}}(t) \leq 2\,\text{mm} \end{cases}$$

### 评价指标

**TRE（目标配准误差，Target Registration Error）**：

$$\text{TRE}(t) = \|\mathbf{c}^{\text{pred}}(t) - \mathbf{c}^{\text{GT}}(t)\|_2\ (\text{mm})$$

**P95 延迟**：所有帧处理时间的第 95 百分位数（ms），要求 < 100ms/帧。

**ID Switch 率**：追踪目标发生跳变的帧数占比，要求 < 5%。

### 当前结果（2026-02-26）

已完成首版：
- 3 个基准病例 × 10 帧伪在线追踪
- 输出：路径占比（fast/deform/resegment）、P95 延迟、终帧 Dice
- 已生成 3D 可视化：病灶质心轨迹 + GT/Pred 表面对比

### 输入输出汇总

```
输入：
  Stage-3B model_best.pth
  Stage-2 t0/t1 volumes + masks
  （未来）Stage-1B: intensity_stats.json, mesh_k, Morton 索引

输出：data/processed/stage3c_online_mdpc_v2/
  ├── {case_id}_tracking_frames.csv    ← 每帧 path类型/Dice/drift/延迟
  ├── {case_id}_state_log.jsonl        ← 每帧状态（质心/体积/置信度）
  ├── {case_id}_trajectory_3d.png      ← 质心轨迹 3D 图
  └── summary_metrics.csv              ← TRE/延迟/路径占比 统计
```

---

## 9. Stage-4：实时手术导航与展示

### Stage-4A：术前多时刻病灶演化分析（离线）

**任务**：在术前规划阶段，分析多时间点 CT 中病灶的演化规律。

**跨时间点匹配**（Hungarian 算法）：

设第 $t$ 时刻有 $N_t$ 个病灶候选 $\{L_i^t\}$，第 $t+1$ 时刻有 $N_{t+1}$ 个候选 $\{L_j^{t+1}\}$，构造代价矩阵：

$$C_{ij} = 1 - \text{IoU}(L_i^t, L_j^{t+1}) + \lambda_d \cdot \frac{\|\mathbf{c}_i^t - \mathbf{c}_j^{t+1}\|}{\text{scale}}$$

Hungarian 匹配 $\pi^* = \arg\min \sum_{ij} \pi_{ij} C_{ij}$（$\pi_{ij} \in \{0,1\}$，双射约束）给出最优的跨帧 ID 关联。

**演化特征**（每个 lesion_id 的时序序列）：

| 特征 | 符号 | 单位 |
|------|------|------|
| 体积 | $V(t)$ | mm³ |
| 球形度 | $\phi(t) = \pi^{1/3}(6V)^{2/3} / A$ | 无量纲 |
| 质心轨迹 | $\mathbf{c}(t)$ | mm |
| 到血管距离 | $d_{\text{vessel}}(t)$ | mm |
| 长短轴比 | $\lambda_1(t)/\lambda_3(t)$ | 无量纲（PCA特征值比） |

### Stage-4B：术中实时 3D 导航（在线）

**每帧刷新的信息**（< 100ms）：

1. **病灶位置与不确定性**：
   - 质心 $\mathbf{c}_{\text{lesion}}(t)$：红色球体（VTK/3D Slicer 渲染）
   - 不确定性椭球（协方差 $\Sigma_{\text{lesion}}(t)$ 的 $2\sigma$ 包络）

2. **安全裕量面板**（基于 EDT 预计算，O(1) 查表）：

   ```
   病灶体积: 12.3 cm³  (↑0.1 vs t-1)
   到血管距离:  4.2 mm  ⚠ WARNING
   到切缘距离:  8.1 mm  ✓ SAFE
   追踪置信度: 0.91
   ```

3. **探针位置查询**（BIM O_index 直接应用）：

   导航探针的 EM 追踪坐标 $\mathbf{p}_{\text{probe}}(t)$ 通过 Morton 索引查最近结构：

   $$S_k^* = \arg\min_k d(\mathbf{p}_{\text{probe}}(t),\ \text{proxy}_k)$$

   延迟：< 1ms（Morton 索引直接支持此类点查询）。

### 验收标准

| 指标 | 目标值 |
|------|--------|
| 追踪 TRE（肝脏） | < 5mm |
| 追踪 TRE（脑部） | < 2mm |
| 实时刷新延迟 P95 | < 100ms/帧 |
| 安全裕量误差 | ± 1mm |
| 形变配准延迟 | < 2s/触发 |
| ID 连续性 | > 95% |

---

## 10. 符号表

| 符号 | 含义 | 单位/范围 |
|------|------|----------|
| $I_t$ | $t$ 时刻的 CT 体数据 | HU（$-1000 \sim +3000$） |
| $I_t(v)$ 或 $I_t(i,j,k)$ | 体素 $v=(i,j,k)$ 的 CT 强度 | HU |
| $M_k$ 或 $M_k^{(s)}$ | 第 $k$ 类结构的 Mask（样本 $s$） | 二值 $\{0,1\}$ |
| $\hat{M}_k$ | 预测的第 $k$ 类 Mask | 二值 $\{0,1\}$ |
| $P_k(v)$ | 体素 $v$ 属于类别 $k$ 的预测概率 | $[0,1]$ |
| $K$ | 解剖结构类别总数 | 正整数（BTCV: 13，TotalSeg: 104） |
| $\mathbf{c}_k(t)$ | 结构 $k$ 在 $t$ 帧的质心（世界坐标） | mm，$\mathbb{R}^3$ |
| $V_k(t)$ | 结构 $k$ 在 $t$ 帧的体积 | mm³ |
| $\mu_k, \sigma_k$ | 结构 $k$ 的 CT 强度均值/标准差先验 | HU |
| $\theta_k$ | O_assoc 强度容差（$= 2\sigma_k$） | HU |
| $\varepsilon$ | O_assoc 几何距离阈值 | mm |
| $\text{drift\_mm}_k$ | 结构 $k$ 两帧间的质心位移距离 | mm |
| $\delta_{\text{fast}}$ | 快速路径阈值（默认 3mm） | mm |
| $\delta_{\text{deform}}$ | 形变路径阈值（默认 10mm） | mm |
| $\text{ROI}_k$ | 结构 $k$ 的在线候选体素集合 | 体素子集 |
| $m_k$ | O_index 的自适应 ROI 半径 | mm，$[4, 18]$ |
| $\mathbf{s}_k(t)$ | 结构 $k$ 的状态向量 | $\mathbb{R}^7$ |
| $\alpha_k$ | O_proj 状态融合权重 | $[0,1]$ |
| $\text{event\_score}_k$ | 综合事件评分 | $[0,1]$ |
| $w_i$ | O_proj 距离加权系数 | $= \exp(-d_i^2/\sigma^2)$ |
| $\hat{\Phi}_k$ | 结构 $k$ 的估计形变场 | $\mathbb{R}^3 \to \mathbb{R}^3$ |
| $\mathcal{L}_{\text{sim}}$ | 图像相似度损失（NCC/MI） | 实数 |
| $\mathcal{L}_{\text{reg}}$ | 形变场正则化损失（$\|\nabla\Phi\|^2$） | 实数 |
| $d_{\text{vessel}}(t)$ | 病灶到最近血管边界的距离 | mm |
| $\text{EDT}(\cdot)$ | 欧氏距离变换 | 返回距离图（mm） |
| $\text{TRE}(t)$ | 目标配准误差 | mm |
| $\text{drift\_mm}$ | 帧间质心位移（简写） | mm |
| $\text{vol\_change}$ | 帧间体积变化率 | $[0, +\infty)$ |
| $(v_x, v_y, v_z)$ | 体素三维整数坐标 | 整数，$[0, 2^{21})$ |
| $\text{expand}(v)$ | Morton 编码位展开（21bit→63bit） | 非负整数 |
| $H_t$ | $t$ 时刻的 Morton 空间哈希索引 | 哈希表 |
| $\mathcal{A}(p_i)$ | 点 $p_i$ 的 27-邻域候选表面集合 | 表面子集 |
| $N(S_k)$ | 表面 $S_k$ 的关联点/体素集合 | 点/体素子集 |
| $\text{proxy}_k$ | 结构 $k$ 的表面代理体素集合 | 体素集合 |
| $\mathbf{A}$ | NIfTI 仿射矩阵（体素→世界坐标） | $\mathbb{R}^{4 \times 4}$，单位 mm |
| $l_{\text{patch}}$ | patch 划分格子尺寸（Stage-1） | mm（默认 20mm） |
| $\alpha$ | 时间插值系数（Stage-3C） | $[0,1]$ |
| $p(v), g(v)$ | 体素 $v$ 的预测概率/真值标签 | $[0,1]$ / $\{0,1\}$ |

---

*文档生成时间：2026-02-26。随各 Stage 实现进展持续更新。*
