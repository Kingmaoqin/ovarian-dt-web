# MDBIMDT 各 Stage 技术详解（数学版）

> 读者定位：熟悉线性代数、概率统计、基础优化理论，但对 3D 医学图像处理不熟悉。
> 本文对每个算法给出完整的数学定义、推导和复杂度分析，并与代码实现一一对应。

---

## 目录

1. [项目问题形式化](#0-项目问题形式化)
2. [Stage-0：BIM 基线实验](#stage-0-bim-基线实验)
3. [Stage-1：医学表面模型构建](#stage-1-医学表面模型构建)
4. [Stage-2：观测流与增量事件定义](#stage-2-观测流与增量事件定义)
5. [Stage-3：分割与增量更新分割](#stage-3-分割与增量更新分割)
6. [端到端数据流与复杂度汇总](#6-端到端数据流与复杂度汇总)

---

## 0. 项目问题形式化

### 0.1 核心问题定义

设 $\mathcal{S} = \{S_1, S_2, \ldots, S_K\}$ 为一组**已知语义结构**的集合（BIM 中的建筑构件，或医学中的解剖结构）。每个结构 $S_k$ 关联一个**状态向量** $\mathbf{s}_k(t) \in \mathbb{R}^d$，包含几何、纹理、信号等属性。

在时刻 $t$ 到来一批新观测 $\mathcal{O}_t = \{o_1, o_2, \ldots, o_N\}$（点云 / 图像帧），目标是求：

$$\hat{\mathbf{s}}_k(t) = f\bigl(\mathbf{s}_k(t-1),\ \mathcal{O}_t^{(k)}\bigr) \quad \forall k$$

其中 $\mathcal{O}_t^{(k)} \subseteq \mathcal{O}_t$ 是与结构 $S_k$ **局部相关**的观测子集。

### 0.2 两种更新策略的复杂度对比

**全局盲搜（Baseline）**：对每个观测 $o_i$，与所有结构的所有候选点做距离计算。
- 设总候选点数为 $M$，总观测数为 $N$，时间复杂度 $O(NM)$。

**结构感知局部更新（本方法）**：利用结构先验，将每个 $o_i$ 的候选集从 $M$ 压缩到 $|\mathcal{C}_i| \ll M$。
- 时间复杂度 $O(N \cdot \bar{c})$，其中 $\bar{c} = \frac{1}{N}\sum_i |\mathcal{C}_i|$。
- 本项目 Stage-0 实测：$\bar{c}/M = \text{selected\_ratio} = 0.0838$，即候选集压缩到全局的 8.38%。

---

## Stage-0：BIM 基线实验

> 脚本：`scripts/stage0_bim_baseline.py`
> 底层：`experiments/scripts/scan2bim_stage_runner.py`
> 输出：`runs/stage0_bim_baseline/`

### 1. 四阶段流水线的数学角色

| 阶段 | 输入 | 输出 | 数学任务 |
|---|---|---|---|
| **prep** | 原始点云 $\mathcal{P}_{\text{raw}}$ | 降采样点云 $\mathcal{P}$ | 体素降采样或步长采样 |
| **index** | $\mathcal{P}$，BIM 表面集 $\mathcal{S}$ | 空间索引结构 $\mathcal{I}$ | 构建加速查询的数据结构 |
| **assoc** | $\mathcal{P}$，$\mathcal{I}$ | 关联对 $(o_i, S_{k(i)})$ | 近邻搜索 / 候选筛选 |
| **proj** | 关联对，$\mathbf{s}_k(t-1)$ | $\hat{\mathbf{s}}_k(t)$ | 状态更新（投影/融合） |

---

### 2. Morton 编码：空间填充曲线的局部性保证

**问题**：三维点云存储在线性内存中，空间上相邻的点在内存中可能相距甚远，导致 CPU 缓存（cache）频繁 miss，访存效率低。

**Morton 编码定义**：

设整数坐标 $(x, y, z) \in \mathbb{Z}^3$，各坐标的二进制展开为：

$$x = \sum_{i=0}^{B-1} x_i \cdot 2^i, \quad y = \sum_{i=0}^{B-1} y_i \cdot 2^i, \quad z = \sum_{i=0}^{B-1} z_i \cdot 2^i$$

Morton 码定义为三路比特交织（bit interleaving）：

$$\text{Morton}(x,y,z) = \sum_{i=0}^{B-1} \bigl(x_i \cdot 2^{3i} + y_i \cdot 2^{3i+1} + z_i \cdot 2^{3i+2}\bigr)$$

**具体例子**（$B=4$ 位）：

```
x = 0101₂  →  x 的各位: x₃=0, x₂=1, x₁=0, x₀=1
y = 0011₂  →  y 的各位: y₃=0, y₂=0, y₁=1, y₀=1
z = 0001₂  →  z 的各位: z₃=0, z₂=0, z₁=0, z₀=1

Morton = z₃y₃x₃ z₂y₂x₂ z₁y₁x₁ z₀y₀x₀
       =  0 0 0   0 0 1   0 1 0   1 1 1
       = 000001010111₂ = 87
```

**局部性保证**（定性）：

若两点的 L∞ 距离 $\|p - q\|_\infty \leq 2^r$，则其 Morton 码之差满足 $|\text{Morton}(p) - \text{Morton}(q)| \leq 7 \cdot 4^r$（仅高 $r$ 位可能不同）。换言之，空间上靠近的点，Morton 码也靠近，从而在内存排布上也靠近。

**对 cache 命中率的影响**：将 $N$ 个点按 Morton 序排列后，访问某点的近邻时，大概率已在 L1/L2 cache 中，减少主存访问（DRAM latency ~100 ns vs cache ~1 ns）。

---

### 3. BIM 感知候选压缩：轴对齐包围盒过滤

设每个 BIM 构件 $S_k$ 有轴对齐包围盒（Axis-Aligned Bounding Box，AABB）：

$$\text{AABB}(S_k) = [\ell_k^x, u_k^x] \times [\ell_k^y, u_k^y] \times [\ell_k^z, u_k^z]$$

对于新观测点 $o_i = (x_i, y_i, z_i)$，设一个搜索半径 $r$，则候选集定义为：

$$\mathcal{C}_i = \bigl\{k : \text{AABB}(S_k) \cap B(o_i, r) \neq \varnothing \bigr\}$$

其中 $B(o_i, r)$ 是以 $o_i$ 为中心、半径 $r$ 的球。AABB 与球的相交判断：

$$\text{dist}(o_i, \text{AABB}(S_k)) = \sqrt{\sum_{d \in \{x,y,z\}} \max(0,\ \ell_k^d - o_i^d,\ o_i^d - u_k^d)^2} \leq r$$

此过滤在 $O(K)$ 内完成（$K$ 为构件数），使后续精确关联只需对 $|\mathcal{C}_i|$ 个候选计算，实测压缩比：

$$\text{selected\_ratio} = \frac{\sum_i |\mathcal{C}_i|}{N \cdot K} = 0.0838$$

---

### 4. 代码结构解析

```python
STAGES = ["prep", "index", "assoc", "proj"]
IMPLS  = ["cpu", "morton_ordered", "gpu_generic", "gpu_bimaware"]
```
构成 $4 \times 4 = 16$ 组实验，由子进程并发调度：

```python
def run_stage(python_exec, runner, stage, impl, ...):
    cmd = [python_exec, str(runner), "--stage", stage, "--impl", impl, ...]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return json.loads((artifact_dir / "metrics.json").read_text())
```
每次调用将指标写入 `metrics.json`，父进程汇总后写出 `summary.md/csv/json`。

---

### 5. 输入 / 输出 与关键指标

**输入**：合成点云 `synth1.zip`（约 25,920 点，dense 配置 300,000 点），BIM 表面数 $K=1024$（dense: 8192）。

**输出关键指标**：

| 指标 | 数学含义 |
|---|---|
| `runtime_sec` | 各阶段挂钟时间 $T_{\text{wall}}$（秒） |
| `selected_ratio` | $\frac{\sum_i \|\mathcal{C}_i\|}{NK}$，候选压缩比 |
| `avg_candidates` | $\bar{c} = \frac{1}{N}\sum_i \|\mathcal{C}_i\|$，平均候选数 |
| `device_mode` | 0=CPU fallback，1=CUDA GPU |

**实测结果**（quickstart 配置）：

```
gpu_bimaware | index: 0.0348s, selected_ratio=0.0838
cpu          | index: 0.1167s, selected_ratio=1.0000
→ index 阶段加速比 ≈ 3.35×，搜索空间压缩 ≈ 11.9×
```

> **注意**：当前所有变体 `device_mode=0`，GPU 分支为算法命名，非 CUDA 实测。dense 配置重跑后行为变化更显著（见 `diagnosis.md`）。

---

## Stage-1：医学表面模型构建

> 脚本：`scripts/stage1_build_surfaces.py`
> 数据：MSD Task09_Spleen（3 病例：spleen_10/12/13）
> 输出：`data/processed/stage1_surfaces_mdpc/`

### 1. 数据表示：体素体与仿射坐标系

CT 图像被表示为三维离散数组：

$$I : \{0,\ldots,D_x-1\} \times \{0,\ldots,D_y-1\} \times \{0,\ldots,D_z-1\} \to \mathbb{R}$$

每个位置 $(i,j,k)$（体素坐标）对应的物理世界坐标通过 **4×4 仿射矩阵** $A$ 变换：

$$\begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix} = A \begin{pmatrix} i \\ j \\ k \\ 1 \end{pmatrix}, \quad A = \begin{pmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

其中 $3\times3$ 子矩阵 $(r_{ij})$ 编码了**体素间距**（voxel spacing）和**方向余弦**（direction cosines，描述扫描机坐标轴相对于解剖坐标系的朝向），平移向量 $(t_x, t_y, t_z)$ 是原点偏移。代码实现：

```python
verts_world = nib.affines.apply_affine(label_img.affine, verts)
# 等价于：verts_world = (A @ np.column_stack([verts, np.ones(len(verts))]).T).T[:, :3]
```

标注体素（掩膜）定义为：

$$M = \{(i,j,k) : L(i,j,k) = 1\} \subseteq \mathbb{Z}^3$$

其中 $L$ 为整数标注数组，$1$ 表示脾脏。spleen_10：$|M| = 53{,}071$；spleen_12：$|M| = 537{,}685$。

---

### 2. Marching Cubes：等值面提取

**问题形式化**：给定标量场 $f : \mathbb{R}^3 \to \mathbb{R}$（由 $M$ 插值得到），提取等值面：

$$\Sigma_c = \{p \in \mathbb{R}^3 : f(p) = c\}$$

代码中取 $c = 0.5$（将 $\{0,1\}$ 二值掩膜视为在 $[0,1]$ 上连续，边界恰好在 $0.5$ 处）。

**算法核心**：对网格中每个元胞（voxel cube），考察其 8 个角点的函数值是否超过 $c$。每个角点有两态（$\geq c$ 或 $< c$），共 $2^8 = 256$ 种构型，经旋转/对称等价后归纳为 **15 种本质不同的三角形拼接模式**（Lorensen & Cline, 1987 的 case table）。

对于单个元胞，设角点 $v_0, \ldots, v_7$ 的函数值为 $f_0, \ldots, f_7$，构型索引为：

$$\text{idx} = \sum_{i=0}^{7} \mathbf{1}[f_i \geq c] \cdot 2^i \in \{0, 1, \ldots, 255\}$$

查表得到需要在哪些**棱（edge）**上插值放置三角形顶点。对棱 $e$ 连接角点 $v_a$（值 $f_a$）和 $v_b$（值 $f_b$），顶点在棱上的线性插值位置为：

$$v_e = v_a + \frac{c - f_a}{f_b - f_a}(v_b - v_a)$$

**输出**：顶点集 $V \subset \mathbb{R}^3$（世界坐标，mm）和面片集 $F \subset \{0,\ldots,|V|-1\}^3$（每行3个顶点索引，构成三角形）。

**复杂度**：$O(|M|)$，每个体素元胞独立处理。

**实测网格规模**：

| 病例 | $|M|$（体素） | $|V|$（顶点） | $|F|$（面片） | $|F|/|V| \approx$ |
|---|---:|---:|---:|---:|
| spleen_10 | 53,071 | 19,246 | 38,488 | 2.00 |
| spleen_12 | 537,685 | 64,492 | 128,980 | 2.00 |
| spleen_13 | 203,903 | 39,722 | 79,440 | 2.00 |

> 比值 $|F|/|V| \approx 2$ 是封闭流形三角网格的 Euler 特征数（$V - E + F = 2$，对球拓扑）所决定的渐近比例，印证了网格拓扑结构的正确性。

---

### 3. Patch 图构建：空间均匀分区 + 邻接图

**分区（Spatial Hashing）**：

设 patch 边长 $\delta = 20\,\text{mm}$。将顶点 $v \in V$（世界坐标）映射到整数格子键：

$$\text{key}(v) = \left\lfloor \frac{v - v_{\min}}{\delta} \right\rfloor \in \mathbb{Z}^3$$

其中 $v_{\min} = \min_{v \in V} v$（逐分量取最小）。同一格子内的所有顶点属于同一 patch 节点：

$$P_n = \{v \in V : \text{key}(v) = \kappa_n\}, \quad n = 1, \ldots, N_P$$

每个节点 $n$ 存储：质心 $\bar{v}_n = \frac{1}{|P_n|}\sum_{v \in P_n} v$，包围盒 $[\min P_n, \max P_n]$。

**邻接图**：定义无向图 $G = (V_G, E_G)$，其中 $V_G$ 为 patch 节点集，边集为：

$$E_G = \bigl\{\{n_1, n_2\} : \exists \text{ 三角形 } (a,b,c) \in F,\ \text{key}(a) = \kappa_{n_1},\ \text{key}(b) = \kappa_{n_2},\ n_1 \neq n_2 \bigr\}$$

即：若一个三角形的顶点跨越两个不同 patch，则这两个 patch 之间连一条边。代码：

```python
for tri in faces:
    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
    pa, pb, pc = key_to_node_id[keys[a]], key_to_node_id[keys[b]], key_to_node_id[keys[c]]
    for u, v in ((pa,pb),(pb,pc),(pa,pc)):
        if u != v:
            edge_set.add((min(u,v), max(u,v)))  # 集合去重保证无重边
```

**图统计**（三病例合并后）：

| 总节点 $|V_G|$ | 总边 $|E_G|$ | 平均度 $\bar{d} = 2|E_G|/|V_G|$ |
|---:|---:|---:|
| 306 | 902 | 5.90 |

平均度 ≈ 6 符合三维流形网格的局部拓扑预期（每个 patch 约与 6 个邻居相邻）。

---

### 4. 输入 / 输出

**输入**：
```
data/raw/Task09_Spleen/
├── imagesTr/spleen_{10,12,13}.nii.gz   # I : Z^3 → R（HU 强度值）
└── labelsTr/spleen_{10,12,13}.nii.gz   # L : Z^3 → {0,1}
```

**输出**：
```
data/processed/stage1_surfaces_mdpc/
├── meshes/spleen_{10,12,13}_label1.ply  # (V, F) 三角网格
├── surface_graph.json                   # G = (V_G, E_G)，含每节点质心/包围盒
├── cases_summary.json                   # 每病例元数据
└── summary.md
```

---

## Stage-2：观测流与增量事件定义

> 脚本：`scripts/stage2_build_observation_stream.py`
> 输出：`data/processed/stage2_observation_stream_mdpc/`

### 1. 伪在线双帧流的生成模型

由于缺少真实时序 CT，采用**合成扰动**构造 $t_0 \to t_1$ 的状态变化。对每个病例，生成两帧：

$$\text{Frame}_{t_0}: (I_{\text{crop}},\ M)$$
$$\text{Frame}_{t_1}: (I'_{\text{crop}},\ M')$$

其中 $M'$ 和 $I'$ 由以下两步扰动构造。

---

### 2. 强度归一化：鲁棒线性变换

原始 CT 强度值 $I(i,j,k) \in \mathbb{R}$（HU 单位，范围约 $[-1000, +3000]$）。

**鲁棒归一化**（代码：`robust_normalize`）：

$$\hat{I} = \frac{\text{clip}(I,\ q_{0.01},\ q_{0.99}) - q_{0.01}}{q_{0.99} - q_{0.01}} \in [0,1]$$

其中 $q_p = \text{quantile}(I, p)$。使用 1% / 99% 而非 min/max 的原因：

- CT 图像常含金属伪影（值可达 $+30000$ HU）或空气（固定 $-1000$ HU），为离群值
- 样本分位数是均值/方差的 **鲁棒估计量**，breakdown point 为 $\min(p, 1-p)$；取 $p=0.01$ 时，最多 1% 的污染数据不影响归一化参数

**AABB 裁剪**（代码：`mask_bbox`）：

$$\text{bbox}(M) = [\mathbf{l} - \Delta, \mathbf{u} + \Delta], \quad \mathbf{l} = \min_{(i,j,k)\in M}(i,j,k),\quad \mathbf{u} = \max_{(i,j,k)\in M}(i,j,k)$$

取扩展量 $\Delta = 16$ 体素，裁剪后的子体积 $I_{\text{crop}} = I[\text{bbox}(M)]$ 大幅减小后续计算规模。

---

### 3. 几何扰动：整体平移模型

**平移生成**：对每个病例采样平移向量 $\Delta \mathbf{v} \sim \text{Uniform}(\{-3,\ldots,3\}^3 \setminus \{\mathbf{0}\})$（整数体素单位）：

```python
shift_vox = tuple(int(v) for v in rng.integers(-3, 4, size=3))
```

**带边界补零的平移**（代码：`shift_mask_no_wrap`）：

$$M'(i,j,k) = \begin{cases} M(i - \Delta v_x,\ j - \Delta v_y,\ k - \Delta v_z) & \text{if } (i-\Delta v_x, j-\Delta v_y, k-\Delta v_z) \in [0,D)^3 \\ 0 & \text{otherwise} \end{cases}$$

使用 `np.roll` 实现循环移位后，对越界部分置零，等价于零填充（无循环边界）。

**几何漂移量**：比较 $t_0$ 和 $t_1$ 的**质心**（mask 上的均匀分布期望）之差：

$$\bar{\mathbf{p}}_{t} = \mathbb{E}_{(i,j,k) \sim \text{Uniform}(M_t)}\bigl[A \cdot (i,j,k,1)^\top\bigr] = \frac{1}{|M_t|} \sum_{(i,j,k) \in M_t} A \cdot (i,j,k,1)^\top$$

```python
drift_mm = float(np.linalg.norm(c1 - c0))   # ||p̄_t1 - p̄_t0||₂，单位 mm
```

**理论值**：$\Delta \mathbf{v}$ 各分量 $\sim \text{Uniform}(-3,3)$ 整数，期望平移 $\mathbb{E}[\|\Delta \mathbf{v}\|_2] = \mathbb{E}\bigl[\sqrt{\Delta v_x^2 + \Delta v_y^2 + \Delta v_z^2}\bigr]$；再经仿射矩阵变换到 mm，spleen_10 实测 6.115 mm（最大偏移），spleen_13 为 1.660 mm（最小）。

---

### 4. 强度扰动：高斯径向基函数叠加

在 mask $M'$ 内随机选中心 $\mathbf{c}_0 \in M'$，在半径 $r \sim \text{Uniform}(5,10)$ 的球形邻域内叠加 Gaussian 形状的强度凸起：

$$\hat{I}'(\mathbf{p}) = \hat{I}(\mathbf{p}) + \delta \cdot \exp\!\left(-\frac{\|\mathbf{p} - \mathbf{c}_0\|_2^2}{2\sigma^2}\right) \cdot \mathbf{1}[\|\mathbf{p}-\mathbf{c}_0\|_2 \leq r]$$

其中 $\delta \sim \text{Uniform}(0.12, 0.28)$ 为幅度，$\sigma^2 = r^2/4$（让球形支撑内约 $e^{-2} \approx 13.5\%$ 处强度降至 $\delta/e^2$）。代码：

```python
sigma2 = max(radius * radius / 4.0, 1.0)
weights = np.exp(-d2 / (2.0 * sigma2))          # d2 = ||p - c0||²
bump = delta * weights
patch[sphere] = np.clip(patch[sphere] + bump[sphere], 0.0, 1.0)
```

结果：`mean_intensity_delta` $= \mathbb{E}_{(i,j,k) \in M'_t}[\hat{I}'(i,j,k)] - \mathbb{E}_{(i,j,k) \in M_t}[\hat{I}(i,j,k)]$，实测约 $-0.001$ 至 $-0.020$（因 $t_1$ 的 mask 因平移而丢失边界体素，导致均值略有变化）。

---

### 5. 事件触发：分段线性评分函数

三类事件的评分 $s \in [0,1]$ 均为截断线性函数：

$$s_{\text{geom}}  = \min\!\left(1,\ \frac{\text{drift\_mm}}{8.0}\right)$$
$$s_{\text{int}}   = \min\!\left(1,\ \frac{|\text{mean\_intensity\_delta}|}{0.10}\right)$$
$$s_{\text{phys}}  = \min\!\left(1,\ \frac{\text{hr\_delta}}{12.0}\right)$$

分级规则（分段常数决策函数）：

$$\text{severity}(s) = \begin{cases} \text{high}   & s \geq 0.6 \\ \text{medium} & 0.3 \leq s < 0.6 \\ \text{low}    & s < 0.3 \end{cases}$$

**生理伪流**：合成心率 $h_t \sim \mathcal{N}(72, 1^2)$，$h_{t_1} = h_{t_0} + \mathcal{N}(0, 2^2) + 12 \cdot \min(\text{changed\_voxel\_ratio},\ 0.2)$，体现"病灶变化越大，心率波动越大"的先验假设。

**变化体素比**（Jaccard 距离的分子）：

$$\text{changed\_voxel\_ratio} = \frac{|M \triangle M'|}{|M|} = \frac{|M \setminus M'| + |M' \setminus M|}{|M|}$$

其中 $\triangle$ 为对称差。当平移 $\Delta\mathbf{v} \sim \text{Uniform}(-3,3)^3$ 时，理论上 $|M \triangle M'|/|M|$ 与 mask 的**表面积/体积比**正相关（薄器官漂移更多体素到 mask 外）。

---

### 6. 表面点云采样

再次对 $M'$ 运行 Marching Cubes，获取表面顶点 $V_{t_1}$，若 $|V_{t_1}| > n_{\text{pts}} = 18000$，做**无放回均匀随机子采样**：

$$\tilde{V}_{t} = \{v_{\pi(1)}, \ldots, v_{\pi(n_{\text{pts}})}\}, \quad \pi \sim \text{Uniform}(\text{Permutations}(|V_t|))$$

代码中等价为 `rng.choice(len(verts_vox), size=n_points, replace=False)`（Fisher-Yates 子采样）。

---

### 7. 输入 / 输出

**输入**：Stage-1 产物（`cases_summary.json`）+ 原始 CT/标注（`.nii.gz`）

**输出与内容**：

```
observation_stream.jsonl   # 6 条记录：每病例 t0 + t1，含路径引用
event_log.jsonl            # 9 条事件：每病例 3 类，含 score 和 severity
cases_metrics.csv          # 每病例：drift_mm, changed_voxel_ratio,
                           #         mean_intensity_delta, hr_delta, roi_voxels
cases/spleen_*/
  t0_volume.npz            # image(float16) + mask(uint8)，裁剪后体积
  t1_volume.npz
  t0_points.npz            # xyz(float32, N×3) + intensity(float32, N,)
  t1_points.npz
  t0_points.ply / t1_points.ply   # 彩色点云，供 3D 可视化
```

**实测事件指标**：

| 病例 | drift_mm | changed_voxel_ratio | severity(geom) |
|---|---:|---:|---|
| spleen_10 | 6.115 | 0.364 | high（$s=0.764$） |
| spleen_12 | 4.563 | 0.165 | high（$s=0.570$） |
| spleen_13 | 1.660 | 0.089 | low（$s=0.208$） |

---

## Stage-3：分割与增量更新分割

> 脚本：`scripts/stage3_incremental_segmentation.py`
> 输出：`data/processed/stage3_segmentation_mdpc_v2/`

### 1. 问题定义

给定归一化体积 $\hat{I}_t \in [0,1]^{D_x \times D_y \times D_z}$ 和真实标注 $G_t \in \{0,1\}^{D_x \times D_y \times D_z}$，求预测掩膜 $\hat{M}_t$ 使得某精度指标最优。

**双路径**：
1. **全量路径**（Full）：$\hat{M}_{t_1}^{\text{full}} = \mathcal{F}(\hat{I}_{t_1})$，不使用任何历史信息
2. **增量路径**（Incremental）：$\hat{M}_{t_1}^{\text{inc}} = \mathcal{F}_{\text{inc}}(\hat{I}_{t_1},\ \hat{M}_{t_0}^{\text{full}},\ \text{drift\_mm})$

---

### 2. 高斯平滑：各向同性 Gaussian 卷积

预处理步骤：

$$\hat{I}_s(\mathbf{p}) = (G_\sigma * \hat{I})(\mathbf{p}) = \sum_{\mathbf{q}} G_\sigma(\mathbf{p}-\mathbf{q}) \cdot \hat{I}(\mathbf{q})$$

$$G_\sigma(\mathbf{u}) = \frac{1}{(2\pi\sigma^2)^{3/2}} \exp\!\left(-\frac{\|\mathbf{u}\|_2^2}{2\sigma^2}\right), \quad \sigma = 0.8 \text{（体素单位）}$$

在 3D 中，各向同性 Gaussian 核可分离为三个 1D 卷积，复杂度 $O(D_x D_y D_z \cdot K_\sigma)$，其中 $K_\sigma \approx 6\sigma + 1 \approx 6$ 体素为核半径截断。

**作用**：平滑高频噪声（CT 量化噪声、图像重建伪影），使后续阈值分割更稳定。

---

### 3. Otsu 阈值：类间方差最大化

设体积 $\hat{I}_s$ 中取值在 $[0,1]$，对其直方图归一化后视为概率质量函数：$p_k = P(\hat{I}_s = k/L)$，$L$ 为灰度级数（256）。

**总方差分解**（Fisher's ANOVA 类比）：

$$\sigma_T^2 = \sigma_W^2(T) + \sigma_B^2(T)$$

- 总方差 $\sigma_T^2$ 与 $T$ 无关（固定值）
- 类内方差：$\sigma_W^2(T) = \omega_0(T)\sigma_0^2(T) + \omega_1(T)\sigma_1^2(T)$
- 类间方差：$\sigma_B^2(T) = \omega_0(T)\omega_1(T)[\mu_0(T) - \mu_1(T)]^2$

其中对阈值 $T$：

$$\omega_0(T) = \sum_{k=0}^{T} p_k, \qquad \omega_1(T) = \sum_{k=T+1}^{L} p_k = 1 - \omega_0(T)$$
$$\mu_0(T) = \frac{\sum_{k=0}^{T} k \cdot p_k}{\omega_0(T)}, \qquad \mu_1(T) = \frac{\sum_{k=T+1}^{L} k \cdot p_k}{\omega_1(T)}$$

由于 $\sigma_T^2$ 固定，**最大化类间方差等价于最小化类内方差**：

$$T^* = \arg\max_{T} \sigma_B^2(T) = \arg\max_T \omega_0(T)\omega_1(T)\bigl[\mu_0(T) - \mu_1(T)\bigr]^2$$

**代码中的修改**：仅对强度 $\geq q_{0.25}$（即 25th 百分位）的体素计算 Otsu，相当于剔除背景空气（CT 中强度极低的区域），让 Otsu 在"前景主导"的分布上工作：

```python
vals = x_s[x_s > np.percentile(x_s, 25)]  # 去除空气体素
thr  = float(filters.threshold_otsu(vals))
pred = x_s > thr
```

**局限性**：Otsu 方法假设直方图为**双峰分布**。但 CT 腹部图像中，肌肉、脂肪、器官、血液的 HU 值分布重叠，使得 Otsu 阈值不能精确分离脾脏，导致 Dice $\approx 0.40$-$0.56$（远低于 nnUNet 的 $\geq 0.95$）。

---

### 4. 形态学后处理

设结构元素（Structuring Element）为单位球 $B_1 = \{(x,y,z) \in \mathbb{Z}^3 : x^2+y^2+z^2 \leq 1\}$（即26邻域球，代码中 `morphology.ball(1)`）。

**形态学膨胀（Dilation）**（Minkowski 和）：

$$A \oplus B = \{a + b : a \in A, b \in B\} = \bigcup_{b \in B} (A + b)$$

**形态学腐蚀（Erosion）**（Minkowski 差）：

$$A \ominus B = \{p : B + p \subseteq A\} = \bigcap_{b \in B} (A - b)$$

**闭运算（Closing）**：先膨胀再腐蚀，填充直径 $\leq 2$ 体素的孔洞：

$$A \bullet B = (A \oplus B) \ominus B$$

**开运算（Opening）**：先腐蚀再膨胀，去除面积 $\leq$ 结构元素的孤立前景岛：

$$A \circ B = (A \ominus B) \oplus B$$

**代码实现**：

```python
def postprocess(mask):
    out = morphology.closing(mask, morphology.ball(1))        # A ● B：填孔
    out = morphology.opening(out,  morphology.ball(1))        # A ○ B：去噪点
    out = morphology.remove_small_holes(out, max_size=128)    # 填充体积 ≤ 128 体素的孔
    out = keep_best_component(out, min_size=100)               # 保留最优连通域
    return out.astype(bool)
```

---

### 5. 连通域分析：最优分量选取

**定义**：掩膜 $\hat{M}$ 的**连通分量**（6-邻域或26-邻域连通）为等价类 $[v]_\sim$，其中 $u \sim v$ 当且仅当存在从 $u$ 到 $v$ 的体素路径（逐步邻接）。

代码用 `scipy.ndimage.label` 实现 3D 连通分量标注（26-连通），返回每个体素的分量 ID。

**分量评分函数**：

$$\text{score}(C_k) = \frac{|C_k|}{1 + \|\bar{c}_k - \mathbf{c}_{\text{center}}\|_2}$$

其中 $\bar{c}_k = \frac{1}{|C_k|}\sum_{v \in C_k} v$ 为分量质心，$\mathbf{c}_{\text{center}}$ 为体积中心。

**偏好**：体积大（$|C_k|$ 大）且位置居中（$\|\bar{c}_k - \mathbf{c}_{\text{center}}\|_2$ 小）的分量。因为 Stage-2 裁剪时以器官包围盒为中心，正确分量应在裁剪体的中央区域。

```python
score = float(size / (1.0 + dist))  # size = |C_k|, dist = ||c̄_k - c_center||₂
```

---

### 6. 增量分割：基于漂移量的自适应 ROI

**ROI 生成**：将上一帧预测掩膜 $\hat{M}_{t_0}$ 做二值膨胀 $m$ 次：

$$\text{ROI}_{t_1} = \hat{M}_{t_0} \oplus B_1^{(m)} = \underbrace{(\hat{M}_{t_0} \oplus B_1) \oplus \cdots \oplus B_1}_{m \text{ 次}}$$

等价于：$\text{ROI}_{t_1} = \{v : d(v, \hat{M}_{t_0}) \leq m\}$，即到 $\hat{M}_{t_0}$ 的 L∞ 距离（体素）不超过 $m$ 的所有体素。

**自适应边界扩张量**：

$$m = \text{clip}\!\left(\left\lfloor \frac{\text{drift\_mm}}{2} + 0.5 \right\rfloor + 4,\quad 4,\quad 18\right)$$

- $\text{drift\_mm} / 2$：漂移量转化为需要的体素扩张（假设体素间距 $\approx 2\,\text{mm}$）
- $+4$：最小安全边界（补偿漂移估计误差）
- $\text{clip}(4, 18)$：防止 ROI 过小（漂移很小时）或过大（漂移很大时退化为全局）

**ROI 体积压缩比**：

$$\rho = \frac{|\text{ROI}_{t_1}|}{|D|} = \text{roi\_ratio}$$

**增量 Otsu**：在 ROI 内执行全量分割的变体：

```python
vals   = x_s[roi]                              # 仅取 ROI 内的强度值
thr    = float(filters.threshold_otsu(vals))   # ROI 内的 Otsu 阈值
pred   = np.zeros_like(prev_mask, dtype=bool)
pred[roi] = (x_s > thr)[roi]                   # 只在 ROI 内写入结果
```

**计算复杂度**：从 $O(|D|)$ 降至 $O(|\text{ROI}|) = O(\rho \cdot |D|)$，理论加速比 $\approx 1/\rho$（忽略 ROI 构建开销）。

---

### 7. 评估指标的数学定义

#### 7.1 Dice 系数（等价于 F₁ Score）

设预测 $\hat{M}$，真值 $G$（均为 $\{0,1\}^{D_x D_y D_z}$ 集合），定义：

$$\text{Dice}(\hat{M}, G) = \frac{2|\hat{M} \cap G|}{|\hat{M}| + |G|} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

与精确率 $P = \text{TP}/(\text{TP}+\text{FP})$ 和召回率 $R = \text{TP}/(\text{TP}+\text{FN})$ 的关系：

$$\text{Dice} = \frac{2PR}{P + R} = F_1$$

与 IoU（Jaccard 系数）的关系：

$$\text{IoU} = \frac{|\hat{M} \cap G|}{|\hat{M} \cup G|} = \frac{\text{Dice}}{2 - \text{Dice}}$$

#### 7.2 HD95（95th Percentile Hausdorff Distance）

**双向 Hausdorff 距离**：

$$d_H(A, B) = \max\!\left(\sup_{a \in \partial A} d(a, \partial B),\ \sup_{b \in \partial B} d(b, \partial A)\right)$$

其中 $\partial A$ 为 $A$ 的表面（通过 1 步腐蚀提取），$d(a, \partial B) = \inf_{b \in \partial B}\|a - b\|_2$。

标准 $d_H$ 对异常值极敏感（单个噪点体素就可以使 $d_H$ 飙升）。**HD95** 使用经验分位数替代 supremum：

$$\text{HD95}(\hat{M}, G) = \text{quantile}_{0.95}\!\bigl(D_{AB} \cup D_{BA}\bigr)$$

$$D_{AB} = \{d(a, \partial G) : a \in \partial \hat{M}\}, \quad D_{BA} = \{d(b, \partial \hat{M}) : b \in \partial G\}$$

**实现**：通过**欧氏距离变换（EDT）**高效计算：

$$\text{EDT}(\mathbf{p}; \partial G) = \min_{\mathbf{q} \in \partial G} \|\mathbf{p} - \mathbf{q}\|_2$$

`scipy.ndimage.distance_transform_edt` 实现了 Saito & Toriwaki (1994) 算法（或 Meijster 变种），时间复杂度 $O(D_x D_y D_z)$，比逐点搜索 $O(|\partial \hat{M}| \cdot |\partial G|)$ 快得多。

```python
dt_b = ndi.distance_transform_edt(~sb)   # EDT from ∂G，对全体积
d_ab = dt_b[sa]                           # 取 ∂M̂ 上的距离值
```

#### 7.3 实验结果与分析

| 病例 | Dice $t_0$ | Dice $t_1^{\text{full}}$ | Dice $t_1^{\text{inc}}$ | HD95 $t_1^{\text{full}}$ | HD95 $t_1^{\text{inc}}$ | $\rho$ | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| spleen_10 | 0.441 | 0.376 | **0.412** | 29.48 | 29.95 | 0.470 | **1.18×** |
| spleen_12 | 0.447 | 0.421 | **0.563** | 75.43 | 78.19 | 0.717 | 0.82× |
| spleen_13 | 0.402 | 0.394 | **0.522** | 53.91 | 51.66 | 0.669 | 0.70× |

**为何增量 Dice 普遍优于全量**：

全量 Otsu 在整个裁剪体上（包含大量背景体素）估计阈值，使直方图被背景峰主导；增量 Otsu 仅在 $\text{ROI}_{t_1}$（器官附近区域）内估计，有效剔除了背景干扰，使前/背景双峰更突出，Otsu 条件更接近满足。这是 ROI 约束的**隐式正则化**效果。

**为何 spleen_10 速度最好（1.18×）而 spleen_12 反而慢（0.82×）**：

$\rho_{\text{spleen\_10}} = 0.470 < \rho_{\text{spleen\_12}} = 0.717$。ROI 越大，增量路径越接近全量路径的计算量。spleen_12 的脾脏体积极大（$|M| = 537{,}685$），ROI 膨胀后几乎覆盖整个裁剪体（$\rho = 0.717$），导致增量路径的形态学操作等后处理开销与全量路径相当，综合时延不降反升。

---

## 6. 端到端数据流与复杂度汇总

```
原始 CT + 标注  (I : Z³→R, L : Z³→{0,1})
       │
       ▼ Stage-1: 表面提取与图构建
       │  Marching Cubes    O(|M|)
       │  Affine 坐标变换   O(|V|)
       │  空间哈希 Patch化  O(|V|)
       │  邻接图构建        O(|F|)
       │
       ├── 输出: (V, F) 网格, G=(V_G, E_G) 拓扑图
       │
       ▼ Stage-2: 伪在线观测流
       │  鲁棒归一化        O(|D|)   D = 体素总数
       │  整体平移          O(|D|)
       │  高斯 RBF 叠加     O(r³)    r = 扰动半径
       │  质心漂移计算      O(|M|)
       │  事件评分          O(1)
       │  Marching Cubes    O(|M|)（点云采样）
       │
       ├── 输出: t0/t1 体积, 点云, 事件日志
       │
       ▼ Stage-3: 双路径分割
       │
       ├─ 全量路径:
       │    高斯卷积         O(|D| · K_σ)
       │    Otsu 阈值        O(|D|)
       │    形态学后处理     O(|D|)
       │    EDT + HD95       O(|D|)
       │    → 时间: 0.13~0.56 s/case
       │
       └─ 增量路径:
            ROI 膨胀         O(|D| · m)
            Otsu (on ROI)    O(ρ|D|)
            形态学后处理     O(|D|)
            → 时间: 0.11~0.68 s/case (ρ 越小越快)
            → Speedup ≈ 1/ρ（理想情况）

评估指标:
  Dice ∈ {0.376, ..., 0.563}  （Otsu 基线，DL 模型可达 ≥ 0.95）
  HD95 ∈ {29.5, ..., 78.2} mm
  ROI ratio ρ ∈ {0.470, 0.669, 0.717}
  Speedup ∈ {0.70, 0.82, 1.18}×
```

### 当前方法的理论上限分析

Otsu 分割的性能由图像的**类可分性（class separability）**决定：

$$\eta = \frac{\sigma_B^2(T^*)}{\sigma_T^2} \in [0, 1]$$

$\eta \to 1$ 表示双峰高度可分。腹部 CT 中，脾脏（约 40-60 HU）与周围肝脏（约 50-70 HU）、肌肉（约 30-50 HU）HU 分布高度重叠，使得 $\eta$ 较低，这从根本上限制了 Otsu 的 Dice 上限。

**后续方向（Stage-3 计划中的 DL 路径）**：
- 用 **3D U-Net** 或 **nnUNet** 替换 Otsu，学习空间上下文特征
- U-Net 的编码器-解码器结构能区分同 HU 值但空间位置不同的组织
- 在线增量路径改为：对 ROI 内的小窗口做 DL 模型的快速推理（patch-wise inference）

---

*本文档生成于 2026-02-25。算法推导基于 Stage 0-3 的实际代码与输出数据。*
