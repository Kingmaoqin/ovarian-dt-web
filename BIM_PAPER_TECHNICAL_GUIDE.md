# BIM 论文技术详解：面向 Scan-to-BIM 的 BIM 感知 GPU 纹理映射管线

> 本文是对 `papers/latex/` 中论文的完整数学解析，读者须有线性代数、概率统计和基础计算机体系结构（内存层次、并行计算）背景。
> 覆盖范围：问题形式化 → 四阶段计算模型 → 每个算子的完整数学定义 → GPU 优化原理 → 性能分析 → 代码与论文的对应关系。

---

## 目录

1. [研究问题与动机](#1-研究问题与动机)
2. [Scan-to-BIM 纹理映射的计算模型](#2-scan-to-bim-纹理映射的计算模型)
3. [数据表示与内存布局](#3-数据表示与内存布局)
4. [算子一：空间索引 O_index](#4-算子一空间索引-o_index)
5. [算子二：点面关联 O_assoc](#5-算子二点面关联-o_assoc)
6. [算子三：核融合纹理投影 O_proj](#6-算子三核融合纹理投影-o_proj)
7. [性能理论分析](#7-性能理论分析)
8. [实验设计与评估指标](#8-实验设计与评估指标)
9. [从 BIM 到医学数字孪生的迁移逻辑](#9-从-bim-到医学数字孪生的迁移逻辑)
10. [代码与论文对应速查](#10-代码与论文对应速查)

---

## 1. 研究问题与动机

### 1.1 Scan-to-BIM 工作流

现代 AEC（建筑、工程、施工）项目使用激光扫描仪（LiDAR）或摄影测量采集建筑物的**点云**（Point Cloud），再将其映射到预先建立的 **BIM（Building Information Model）** 模型上——这一过程称为 **Scan-to-BIM**。

具体而言，纹理映射（Texture Mapping）步骤要把点云中每个点的颜色/纹理信息，投影到 BIM 模型的每一个表面元素上，使 BIM 模型呈现出真实外观。

### 1.2 核心瓶颈：点云与 BIM 表面的结构不匹配

设点云为 $P = \{p_i = (x_i, y_i, z_i, c_i)\}_{i=1}^N$（$c_i$ 为 RGB 颜色），BIM 表面元素集合为 $\mathcal{S} = \{S_k\}_{k=1}^K$。

**问题**：给定 $P$ 和 $\mathcal{S}$，对每个 $S_k$ 计算其纹理值 $T(S_k)$。

这在数学上等价于：**对每个表面元素 $S_k$，在点云 $P$ 中找出距离 $S_k$ 足够近的所有点，再做加权平均**。

**瓶颈的根源**：
- 点云 $P$ 是**无结构**的（unstructured）：$N$ 个点随机分布在三维空间，在内存中也没有空间局部性
- BIM 表面 $\mathcal{S}$ 是**结构化**的：墙、柱、门都有明确的几何位置和邻接关系
- 现有 GPU 方法针对"点找点"（点-点邻域搜索）设计，**忽略了 BIM 表面的结构先验**，导致 GPU 访存不规则、线程分歧（warp divergence）严重

论文的核心洞察：**将 GPU 计算范式从"点中心"（point-centric）改为"表面中心"（surface-centric）**，利用 BIM 表面的空间结构来引导数据访问，从根本上改善访存局部性和并行效率。

---

## 2. Scan-to-BIM 纹理映射的计算模型

### 2.1 端到端时间分解

论文将整个纹理映射流程分解为四个串行计算阶段（论文第3节）：

$$T_{\text{Scan2BIM}} = T_{\text{prep}} + T_{\text{index}} + T_{\text{assoc}} + T_{\text{proj}}$$

| 阶段 | 含义 | 主要计算任务 |
|---|---|---|
| $T_{\text{prep}}$ | 预处理 | 点云加载、降采样、坐标变换 |
| $T_{\text{index}}$ | 空间索引构建 | 体素化、Morton 排序、前缀和 |
| $T_{\text{assoc}}$ | 点面关联 | 对每个表面找候选点 |
| $T_{\text{proj}}$ | 纹理投影 | 加权聚合计算纹理值 |

实验分析表明：$T_{\text{index}}$ 和 $T_{\text{assoc}} + T_{\text{proj}}$ 在大规模场景下主导总时延。

### 2.2 算子合成视角

论文将纹理映射形式化为算子的函数复合（论文第3节）：

$$\mathcal{F}_{\text{BIM}}(P,\, \mathcal{S}) = \mathcal{O}_{\text{proj}}\!\bigl(\mathcal{O}_{\text{assoc}}\!\bigl(\mathcal{O}_{\text{index}}(P),\ \mathcal{S}\bigr)\bigr)$$

- $\mathcal{O}_{\text{index}}(P)$：将点云 $P$ 变换为可快速查询的空间索引结构
- $\mathcal{O}_{\text{assoc}}(\mathcal{I}, \mathcal{S})$：利用索引 $\mathcal{I}$ 为每个 $S_k$ 建立候选点集 $N(S_k)$
- $\mathcal{O}_{\text{proj}}(\{N(S_k)\})$：对每个候选集做加权聚合，输出 $T(S_k)$

这种算子抽象让每个组件可以独立优化，也便于评估 Ablation Study（消融实验）中各组件的贡献。

---

## 3. 数据表示与内存布局

### 3.1 AoS vs SoA：GPU 内存合并访问

**Array-of-Structures（AoS）**——传统布局，每个点的所有属性连续存储：

$$\text{内存} = [x_1, y_1, z_1, r_1, g_1, b_1,\ x_2, y_2, z_2, r_2, g_2, b_2,\ \ldots]$$

当 GPU warp（32个线程）同时访问 $x$ 坐标时，各线程读取的内存地址间隔为 6 个 float（每个点 6 个属性），**步长为 24 字节**，导致访存**不合并（non-coalesced）**。

**Structure-of-Arrays（SoA）**——论文采用的布局，同一属性的所有点连续存储：

$$\text{内存} = [\underbrace{x_1, x_2, \ldots, x_N}_{\text{x array}},\ \underbrace{y_1, y_2, \ldots, y_N}_{\text{y array}},\ \underbrace{z_1, \ldots}_{\text{z array}},\ \ldots]$$

当 GPU warp 同时读取 32 个点的 $x$ 坐标时，地址连续，形成单次 128 字节的合并读取（coalesced memory access），**有效带宽最大化**。

代码（`scan2bim_stage_runner.py`）中：

```python
# SoA 布局：每个属性独立的 float32 数组
points = {
    "x": np.asarray(xs, dtype=np.float32),   # shape: (N,)
    "y": np.asarray(ys, dtype=np.float32),   # shape: (N,)
    "z": np.asarray(zs, dtype=np.float32),   # shape: (N,)
    "r": np.asarray(rs, dtype=np.float32),   # shape: (N,)
    "g": np.asarray(gs, dtype=np.float32),   # shape: (N,)
    "b": np.asarray(bs, dtype=np.float32),   # shape: (N,)
}
```

---

## 4. 算子一：空间索引 $\mathcal{O}_{\text{index}}$

### 4.1 均匀体素化

将三维空间离散化为边长 $s$（体素大小，默认 $s = 0.2$ m）的均匀网格。每个点 $p_i$ 的体素坐标为：

$$\mathbf{v}(p_i) = \left(\left\lfloor\frac{x_i - x_{\min}}{s}\right\rfloor,\ \left\lfloor\frac{y_i - y_{\min}}{s}\right\rfloor,\ \left\lfloor\frac{z_i - z_{\min}}{s}\right\rfloor\right) \in \mathbb{Z}^3$$

代码：

```python
vx = np.floor(points["x"] / voxel_size).astype(np.int32)
vy = np.floor(points["y"] / voxel_size).astype(np.int32)
vz = np.floor(points["z"] / voxel_size).astype(np.int32)
```

### 4.2 体素键打包（Voxel Key Packing）

三维整数坐标 $\mathbf{v} = (v_x, v_y, v_z)$ 需要映射到一个**唯一的 64 位整数键**，用于哈希和排序。代码采用如下位移方案（每个分量 21 位，支持负坐标通过偏移量 $2^{20}$ 处理）：

$$\text{key}(v_x, v_y, v_z) = \bigl[(v_x + 2^{20}) \mathbin{\&} \texttt{0x1FFFFF}\bigr] \ll 42 \;\Big|\; \bigl[(v_y + 2^{20}) \mathbin{\&} \texttt{0x1FFFFF}\bigr] \ll 21 \;\Big|\; \bigl[(v_z + 2^{20}) \mathbin{\&} \texttt{0x1FFFFF}\bigr]$$

这保证了 $v_x, v_y, v_z \in [-2^{20}, 2^{20}-1]$ 时键唯一（$3 \times 21 = 63$ 位），并支持 O(1) 的键提取和邻域计算（加减 1 后重新打包）。

代码：

```python
def pack_voxel_keys(vx, vy, vz):
    off = 1 << 20
    xx = (vx.astype(np.int64) + off) & 0x1FFFFF
    yy = (vy.astype(np.int64) + off) & 0x1FFFFF
    zz = (vz.astype(np.int64) + off) & 0x1FFFFF
    return (xx << 42) | (yy << 21) | zz
```

### 4.3 Morton 编码：空间填充曲线排序

**Morton 码（Z-order curve）**将三维体素坐标的二进制比特三路交织，映射为一维整数，使**空间上相邻的体素在一维排列中也相邻**：

$$\text{Morton}(v_x, v_y, v_z) = \text{expand}(v_x) \;\big|\; \bigl(\text{expand}(v_y) \ll 1\bigr) \;\big|\; \bigl(\text{expand}(v_z) \ll 2\bigr)$$

其中 $\text{expand}(v)$ 将 $v$ 的每个比特扩展为每隔 3 位放置一个（其余位填 0）——即 21 位整数扩展为 63 位：

$$\text{expand}: b_{20}\,b_{19}\,\cdots\,b_1\,b_0 \;\mapsto\; b_{20}\,00\,b_{19}\,00\,\cdots\,b_1\,00\,b_0$$

代码使用**位并行展开**（parallel prefix bit spreading，也称 magic number trick）高效实现：

```python
def _expand_bits21(v):
    v = v.astype(np.uint64) & np.uint64(0x1FFFFF)        # 保留低 21 位
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)   # 第1轮分散
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)   # 第2轮
    v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F) # 第3轮
    v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3) # 第4轮
    v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249) # 第5轮：每3位1个
    return v
```

每轮掩码的含义：通过将 $v$ 与自身的不同偏移版本做 OR，逐步把比特"撑开"到正确位置。5 轮操作将 21 位整数分散到 63 位空间中，每个有效位间隔 2 个零。

```python
def compute_morton_codes(vx, vy, vz):
    off = np.uint64(1 << 20)
    x = _expand_bits21((vx.astype(np.int64) + (1 << 20)).astype(np.uint64))
    y = _expand_bits21((vy.astype(np.int64) + (1 << 20)).astype(np.uint64))
    z = _expand_bits21((vz.astype(np.int64) + (1 << 20)).astype(np.uint64))
    return (x | (y << np.uint64(1)) | (z << np.uint64(2)))
    # 比特交织顺序：z₀y₀x₀ z₁y₁x₁ z₂y₂x₂ ...（x占偶数位从0开始，y从1，z从2）
```

**Morton 编码的局部性性质**（形式化）：

对任意两个体素 $\mathbf{u}, \mathbf{v} \in \mathbb{Z}^3$，若 $\|\mathbf{u} - \mathbf{v}\|_\infty \leq 2^r$（即在 $2r+1$ 边长的立方体内），则：

$$|\text{Morton}(\mathbf{u}) - \text{Morton}(\mathbf{v})| \leq 7 \cdot 8^r - 1$$

反过来，若 $|\text{Morton}(\mathbf{u}) - \text{Morton}(\mathbf{v})| \leq C$，则两点仍在一个 $O(C^{1/3})$ 大小的立方体内（**不是精确双向等价**，但对 cache 命中已足够）。

**对 GPU 性能的影响**：

将 $N$ 个点按 Morton 码升序排列（`np.argsort(codes)`，代码 `apply_morton_sort`）后，处理某体素的邻域体素时，这些体素大概率已在 GPU L2 cache（通常 4-40 MB）中，减少 DRAM 访问（延迟约 500-800 ns 对比 L2 的 ~40 ns，差 10-20×）。

代码：

```python
def apply_morton_sort(points, voxel_size):
    vx = np.floor(points["x"] / voxel_size).astype(np.int32)
    ...
    codes = compute_morton_codes(vx, vy, vz)
    perm  = np.argsort(codes, kind="stable").astype(np.int32)  # stable 保证唯一性
    return {k: (v[perm] if v.shape[0] == N else v) for k, v in points.items()}
```

### 4.4 BIM 感知候选压缩（核心创新）

**通用 GPU 方法（`gpu_generic`）**：所有点都参与索引，`selected_mask = ones(N)`，选择比例 `selected_ratio = 1.0`。

**BIM 感知方法（`gpu_bimaware`）**：

核心假设：**只有位于 BIM 表面附近的点才可能被关联到某个表面元素**，与所有表面都距离遥远的点可以直接丢弃。

算法：

1. 随机采样 $K$ 个点作为 BIM 表面的代理（surface proxy）：

$$\text{surface\_idx} = \text{RandomSample}(\{0,\ldots,N-1\}, K)$$

2. 计算这 $K$ 个表面代理点的体素坐标 $\{(sv_x^{(k)}, sv_y^{(k)}, sv_z^{(k)})\}$

3. 对每个表面代理点，计算其 **26+1=27 邻域体素**（3×3×3 网格）的键集合：

$$\mathcal{A} = \bigcup_{k=1}^{K} \bigcup_{(\delta x, \delta y, \delta z) \in \{-1,0,1\}^3} \text{key}(sv_x^{(k)} + \delta x,\ sv_y^{(k)} + \delta y,\ sv_z^{(k)} + \delta z)$$

4. 对全部点云做集合成员测试：

$$\text{selected\_mask}[i] = \bigl[\text{key}(p_i) \in \mathcal{A}\bigr]$$

代码（CPU 路径）：

```python
neighborhood = []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            neighborhood.append(pack_voxel_keys(svx+dx, svy+dy, svz+dz))
active_keys = np.unique(np.concatenate(neighborhood))      # |A| ≤ 27K（含去重）
selected_mask = np.isin(keys, active_keys)                 # O(N log|A|) 二分搜索
```

**压缩效果分析**：

设体素边长 $s = 0.2$ m，BIM 表面覆盖的空间体积为 $V_{\mathcal{S}}$，总点云空间体积为 $V_P$。理论压缩比：

$$\text{selected\_ratio} \approx \frac{V_{\mathcal{S}} + K \cdot (3s)^3}{V_P} = \frac{V_{\mathcal{S}} + 27Ks^3}{V_P}$$

实测：`selected_ratio = 0.0838`（约 8.4%），即**候选候选集压缩了约 11.9 倍**。

### 4.5 GPU 并行前缀和（Prefix Sum）与边界计算

按体素键排序后，对已选点做**前缀和**（prefix sum，也称 scan）计算每个唯一体素的起始位置：

设已排序体素键为 $[k_1, k_2, \ldots, k_M]$（含重复），唯一键为 $[\tilde{k}_1, \ldots, \tilde{k}_U]$，各唯一键的出现次数为 $[c_1, \ldots, c_U]$，则唯一键 $\tilde{k}_j$ 对应的点集在排序数组中的位置为：

$$\text{start}(\tilde{k}_j) = \sum_{i=1}^{j-1} c_i, \quad \text{end}(\tilde{k}_j) = \sum_{i=1}^{j} c_i$$

这允许 O(1) 时间定位任意体素内的点集。代码：

```python
uniq_keys, starts, uniq_counts = np.unique(sorted_keys,
                                            return_index=True,
                                            return_counts=True)
```

### 4.6 内存访问质量度量：jump 统计

代码引入了两个量化内存访问局部性的代理指标（代码 `_jump_stats`）：

**index_order_jump_mean**：

$$\bar{J} = \frac{1}{|\text{selected}| - 1} \sum_{i=1}^{|\text{selected}|-1} \bigl|\text{selected\_idx}[i+1] - \text{selected\_idx}[i]\bigr|$$

物理含义：按关联顺序访问点时，每步在原始点云数组中跳跃的平均距离（体素数）。越大表示内存访问越分散，cache 命中率越低。

**coalescing_proxy**：

$$\text{coalescing} = \frac{1}{1 + \bar{J}} \in (0, 1]$$

值越接近 1 表示内存访问越连续（理想：$\bar{J} = 0$，即顺序访问）。Morton 排序通过让空间相邻点在内存中也相邻，直接降低 $\bar{J}$。

实测对比（`gpu_generic` 使用 shuffle，故 jump 极大；`morton_ordered` 保持空间局部性）。

---

## 5. 算子二：点面关联 $\mathcal{O}_{\text{assoc}}$

### 5.1 关联集合的数学定义

对每个 BIM 表面元素 $S_k$，关联点集定义为（论文第4节）：

$$N(S_k) = \{p_i \in P \mid d(p_i, S_k) < \varepsilon\}$$

其中 $d(p_i, S_k)$ 为点到表面的**有向距离**（signed point-to-plane distance），$\varepsilon$ 为容差阈值。

在体素化实现中，这通过**限制邻域体素搜索**来近似：只在表面 $S_k$ 所在体素的 27 邻域内寻找候选点，而非全局搜索。

### 5.2 候选数量统计与不规则性度量

代码在 `stage_assoc` 中统计每个表面的候选点数量 $|N(S_k)|$（`candidate_counts`），并计算：

**平均候选数**：

$$\bar{c} = \frac{1}{K} \sum_{k=1}^{K} |N(S_k)|$$

**候选不规则性**（Candidate Irregularity）——作为 GPU **warp 分歧（warp divergence）**的代理指标：

$$\text{irregularity} = \frac{\sigma_c}{\bar{c} + \varepsilon} = \frac{\text{std}(|N(S_k)|)}{\text{mean}(|N(S_k)|) + \varepsilon}$$

**为何不规则性与 warp 分歧相关**：

GPU 中一个 warp 的 32 个线程执行相同指令，但若各线程处理的表面元素的 $|N(S_k)|$ 差异很大，则有些线程早早完成（空转等待），有些还在处理大量候选点——这就是 warp 分歧，导致 GPU 并行效率下降。$\text{irregularity} \approx \sigma_c / \bar{c}$ 是变异系数（CV），完全均匀时 CV=0，高度不均匀时 CV 趋向无穷。

**尾部比例**：

$$\text{tail\_ratio} = \frac{\text{quantile}_{0.95}(|N(S_k)|)}{\bar{c}}$$

衡量"最重尾"的 5% 表面比平均表面多承载多少候选点。

代码：

```python
mean_c      = candidate_counts.mean()
std_c       = candidate_counts.std()
p95         = np.percentile(candidate_counts, 95)
irregularity = std_c / (mean_c + 1e-6)         # CV，warp 分歧代理
tail_ratio   = p95 / (mean_c + 1e-6)
```

### 5.3 Warp 级并行策略

论文提出将 **一个 GPU warp（32 线程）分配给一个表面元素**：

$$\text{warp}_j \to S_j$$

warp 内的 32 个线程协作遍历 $N(S_j)$ 中的候选点，计算距离和权重的归约（reduction）。这比"一线程一点"更高效，因为归约可用 warp-level primitives（`__shfl_down_sync`）实现，无需共享内存同步。

---

## 6. 算子三：核融合纹理投影 $\mathcal{O}_{\text{proj}}$

### 6.1 加权颜色聚合

表面元素 $S_k$ 的纹理值为（论文第4节）：

$$T(S_k) = \frac{\displaystyle\sum_{p_i \in N(S_k)} w_i\, c_i}{\displaystyle\sum_{p_i \in N(S_k)} w_i}$$

权重使用**高斯径向基函数（Gaussian RBF）**，以点到表面距离为参数：

$$w_i = \exp\!\left(-\frac{d(p_i, S_k)^2}{\sigma^2}\right)$$

$\sigma$ 控制权重衰减速率：距离为 $\sigma$ 的点权重为 $e^{-1} \approx 0.368$，距离为 $2\sigma$ 的点权重仅为 $e^{-4} \approx 0.018$。

**为何使用距离反比加权而非简单平均**：

BIM 扫描中，近处点的颜色更准确（反射率更高、定位误差更小）；远处点可能因遮挡、噪声产生颜色偏差。Gaussian 权重自然地给近点更高权重，实现**鲁棒的颜色融合**。

代码中的简化版本（`stage_proj`）：

```python
# 用 candidate_counts 作为距离代理（候选多 ≈ 区域稠密 ≈ 距离近）
w = np.exp(-candidate_counts / (candidate_counts.mean() + 1e-6))
T_r = (w * r).sum() / (w.sum() + 1e-6)   # 加权均值 R 通道
T_g = (w * g).sum() / (w.sum() + 1e-6)
T_b = (w * b).sum() / (w.sum() + 1e-6)
```

### 6.2 核融合：消除中间缓冲区

**分离式管线**（Separated Pipeline）：

```
关联阶段输出: N(S_k) 写入全局内存    →   投影阶段读取 N(S_k) 做聚合
└── 内存流量: M_write(assoc) + M_read(proj)
```

**核融合（Kernel Fusion）**：

将关联和投影合并为**单一 CUDA kernel**，在寄存器/共享内存中直接完成距离计算和权重累加，消除中间全局内存写入：

```
单一 kernel：计算 d(p_i, S_k) → 立即累加 w_i c_i → 输出 T(S_k)
└── 内存流量: 仅读取 P，直接写出 T(S_k)
```

**理论内存流量对比**：

设点云数据量 $M_P$（字节），关联结果大小 $M_{\text{assoc}} \approx \bar{c} \cdot K \cdot (\text{索引大小})$，纹理输出 $M_T = K \cdot 3 \cdot \text{sizeof(float)}$：

$$M_{\text{sep}} = M_P + M_{\text{assoc}} + M_{\text{assoc}} + M_T = M_P + 2M_{\text{assoc}} + M_T$$
$$M_{\text{fused}} = M_P + M_T$$

融合减少了 $2M_{\text{assoc}}$ 的内存流量。在大规模场景（$\bar{c}$ 较大）时，$M_{\text{assoc}}$ 可能超过 $M_P$，融合的收益尤为显著。

---

## 7. 性能理论分析

### 7.1 总体加速比公式

论文给出的理论加速比（第4节）：

$$\text{Speedup} \approx \frac{M_{\text{sep}}}{M_{\text{fused}}} \cdot \frac{B_{\text{eff}}}{B_{\text{base}}}$$

其中：
- $M_{\text{sep}} / M_{\text{fused}}$：内存流量减少比（由核融合带来）
- $B_{\text{eff}} / B_{\text{base}}$：有效带宽提升比（由 SoA 布局和 Morton 排序带来）

### 7.2 计算瓶颈分析

整体时间：

$$T_{\text{Scan2BIM}} = T_{\text{prep}} + T_{\text{index}} + T_{\text{assoc}} + T_{\text{proj}}$$

BIM 感知优化主要作用于 $T_{\text{index}}$ 和 $T_{\text{assoc}}$：

$$T_{\text{index}}^{\text{bimaware}} = O\!\left(\frac{N \cdot \text{selected\_ratio}}{B_{\text{GPU}}} + K \cdot 27\right) \ll T_{\text{index}}^{\text{cpu}} = O\!\left(\frac{N \log N}{B_{\text{CPU}}}\right)$$

$$T_{\text{assoc}}^{\text{bimaware}} = O\!\left(\frac{K \cdot \bar{c}_{\text{bimaware}}}{B_{\text{GPU}}}\right), \quad \bar{c}_{\text{bimaware}} \ll \bar{c}_{\text{generic}}$$

### 7.3 五个系统优化维度的作用分解

| 优化手段 | 作用阶段 | 效果指标 |
|---|---|---|
| SoA 内存布局 | 全部 | $B_{\text{eff}}$ 提升，减少内存事务数 |
| Morton 排序 | index + assoc | $\bar{J}$ 降低，coalescing_proxy 上升 |
| BIM 感知候选压缩 | index + assoc | selected_ratio 降低，$\bar{c}$ 减小 |
| 核融合 | assoc + proj | $M_{\text{assoc}}$ 消除，带宽节省 |
| Warp 级归约 | assoc + proj | warp divergence 减少，GPU 利用率上升 |

**消融实验结论**（论文第5节）：核融合和表面中心索引贡献最大。

### 7.4 实验结果

**数据集**：SLABIM（HKUST 建筑室内，5层）+ 3DCityDB 城市级 BIM

| 对比方法 | 加速比 vs 此方法 |
|---|---|
| 多线程 CPU 基线 | **14.8×** 更慢 |
| 通用 GPU 方法（不带 BIM 感知） | **5.3×** 更慢 |

**精度**：颜色偏差误差（Color Deviation Error）$< 0.015$，与基线方法相当。

**可扩展性**：运行时间随点云规模**近似线性增长**（$O(N \cdot \text{selected\_ratio})$），而 CPU 方法接近 $O(N^{1.5} \sim N \log N)$。

---

## 8. 实验设计与评估指标

### 8.1 数据集与变量

**点云规模**：$N \in [10^5, 10^8]$，跨越 3 个数量级验证可扩展性。

**BIM 表面数**：$K \in [768, 8192]$（代码中 `surface_count` 参数）。

**体素大小** $s$：控制空间分辨率与内存之间的权衡：
- $s$ 小 → 体素数多 → 索引精确 → 内存增大
- $s$ 大 → 体素数少 → 候选污染增加 → 精度下降

代码默认 $s = 0.2$ m，对应建筑场景中约 1-5 个激光点/体素的密度。

### 8.2 颜色偏差误差（Color Deviation Error）

论文用于量化纹理精度的指标（`parse_results.py`）：

$$\text{ColorDev}_i = \|c_i - \bar{c}_{\text{dataset}}\|_2 = \sqrt{(r_i - \bar{r})^2 + (g_i - \bar{g})^2 + (b_i - \bar{b})^2}$$

其中 $\bar{c}_{\text{dataset}}$ 为数据集级别的平均 RGB 颜色。实验结果 $\text{ColorDev} < 0.015$（以 [0,1] 归一化 RGB 计量），与 CPU 基线方法相差不超过 1.5%。

### 8.3 运行时分位数统计

代码 `summarize_profile.py` 中对重复运行（3次 measured + 1次 warmup）的时延统计：

$$\hat{T}_{q} = \text{quantile}_{q}(T_1, T_2, T_3), \quad q \in \{0.50, 0.95, 0.99\}$$

使用线性插值分位数（`percentile` 函数）。报告 P50/P95/P99 而非均值，原因：GPU 时延分布有重尾（偶发性内存页故障、OS 调度），分位数比均值更鲁棒。

---

## 9. 从 BIM 到医学数字孪生的迁移逻辑

本节揭示论文方法论与 `MDBIMDT` 项目的数学同构关系，这是本研究的**核心迁移假设**。

### 9.1 两个域的结构同构

| 概念 | BIM 纹理映射 | 医学数字孪生 |
|---|---|---|
| **结构元素集合** $\mathcal{S}$ | BIM 表面（墙/柱/门/楼板） | 解剖结构（器官/病灶/血管） |
| **观测流** $P$ | 点云（坐标+颜色） | CT/MRI 体数据（坐标+强度） |
| **关联算子** $\mathcal{O}_{\text{assoc}}$ | 点找最近表面 | 体素找最近解剖结构 |
| **投影算子** $\mathcal{O}_{\text{proj}}$ | 加权颜色聚合 $\to T(S_k)$ | 加权强度聚合 $\to$ 分割/状态更新 |
| **索引结构** | 27 邻域体素 | ROI 局部膨胀（26 邻域体素） |
| **结构先验** | BIM 表面位置已知 | 器官 Mask $M_{t_0}$ 已知 |
| **候选压缩** | selected_ratio = 8.4% | roi_ratio = 47-72% |

### 9.2 统一问题形式化

**BIM 纹理映射（离线）**：给定 $(P, \mathcal{S})$，一次性计算所有 $T(S_k)$。

**医学数字孪生（在线增量）**：在时刻序列 $t_0, t_1, t_2, \ldots$ 上，给定上一时刻状态 $\mathbf{s}_k(t-1)$ 和新观测 $P_t$，计算：

$$\mathbf{s}_k(t) = \mathcal{O}_{\text{proj}}^{\text{inc}}\!\bigl(\mathcal{O}_{\text{assoc}}\!\bigl(\mathcal{O}_{\text{index}}^{\text{local}}(P_t, \mathbf{s}_k(t-1))\bigr)\bigr)$$

与 BIM 方法的区别仅在于 $\mathcal{O}_{\text{index}}^{\text{local}}$：不是针对全局点云建立索引，而是**以 $\mathbf{s}_k(t-1)$（即 $\hat{M}_{t_0}$）为中心建立局部 ROI 索引**——这正是 Stage-3 增量分割的核心逻辑。

### 9.3 关键假设的传承

| 假设 | BIM 场景 | 医学场景 |
|---|---|---|
| **假设 A**（结构先验存在） | BIM 构件位置已知 → AABB 可构建 | 前一帧器官 Mask 已知 → ROI 可构建 |
| **假设 B**（更新局部连续） | 施工仅影响局部构件 | 病灶漂移 ≤ 6mm → 邻域范围有限 |
| **假设 C**（结构单元并行） | 每个 BIM 表面独立更新 | 每个 Patch 节点独立更新 |

**假设 B 的量化验证**（来自 Stage-2）：

$$\text{drift\_mm} = \|\bar{\mathbf{p}}_{t_1} - \bar{\mathbf{p}}_{t_0}\|_2 \in [1.66, 6.12] \text{ mm}$$

相对于器官直径（脾脏约 100-150 mm），漂移量仅占 1-6%，印证了局部更新的合理性。

### 9.4 候选压缩比的理论差异

BIM 的 selected_ratio（8.4%）远低于医学的 roi_ratio（47-72%），原因在于：

$$\frac{\text{BIM selected\_ratio}}{\text{Medical roi\_ratio}} \approx \frac{K \cdot (3s)^3}{V_P} \bigg/ \frac{|M_{t_0} \oplus B_1^{(m)}|}{|D|}$$

- BIM 中，$K = 1024$ 个表面代理点覆盖的 $27Ks^3$ 体积远小于总扫描体积 $V_P$
- 医学中，脾脏体积本身就占 CT 裁剪体的 20-50%，膨胀后 ROI 自然更大

这说明**BIM 场景的候选压缩收益更极端**，而医学场景需要发展更强的局部化策略（如 patch 级并行）才能达到类似压缩比。

---

## 10. 代码与论文对应速查

### 四阶段管线

| 论文算子 | 代码函数 | 所在文件 |
|---|---|---|
| $\mathcal{O}_{\text{index}}$（整体） | `stage_index()` | `scan2bim_stage_runner.py:332` |
| 体素坐标计算 | `np.floor(x/voxel_size)` | `scan2bim_stage_runner.py:208-211` |
| 体素键打包 | `pack_voxel_keys()` | `scan2bim_stage_runner.py:77-83` |
| Morton 码计算 | `compute_morton_codes()` | `scan2bim_stage_runner.py:97-109` |
| 比特扩展（expand） | `_expand_bits21()` | `scan2bim_stage_runner.py:86-94` |
| Morton 排序 | `apply_morton_sort()` | `scan2bim_stage_runner.py:145-158` |
| BIM 感知候选压缩 | `stage_index_cpu(..., bimaware=True)` | `scan2bim_stage_runner.py:213-247` |
| 27 邻域体素枚举 | `for dx in (-1,0,1):...` | `scan2bim_stage_runner.py:222-227` |
| 前缀和/边界计算 | `np.unique(..., return_index=True)` | `scan2bim_stage_runner.py:425` |
| $\mathcal{O}_{\text{assoc}}$ | `stage_assoc()` | `scan2bim_stage_runner.py:406-470` |
| 候选数量统计 | `_counts_for_surface_keys()` | `scan2bim_stage_runner.py:392-403` |
| 候选不规则性 $\sigma_c/\bar{c}$ | `irregularity = std_c / (mean_c + 1e-6)` | `scan2bim_stage_runner.py:455` |
| 访存 jump 统计 | `_jump_stats()` | `scan2bim_stage_runner.py:130-142` |
| coalescing_proxy $1/(1+\bar{J})$ | `coalescing = 1.0 / (1.0 + mean_jump)` | `scan2bim_stage_runner.py:137` |
| $\mathcal{O}_{\text{proj}}$（简化） | `stage_proj()` | `scan2bim_stage_runner.py:473-499` |
| Gaussian 权重 $e^{-d^2/\sigma^2}$ | `w = np.exp(-candidate_counts / ...)` | `scan2bim_stage_runner.py:486` |
| 加权颜色聚合 | `(w * r).sum() / (w.sum() + 1e-6)` | `scan2bim_stage_runner.py:487-489` |

### 关键公式与实验指标

| 论文公式 | 代码变量 | 实测值 |
|---|---|---|
| $\text{selected\_ratio} = |\mathcal{A}|/N$ | `selected_ratio` | 0.0838（BIM感知），1.0（通用） |
| $\bar{c} = \frac{1}{K}\sum_k \|N(S_k)\|$ | `avg_candidates` | 1.035（所有变体） |
| $\text{irregularity} = \sigma_c / \bar{c}$ | `candidate_irregularity` | 记录在 metrics.json |
| $\bar{J}$ = index jump mean | `index_order_jump_mean` | morton 低，gpu_generic 高 |
| $\text{Speedup}_{\text{CPU}}$ | 论文报告 | **14.8×** |
| $\text{Speedup}_{\text{GPU generic}}$ | 论文报告 | **5.3×** |
| ColorDev $< 0.015$ | `color_deviation_error` | 在 summary.csv |

---

## 附：完整数据流与时间分解图

```
输入: P = {(x,y,z,r,g,b)}_{i=1}^N      (SoA 布局，PTX 格式)
      S = {S_k}_{k=1}^K                 (BIM 表面元素集合)

───────────────────────────────────────────────────────────────────────
Stage PREP  [T_prep]
  1. 从 ZIP 加载 PTX 点云（SoA 布局）
  2. [若 morton_ordered] 按 Morton 码排序，降低后续访存 jump
  输出: points.npz (x, y, z, r, g, b)

───────────────────────────────────────────────────────────────────────
Stage INDEX  [T_index]  ← 论文主要优化点之一
  1. 体素化: v(p_i) = floor(p_i / s)
  2. 键打包: key = pack_voxel_keys(vx, vy, vz)   [21+21+21 = 63 bits]
  3. [若 morton_ordered] 已排序 → 直接使用
  4. [若 gpu_generic]  selected_mask = ones(N)   → all points
  5. [若 gpu_bimaware] 计算 surface 代理点的27邻域 active_keys
                       selected_mask = isin(keys, active_keys)
                       → selected_ratio = 0.0838

  输出: index.npz (voxel_keys, selected_idx, surface_idx, uniq_keys, counts)
  关键指标: selected_ratio, index_order_jump_mean, coalescing_proxy

───────────────────────────────────────────────────────────────────────
Stage ASSOC  [T_assoc]  ← 论文主要优化点之二
  1. 对每个表面 S_k，查找其体素键对应的点集 N(S_k)
  2. 统计 |N(S_k)| → candidate_counts
  3. 计算不规则性 CV = σ_c / μ_c → warp divergence 代理

  输出: assoc.npz (surface_idx, candidate_counts)
  关键指标: avg_candidates, candidate_irregularity, assoc_memory_jump

───────────────────────────────────────────────────────────────────────
Stage PROJ  [T_proj]  ← 核融合消除此阶段独立存在（理想情况）
  1. 对每个表面: w_i = exp(-d²/σ²)
  2. T(S_k) = Σ w_i c_i / Σ w_i

  输出: proj_mean_r/g/b（纹理值）
  关键指标: ColorDev = ||T(S_k) - T̄||₂ < 0.015

───────────────────────────────────────────────────────────────────────
总时延: T_Scan2BIM = T_prep + T_index + T_assoc + T_proj
加速比: 14.8× vs CPU，5.3× vs generic GPU
```

---

*本文档生成于 2026-02-25，基于 `papers/latex/` 论文正文与 `experiments/scripts/scan2bim_stage_runner.py` 实现代码的完整对应分析。*
