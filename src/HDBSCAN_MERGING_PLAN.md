# 6D 相空间 HDBSCAN 冷流合并方案

## 1. 背景与动机

### 1.1 当前管线

当前冷流识别管线 (`adaptive_pipeline.py`) 包含以下阶段：

| 阶段 | 功能 | 输出规模 |
|------|------|----------|
| Phase 0 | 空间过滤 + 物理条件过滤 | ~1e4 粒子 |
| Phase 1 | Delaunay 拓扑扫描 + plateau 自适应选参 | ~1e3 碎片组件 |
| Phase 2 | KDTree 空间质心合并 + plateau 自适应选参 | ~1e1-1e2 组件 |

### 1.2 问题

Phase 2 仅基于 3D 空间质心距离做合并。属于同一原始冷流的碎片在潮汐拉伸后可能在空间上已分散，但在 6D 相空间（位置+速度）中仍然聚集。纯空间距离无法正确归并这些碎片，导致最终识别出的是**破碎后的冷流片段**而非**原始冷流**。

### 1.3 目标

在现有识别方案不变的前提下，利用 6D 相空间 HDBSCAN 聚类，将 ~1e3 量级的破碎冷流合并为 ~1 量级的原始冷流结构。

---

## 2. 方案总览

### 2.1 核心策略

1. **保留 Phase 0 和 Phase 1 完全不变**：Delaunay 拓扑扫描 + plateau 自适应选参照常执行
2. **改造 Phase 2**：KDTree 合并改为**固定参数单次执行**，去掉 plateau 扫描，作为粗筛降低 HDBSCAN 输入规模
3. **新增 Phase 3**：对 Phase 2 输出的碎片，在 6D 相空间上用 HDBSCAN 做密度聚类，通过粗扫+细扫的 plateau 策略自适应选取最优 `cluster_selection_epsilon`，合并属于同一原始冷流的碎片

### 2.2 修改后管线

```
Phase 0: 空间 + 物理过滤（不变）
    ↓
Phase 1: Delaunay 拓扑自适应扫描（不变）
    ↓  ~1e3 碎片
Phase 2: KDTree 固定参数单次合并（改造：去掉 plateau 扫描）
    ↓  ~1e2 碎片
Phase 3: 6D 相空间 HDBSCAN + plateau 自适应 epsilon（新增）
    ↓  ~1-5 个原始冷流
Final:  数据输出（不变）
```

### 2.3 设计理由

- **KDTree 作为预筛**：KDTree 合并是 O(N log N)，极快。先做一次 KDTree 把明显空间邻近的碎片合并掉，HDBSCAN 的输入从 ~1e3 降到 ~1e2，计算量降低一个量级。
- **HDBSCAN 作为精合并**：HDBSCAN 复杂度约 O(N log N) 到 O(N²)，对 ~1e2 量级输入非常快，且能在 6D 相空间中发现空间分散但速度相干的碎片簇。
- **KDTree 固定参数**：不再做 plateau 扫描，因为 KDTree 只是粗筛，不需要最优参数。使用经验固定值即可。
- **Phase 3 自适应 epsilon**：`cluster_selection_epsilon` 控制簇合并粒度，是 Phase 3 唯一敏感参数。沿用 Phase 1/2 的"粗扫+细扫+plateau 选稳"策略，在有效组件数 ≤10 的收敛区内寻找最稳定的 epsilon，确保合并结果不敏感于参数微扰。

---

## 3. Phase 2 改造细节

### 3.1 改动前（`adaptive_pipeline.py` L147-183）

Phase 2 包含粗扫描 (6 个点) + 细扫描 (8 个点) 共 14 次 `merge_fragmented_streams` 调用，通过 `find_plateau_1d` 选取最优 `merge_fraction`。

### 3.2 改动后

```python
# Phase 2: Merging (fixed-parameter single pass)
MERGE_FRACTION_FIXED = 0.02  # 固定参数

identity, count = api.merge_fragmented_streams(
    gas_data, subhalo_info, identity, count, BoxSize,
    merge_fraction=MERGE_FRACTION_FIXED
)

if count <= 1:
    print(f"[{subhalo_id}] Phase 2 count <= 1, short-circuit to IO.")
    return api.extract_stream_properties(...)
```

- **去掉**：`merge_fracs_coarse` 粗扫描 + `merge_fracs_fine` 细扫描
- **去掉**：`find_plateau_1d` 调用
- **保留**：`api.merge_fragmented_streams` 单次调用，`merge_fraction` 设为固定值

### 3.3 固定参数选择

`MERGE_FRACTION_FIXED = 0.02`，即 `0.02 * R200c` 作为合并距离阈值。该值对应于原有 plateau 扫描的中间区域，是一个保守但有效的粗筛参数。

---

## 4. Phase 3 新增实现细节

### 4.1 内部函数签名

在 `API_methods.py` 中新增三个函数：

#### 4.1.1 6D 特征构建（公共子程序）

```python
def _build_hdbscan_features(gas_data, identity, count, BoxSize):
    """
    Build 6D phase-space feature matrix for stream fragments.
    Returns (features, valid_indices, local_identities) tuple.
    """
    valid_mask = identity >= 0
    valid_indices = np.where(valid_mask)[0]
    pos = gas_data['Coordinates'][valid_indices]      # (N, 3)
    vel = gas_data['PhysicalVelocity'][valid_indices]  # (N, 3)
    mass = gas_data['Masses'][valid_indices]           # (N,)
    local_identities = identity[valid_indices]         # (N,)

    features = np.zeros((count, 6))
    for i in range(count):
        member = local_identities == i
        # Position centroid with periodic boundary handling
        member_pos = pos[member]
        ref = member_pos[0]
        dx = periodic_displacement(member_pos, ref, BoxSize)
        features[i, :3] = (ref + dx.mean(axis=0)) % BoxSize
        # Velocity centroid: mass-weighted average
        features[i, 3:] = np.average(vel[member], axis=0, weights=mass[member])

    return features, valid_indices, local_identities
```

#### 4.1.2 特征标准化（公共子程序）

```python
def _scale_features(features, subhalo_info, v_vir=None):
    """
    Normalize position by R200c, velocity by virial velocity.
    Returns (scaled_features, v_vir) tuple.
    """
    features_scaled = features.copy()
    features_scaled[:, :3] /= subhalo_info['r200c']

    if v_vir is None:
        v_vir = calculate_virial_velocity(
            subhalo_info['m_vir'],
            subhalo_info['r200c'],
            subhalo_info['a']
        )
    features_scaled[:, 3:] /= v_vir
    return features_scaled, v_vir
```

#### 4.1.3 单点 HDBSCAN 执行 + 重映射（公共子程序）

```python
def _run_hdbscan_at_epsilon(features_scaled, count, epsilon, min_cluster_size=3, min_samples=1):
    """
    Run HDBSCAN at a given epsilon, return (identity_update, new_count).
    Returns (new_local_labels, n_components) for the valid-index subset.
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(features_scaled)

    if count <= 1:
        return np.arange(count, dtype=np.int32), count

    # Relabel: noise points keep their original IDs, clusters get new sequential IDs
    new_labels = np.zeros(count, dtype=np.int32)
    next_label = 0

    noise_components = np.where(labels == -1)[0]
    for comp_id in noise_components:
        new_labels[comp_id] = next_label
        next_label += 1

    cluster_ids = sorted([l for l in set(labels) if l != -1])
    for cluster_id in cluster_ids:
        member_components = np.where(labels == cluster_id)[0]
        for comp_id in member_components:
            new_labels[comp_id] = next_label
        next_label += 1

    n_components = next_label
    return new_labels, n_components
```

### 4.2 自适应 epsilon 搜索

#### 4.2.1 核心逻辑

沿用 `find_plateau_1d` 的"粗扫→细扫→plateau 选稳"策略，但目标输出不再是组件数量本身，而是**聚类状态稳定性**。具体流程：

1. **粗扫**：在一组预定义的 epsilon 候选值上执行 HDBSCAN，记录每个 epsilon 对应的有效组件数
2. **收敛区筛选**：保留组件数 ≤10 的 epsilon 候选点（认为此时合并程度已达到物理合理的范围）
3. **plateau 选稳**：在收敛区候选点上调用 `find_plateau_1d`，选取组件数最稳定的 epsilon
4. **细扫**：以粗扫最优值为中心，在 ±20% 范围内取更密的候选点，重复步骤 1-3
5. **固化工**：用最终选定的 epsilon 执行一次 HDBSCAN 并返回结果

```python
def find_optimal_epsilon(
    features_scaled, count,
    min_cluster_size=3, min_samples=1,
):
    """
    Coarse-to-fine epsilon scan with plateau selection.
    Only considers epsilons where the resulting component count ≤ 10.
    Returns (best_epsilon, best_labels, n_components).
    """
    # Coarse scan: 8 points across the full range
    eps_coarse = np.linspace(0.02, 0.50, 8)
    counts_coarse = []
    labels_list_coarse = []
    for eps in eps_coarse:
        lbls, cnt = _run_hdbscan_at_epsilon(
            features_scaled, count, eps, min_cluster_size, min_samples
        )
        counts_coarse.append(cnt)
        labels_list_coarse.append(lbls)

    # Filter to convergence zone: count <= 10
    conv_mask = [c <= 10 for c in counts_coarse]
    if not any(conv_mask):
        # No epsilon achieves convergence; fall back to the one with minimum count
        best_idx = int(np.argmin(counts_coarse))
        print(f"  [epsilon scan] No convergence (count<=10) found; using min-count epsilon={eps_coarse[best_idx]:.3f} (count={counts_coarse[best_idx]})")
        return eps_coarse[best_idx], labels_list_coarse[best_idx], counts_coarse[best_idx]

    # Apply plateau selection within convergence zone
    eps_conv = eps_coarse[conv_mask]
    counts_conv = [c for c, m in zip(counts_coarse, conv_mask) if m]
    labels_conv = [l for l, m in zip(labels_list_coarse, conv_mask) if m]

    best_eps = find_plateau_1d(eps_conv, counts_conv)
    if best_eps is None:
        best_eps = float(eps_conv[len(eps_conv) // 2])

    # Fine scan: 10 points within ±20% of coarse best, clamped to [0.02, 0.50]
    lo = max(0.02, best_eps * 0.80)
    hi = min(0.50, best_eps * 1.20)
    eps_fine = np.linspace(lo, hi, 10)
    counts_fine = []
    labels_fine = []
    for eps in eps_fine:
        lbls, cnt = _run_hdbscan_at_epsilon(
            features_scaled, count, eps, min_cluster_size, min_samples
        )
        counts_fine.append(cnt)
        labels_fine.append(lbls)

    # Filter fine scan to convergence zone
    conv_mask_fine = [c <= 10 for c in counts_fine]
    if any(conv_mask_fine):
        eps_fine_conv = eps_fine[conv_mask_fine]
        counts_fine_conv = [c for c, m in zip(counts_fine, conv_mask_fine) if m]
        labels_fine_conv = [l for l, m in zip(labels_fine, conv_mask_fine) if m]

        best_eps_fine = find_plateau_1d(eps_fine_conv, counts_fine_conv)
        if best_eps_fine is not None:
            best_eps = best_eps_fine
            # Find the corresponding labels
            idx = np.argmin(np.abs(eps_fine_conv - best_eps_fine))
            best_labels = labels_fine_conv[idx]
            best_count = counts_fine_conv[idx]
            print(f"  [epsilon scan] Locked epsilon={best_eps:.4f} (count={best_count})")
            return best_eps, best_labels, best_count
        else:
            best_idx = int(np.argmin(counts_fine_conv))
            best_labels = labels_fine_conv[best_idx]
            best_count = counts_fine_conv[best_idx]
    else:
        best_idx = int(np.argmin(counts_fine))
        best_labels = labels_fine[best_idx]
        best_count = counts_fine[best_idx]

    print(f"  [epsilon scan] No fine plateau; using min-count epsilon={eps_fine[best_idx]:.3f} (count={best_count})")
    return eps_fine[best_idx], best_labels, best_count
```

### 4.3 主函数：merge_streams_hdbscan

组装上述子程序，作为 `adaptive_pipeline.py` 的调用入口：

```python
def merge_streams_hdbscan(
    gas_data,
    subhalo_info,
    identity,
    count,
    BoxSize,
    v_vir=None,
    min_cluster_size=3,
    min_samples=1,
):
    """
    Merge stream fragments using HDBSCAN in 6D phase space
    with adaptive epsilon selection via plateau scanning.

    Parameters
    ----------
    gas_data : dict
        Gas particle data from cutout file.
    subhalo_info : dict
        Subhalo metadata including center_pos, center_vel, m_vir, r200c, a.
    identity : ndarray
        Array mapping particle index to stream component ID. -1 = not a cold stream cell.
    count : int
        Number of stream components (max component ID + 1).
    BoxSize : float
        Simulation box size (kpc/h) — used for periodic boundary handling only.
    v_vir : float, optional
        Virial velocity for velocity scaling. Computed if not provided.
    min_cluster_size : int
        Minimum number of fragments to form a cluster in HDBSCAN.
    min_samples : int
        Minimum samples for core point determination in HDBSCAN.

    Returns
    -------
    identity : ndarray
        Updated array with merged component labels.
    count : int
        New number of stream components after merging.
    """
    import hdbscan  # check import

    if count <= 1:
        return identity, count

    print(f"[{subhalo_info['subhalo_id']}] Phase 3: HDBSCAN adaptive epsilon scan, initial components: {count}")

    # Step 1: Build features
    features, valid_indices, local_identities = _build_hdbscan_features(
        gas_data, identity, count, BoxSize
    )

    # Step 2: Scale features
    features_scaled, v_vir = _scale_features(features, subhalo_info, v_vir)

    # Step 3: Adaptive epsilon scan
    best_eps, best_labels, n_components = find_optimal_epsilon(
        features_scaled, count, min_cluster_size, min_samples
    )

    # Step 4: Remap identity
    identity[valid_indices] = best_labels[local_identities]

    print(f"[{subhalo_info['subhalo_id']}] Phase 3 complete: {count} fragments merged into {n_components} structures (epsilon={best_eps:.4f})")

    return identity, n_components
```

### 4.4 参数说明

| 参数 | 默认值 | 含义 | 调节方向 |
|------|--------|------|----------|
| `min_cluster_size` | 3 | 最少碎片数才构成一个簇 | 增大 → 更保守，更少合并 |
| `min_samples` | 1 | 核心点判定所需最小样本数 | 1 = 最宽松 |
| `epsilon` 搜索范围 | [0.02, 0.50] | 粗扫范围，缩放空间中的距离阈值 | 收敛区筛选自动过滤过大/过小值 |
| 收敛阈值 | 10 | 组件数 ≤ 此值才进入 plateau 候选 | 增大 → 考虑更宽松的合并状态 |

---

## 5. 文件修改清单

### 5.1 `src/API_methods.py`

新增内容：
- `merge_streams_hdbscan()` 函数（~80 行）
- 在文件顶部 import `hdbscan`（需要 try/except 处理）

```python
try:
    import hdbscan
except ImportError:
    hdbscan = None
```

在 `merge_streams_hdbscan` 函数开头检查：

```python
if hdbscan is None:
    raise ImportError("hdbscan is required. Install with: pip install hdbscan")
```

### 5.2 `src/adaptive_pipeline.py`

改动内容：
1. Phase 2 改为固定参数单次执行（替换 L147-183 的 plateau 扫描代码，约 30 行 → 约 8 行）
2. 新增 Phase 3 块（约 10 行），调用自适应 epsilon 扫描
3. 新增顶部配置常量：

```python
# Phase 2 fixed parameter
MERGE_FRACTION_FIXED = 0.02

# Phase 3 HDBSCAN parameters
HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 1

# Phase 3 epsilon scan parameters
EPS_COARSE_MIN = 0.02
EPS_COARSE_MAX = 0.50
EPS_COARSE_N = 8
EPS_FINE_N = 10
EPS_FINE_FRAC = 0.20  # ±20% around coarse best
CONVERGENCE_THRESHOLD = 10  # component count <= this enters plateau candidate zone
```

### 5.3 `src/batch_run.py`

无需修改。

---

## 6. 依赖管理

新增 Python 依赖：

```
hdbscan>=0.8.33
```

安装：
```bash
pip install hdbscan
```

`hdbscan` 的依赖：
- `numpy` (已安装)
- `scipy` (已安装)
- `scikit-learn` (可能已有)
- `cython` (构建依赖)

---

## 7. 预期效果

### 7.1 数量级变化

```
Phase 1 后: ~1e3 碎片组件
Phase 2 后: ~1e2 碎片组件 (KDTree 粗筛，去掉 80-90% 明显邻近碎片)
Phase 3 后: ~1-5 个原始冷流 (HDBSCAN 相空间精合并)
```

### 7.2 性能评估

- **KDTree 粗筛**：O(N log N)，N~1e3，耗时 < 0.1s
- **Phase 3 epsilon 粗扫**：8 次 HDBSCAN，N~1e2，耗时 ~8-40s
- **Phase 3 epsilon 细扫**：10 次 HDBSCAN，N~1e2，耗时 ~10-50s
- **总计**：Phase 3 自适应扫描约 20-90s（取决于 N 和收敛区大小）
- 如不需要自适应，固定 epsilon 单次执行 ~1-5s

### 7.3 物理意义

- 空间分散但速度相干的冷流碎片会被正确归并
- 速度不相关的空间邻近碎片不会被错误合并（HDBSCAN 的速度维度提供了区分能力）
- 噪声点（孤立碎片）保持独立，不影响主流结构

---

## 8. 风险与注意事项

### 8.1 HDBSCAN 在 6D 空间的距离度量

欧氏距离在 6D 中可能不如马氏距离鲁棒，但 HDBSCAN 对距离度量的选择相对宽容。如效果不佳，可尝试：
- 使用 `metric='manhattan'`
- 对位置/速度维度使用不同权重（非 1:1 归一化）

### 8.2 周期性边界条件

位置质心通过 `periodic_displacement()` 以组内首个点为参考点计算最短路径位移，再求均值加回参考点，正确处理跨越模拟盒边界的组件。

### 8.3 HDBSCAN 版本兼容性

`cluster_selection_epsilon` 参数在较新版本的 hdbscan 中引入。如果服务器上的版本较旧，可能需要升级：
```bash
pip install --upgrade hdbscan
```

### 8.4 收敛区可能为空

当所有 epsilon 候选点的组件数都 >10 时（极端情况：碎片极其分散，速度也不相干），`find_optimal_epsilon` 回退到取最小组件数对应的 epsilon。此时输出日志会标注"未找到收敛区"，便于后续调整搜索范围或收敛阈值。

### 8.5 性能与精度权衡

自适应扫描需执行 18 次 HDBSCAN（8 粗 + 10 细）。如批量处理大量子晕时耗时不可接受，可改为固定 epsilon（单次执行 ~1-5s）。建议调试阶段用自适应，生产阶段用固定参数。

Phase 3 后的 `count` 将作为最终输出值传递给 `extract_stream_properties`。需确保 HDF5 输出文件的 `count` 字段反映的是 HDBSCAN 合并后的数量。
