import numpy as np
from scipy import signal
from scipy import ndimage
import itertools
import argparse
import os
import time

# 导入底层核心计算库
import API_methods as api

# =========================================================================
# Utils: 平台搜索算法 (Platform Search Algorithms)
# =========================================================================

def find_plateau_1d(x_values, y_values):
    """
    一维平台搜索算法。
    
    逻辑:
    1. 应用中值滤波 (scipy.signal.medfilt) 抹平离散的数值毛刺。
    2. 使用 itertools.groupby 寻找数值相同的连续区间。
    3. 返回最长连续区间的中点 x 值作为最稳定的参数选择。
    """
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    
    if len(x_values) == 0:
        return None
    if len(x_values) < 3:
        return float(x_values[len(x_values) // 2])
        
    # 1. 应用中值滤波 (窗口大小为3)，抹平由于离散网格造成的单点毛刺
    y_filtered = signal.medfilt(y_values, kernel_size=3)
    
    # 2. 寻找数值相同的连续区间
    segments = []
    start_idx = 0
    for k, g in itertools.groupby(y_filtered):
        length = len(list(g))
        segments.append((start_idx, length))
        start_idx += length
        
    # 获取最长区间的长度
    max_len = max([seg[1] for seg in segments])
    
    # 筛选出所有达到最大长度的区间
    candidates = [seg for seg in segments if seg[1] == max_len]
    
    # 3. 若存在多个相同最大长度的区间，优先选择最靠近参数空间中部的区间
    center_idx = len(x_values) / 2.0
    best_segment = min(candidates, key=lambda seg: abs((seg[0] + seg[1] / 2.0) - center_idx))
    
    # 取最佳区间的中点索引
    best_mid_idx = int(best_segment[0] + best_segment[1] // 2)
    
    return float(x_values[best_mid_idx])

def find_plateau_2d(X_grid, Y_grid, Z_counts):
    """
    二维平台搜索算法。
    
    逻辑:
    1. 应用轻量级二维中值滤波平滑物理孤立噪点，防止平台被切碎。
    2. 寻找二维空间中 Z_counts 最大的连通域 (scipy.ndimage.label)。
    3. 对该最大连通域使用距离变换 (scipy.ndimage.distance_transform_edt)。
    4. 寻找距离边界最远的像素点（即安全中心），将其映射回 (X, Y) 坐标。
    """
    Z_counts = np.asarray(Z_counts)
    
    # 容错：如果全为空或全为0，直接返回网格中心
    if Z_counts.size == 0 or np.all(Z_counts == 0):
        mid_y, mid_x = Z_counts.shape[0] // 2, Z_counts.shape[1] // 2
        return float(X_grid[mid_y, mid_x]), float(Y_grid[mid_y, mid_x])
        
    # [优化] 使用中值滤波抹平微小的数值波动，提升形态学参数搜索鲁棒性
    Z_smoothed = ndimage.median_filter(Z_counts, size=3)
    
    # 提取所有大于0的独立 count 值
    unique_counts = np.unique(Z_smoothed)
    unique_counts = unique_counts[unique_counts > 0]
    
    max_area = 0
    best_label_mask = None
    
    # 1. 遍历所有非零 count 值，寻找面积最大的连续等值“岛屿”
    for val in unique_counts:
        # 生成当前 count 值的二值化掩膜
        binary_mask = (Z_smoothed == val)
        
        # 连通域分析
        labeled_array, num_features = ndimage.label(binary_mask)
        
        # 统计最大的子连通域面积
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            area = np.sum(component_mask)
            
            if area > max_area:
                max_area = area
                best_label_mask = component_mask
                
    if best_label_mask is None:
        mid_y, mid_x = Z_counts.shape[0] // 2, Z_counts.shape[1] // 2
        return float(X_grid[mid_y, mid_x]), float(Y_grid[mid_y, mid_x])
        
    # 2. 对最大的连通域使用欧氏距离变换 (EDT)
    # 计算岛内每个像素到最近背景(边界)的距离
    distance_map = ndimage.distance_transform_edt(best_label_mask)
    
    # 3. 寻找距离边界最远的安全中心 (即 distance_map 上的最大值点)
    center_y, center_x = np.unravel_index(np.argmax(distance_map), distance_map.shape)
    
    best_X = X_grid[center_y, center_x]
    best_Y = Y_grid[center_y, center_x]
    
    return float(best_X), float(best_Y)

# =========================================================================
# Pipeline Entrance: 主流水线装配
# =========================================================================

def adaptive_cold_stream_pipeline(subhalo_id, snapNum, catalogs, cutout_dir, output_dir):
    """
    序贯多尺度自适应冷流识别主控流水线。
    包含预处理、第一阶段（拓扑扫描）、第二阶段（合并扫描）和第三阶段（形态学扫描），
    最后执行数据落盘。
    """
    print(f"\n{'='*65}")
    print(f"Starting Adaptive Cold Stream Pipeline | Snap: {snapNum} | Subhalo: {subhalo_id}")
    print(f"{'='*65}")
    pipeline_start = time.time()

    # ---------------------------------------------------------------------
    # Phase 0: Pre-processing (数据预处理与全局缓存)
    # ---------------------------------------------------------------------
    # 建立不可变的全局缓存状态，严禁在后续流程中被覆盖
    spatial_mask, gas_data, subhalo_info = api.stream_spatial_mask(
        subhalo_id, snapNum, catalogs, cutout_dir=cutout_dir
    )
    
    if spatial_mask is None:
        print(f"[{subhalo_id}] Pipeline terminated: invalid spatial mask or missing data")
        return {}, {}

    BoxSize = catalogs['BoxSize']
    final_valid_mask = api.generate_physical_mask(
        gas_data, spatial_mask, subhalo_info, BoxSize
    )

    # ---------------------------------------------------------------------
    # Phase 1: Topology (拓扑容差参数扫描)
    # ---------------------------------------------------------------------
    print("\n--- Phase 1: Topology Scanning ---")
    # 参数空间: edge_tolerance 在 1.10 到 1.50 之间扫描 (例如20个点)
    edge_tols = np.linspace(1.10, 1.50, 20)
    phase1_counts = []
    
    for tol in edge_tols:
        # 在只读模式下调用底层，记录 count
        _, cnt = api.identify_stream_components(gas_data, final_valid_mask, BoxSize, edge_tolerance=tol)
        phase1_counts.append(cnt)
        
    best_edge_tol = find_plateau_1d(edge_tols, phase1_counts)
    print(f"[{subhalo_id}] Phase 1 Locked Best edge_tolerance: {best_edge_tol:.3f}")
    
    # 固化状态: 使用最佳参数执行一次真实的计算并保存 identity
    identity, count = api.identify_stream_components(
        gas_data, final_valid_mask, BoxSize, edge_tolerance=best_edge_tol
    )
    
    # 短路拦截
    if count <= 1:
        print(f"[{subhalo_id}] Phase 1 count <= 1, triggering short-circuit to IO.")
        return api.extract_stream_properties(gas_data, subhalo_info, identity, count, catalogs, output_dir, snapNum)

    # ---------------------------------------------------------------------
    # Phase 2: Merging (碎片合并扫描)
    # ---------------------------------------------------------------------
    print("\n--- Phase 2: Merging Scanning ---")
    # 参数空间: merge_fraction 在 0.005 到 0.05 之间扫描 (例如15个点)
    merge_fracs = np.linspace(0.005, 0.05, 15)
    phase2_counts = []
    
    for frac in merge_fracs:
        # 注意：这里必须传入 identity.copy()，防止底层函数原地修改污染基准状态
        _, cnt = api.merge_fragmented_streams(
            gas_data, subhalo_info, identity.copy(), count, BoxSize, merge_fraction=frac
        )
        phase2_counts.append(cnt)
        
    best_merge_frac = find_plateau_1d(merge_fracs, phase2_counts)
    print(f"[{subhalo_id}] Phase 2 Locked Best merge_fraction: {best_merge_frac:.4f}")
    
    # 固化状态
    identity, count = api.merge_fragmented_streams(
        gas_data, subhalo_info, identity, count, BoxSize, merge_fraction=best_merge_frac
    )
    
    # 短路拦截
    if count <= 1:
        print(f"[{subhalo_id}] Phase 2 count <= 1, triggering short-circuit to IO.")
        return api.extract_stream_properties(gas_data, subhalo_info, identity, count, catalogs, output_dir, snapNum)

    # ---------------------------------------------------------------------
    # Phase 3: Morphology (二维形态学过滤扫描) [解耦与性能提升版]
    # ---------------------------------------------------------------------
    print("\n--- Phase 3: Morphology Scanning ---")
    # 构建 thresh_ab (X) 和 thresh_ac (Y) 的 2D 网格 (例如 10x10 的网格)
    ab_vals = np.linspace(1.2, 2.0, 10)
    ac_vals = np.linspace(2.0, 4.0, 10)
    
    # [优化] 显式指定 indexing='ij'，防止非对称网格导致坐标轴翻转
    AB_grid, AC_grid = np.meshgrid(ab_vals, ac_vals, indexing='ij')
    phase3_counts = np.zeros_like(AB_grid, dtype=int)
    
    # [性能核心优化]：循环外预计算所有碎片的PCA形态学比例，消除 O(N^2) 重复计算
    ratios_ab, ratios_ac = api.precompute_morphology_ratios(
        gas_data, subhalo_info, identity, count, BoxSize
    )
    
    for i in range(AB_grid.shape[0]):
        for j in range(AB_grid.shape[1]):
            # 纯粹的轻量级掩码比较，极速算出当前参数下的存活冷流数量
            ab_thresh = AB_grid[i, j]
            ac_thresh = AC_grid[i, j]
            
            surviving_fragments = np.sum((ratios_ab > ab_thresh) & (ratios_ac > ac_thresh))
            phase3_counts[i, j] = surviving_fragments
            
    best_ab, best_ac = find_plateau_2d(AB_grid, AC_grid, phase3_counts)
    print(f"[{subhalo_id}] Phase 3 Locked Best Morphology -> ab: {best_ab:.2f}, ac: {best_ac:.2f}")
    
    # 固化最终状态
    identity, count = api.filter_streams_morphology(
        gas_data, subhalo_info, identity, count, BoxSize, 
        thresh_ab=best_ab, thresh_ac=best_ac
    )

    # ---------------------------------------------------------------------
    # Final Step: 数据落盘 (Logging & IO)
    # ---------------------------------------------------------------------
    print("\n--- Final Step: Data IO & Logging ---")
    objects, props = api.extract_stream_properties(
        gas_data, subhalo_info, identity, count, catalogs, output_dir, snapNum
    )
    
    print(f"\n{'-'*65}")
    print(f"[{subhalo_id}] Adaptive pipeline finished. Total time: {time.time()-pipeline_start:.2f}s")
    print(f"{'='*65}\n")
    
    return objects, props

# =========================================================================
# 脚本执行入口
# =========================================================================
if __name__ == "__main__":
    # [优化] 引入 argparse 取代硬编码的绝对路径，增强脚本的可移植性和批处理友好性
    parser = argparse.ArgumentParser(description="Adaptive Cold Stream Identification Pipeline")
    
    # 设置默认值为您原来的路径，保证不加任何参数时也能向后兼容直接运行
    parser.add_argument("--basePath", type=str, 
                        default='/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_data',
                        help="Illustris/TNG data base path")
    parser.add_argument("--cutout_dir", type=str, 
                        default='/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/cutouts',
                        help="Directory containing HDF5 cutout files")
    parser.add_argument("--output_dir", type=str, 
                        default='/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/streams',
                        help="Directory to save final stream property files")
    parser.add_argument("--snapNum", type=int, default=99, help="Snapshot number")
    parser.add_argument("--subhalo_id", type=int, default=0, help="Target Subhalo ID")
    
    args = parser.parse_args()

    print("Loading global catalogs... (this may take a while)")
    try:
        catalogs = api.load_catalogs(args.basePath, args.snapNum)
        print("Catalogs loaded successfully.")
        
        # 运行自适应识别流水线
        adaptive_cold_stream_pipeline(
            subhalo_id=args.subhalo_id, 
            snapNum=args.snapNum, 
            catalogs=catalogs,
            cutout_dir=args.cutout_dir,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"Failed to run pipeline: {e}")