import numpy as np
from scipy import signal
from scipy import ndimage
import itertools
import copy
import os
import time

# 导入底层核心计算库（假设底层计算库命名为 API_methods）
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
    
    参数:
        x_values (array-like): 参数空间的值 (例如 edge_tolerance 或 merge_fraction)。
        y_values (array-like): 对应参数空间产生的冷流计数 (count)。
        
    返回:
        float: 最优参数值。
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
    1. 寻找二维空间中 Z_counts 最大的连通域 (scipy.ndimage.label)。
    2. 对该最大连通域使用距离变换 (scipy.ndimage.distance_transform_edt)。
    3. 寻找距离边界最远的像素点（即安全中心），将其映射回 (X, Y) 坐标。
    
    参数:
        X_grid (2D array): 参数 X 的网格 (例如 thresh_ab)。
        Y_grid (2D array): 参数 Y 的网格 (例如 thresh_ac)。
        Z_counts (2D array): 各个参数组合下产生的冷流计数。
        
    返回:
        tuple (float, float): 最优参数值 (best_X, best_Y)。
    """
    Z_counts = np.asarray(Z_counts)
    
    # 容错：如果全为空或全为0，直接返回网格中心
    if Z_counts.size == 0 or np.all(Z_counts == 0):
        mid_y, mid_x = Z_counts.shape[0] // 2, Z_counts.shape[1] // 2
        return float(X_grid[mid_y, mid_x]), float(Y_grid[mid_y, mid_x])
        
    # 提取所有大于0的独立 count 值
    unique_counts = np.unique(Z_counts)
    unique_counts = unique_counts[unique_counts > 0]
    
    max_area = 0
    best_label_mask = None
    
    # 1. 遍历所有非零 count 值，寻找面积最大的连续等值“岛屿”
    for val in unique_counts:
        # 生成当前 count 值的二值化掩膜
        binary_mask = (Z_counts == val)
        
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
    # np.argmax 返回的是展平后的索引，使用 unravel_index 转换回 2D 坐标 (row, col)
    center_y, center_x = np.unravel_index(np.argmax(distance_map), distance_map.shape)
    
    best_X = X_grid[center_y, center_x]
    best_Y = Y_grid[center_y, center_x]
    
    return float(best_X), float(best_Y)

# =========================================================================
# Pipeline Entrance: 主流水线装配
# =========================================================================

def adaptive_cold_stream_pipeline(subhalo_id, snapNum, catalogs, 
                                  cutout_dir='/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/cutouts',
                                  output_dir='/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/streams'):
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
    # Phase 3: Morphology (二维形态学过滤扫描)
    # ---------------------------------------------------------------------
    print("\n--- Phase 3: Morphology Scanning ---")
    # 构建 thresh_ab (X) 和 thresh_ac (Y) 的 2D 网格 (例如 10x10 的网格)
    ab_vals = np.linspace(1.2, 2.0, 10)
    ac_vals = np.linspace(2.0, 4.0, 10)
    AB_grid, AC_grid = np.meshgrid(ab_vals, ac_vals)
    phase3_counts = np.zeros_like(AB_grid, dtype=int)
    
    for i in range(AB_grid.shape[0]):
        for j in range(AB_grid.shape[1]):
            # 同样必须传入 identity.copy() 隔离状态
            _, cnt = api.filter_streams_morphology(
                gas_data, subhalo_info, identity.copy(), count, BoxSize, 
                thresh_ab=AB_grid[i, j], thresh_ac=AC_grid[i, j]
            )
            phase3_counts[i, j] = cnt
            
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
    # 示例调用
    # 实际应用中，您可以在这里解析 argparse 或读取子晕列表循环执行
    
    # 假定运行参数
    basePath = '/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_data' # Illustris/TNG 数据路径
    snapNum = 99
    subhalo_id_to_test = 0  # 替换为目标 Subhalo ID
    
    print("Loading global catalogs... (this may take a while)")
    try:
        # 需要确保 api.load_catalogs 能够正确访问 basePath
        catalogs = api.load_catalogs(basePath, snapNum)
        print("Catalogs loaded successfully.")
        
        # 运行自适应识别流
        adaptive_cold_stream_pipeline(
            subhalo_id=subhalo_id_to_test, 
            snapNum=snapNum, 
            catalogs=catalogs
        )
    except Exception as e:
        print(f"Failed to run pipeline: {e}")