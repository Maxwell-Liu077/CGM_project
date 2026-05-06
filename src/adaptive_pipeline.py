import numpy as np
from scipy import signal
import itertools
import os
import time
import h5py

# Import core computational library
import API_methods as api

# =========================================================================
# Utils: Platform Search Algorithms
# =========================================================================

def find_plateau_1d(x_values, y_values):
    """
    1D plateau search algorithm.

    Logic:
    1. Apply median filtering (scipy.signal.medfilt) to smooth discrete numerical spikes.
    2. Use itertools.groupby to find consecutive intervals with identical values.
    3. Return the midpoint x-value of the longest consecutive interval as the most stable parameter choice.
    """
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    
    if len(x_values) == 0:
        return None
    if len(x_values) < 3:
        return float(x_values[len(x_values) // 2])
        
    # 1. Apply median filtering (kernel size 3) to smooth single-point spikes from discrete grids
    y_filtered = signal.medfilt(y_values, kernel_size=3)
    
    # 2. Find consecutive intervals with identical values
    segments = []
    start_idx = 0
    for k, g in itertools.groupby(y_filtered):
        length = len(list(g))
        segments.append((start_idx, length))
        start_idx += length
        
    # Get the length of the longest interval
    max_len = max([seg[1] for seg in segments])
    
    # Filter all intervals that reach the maximum length
    candidates = [seg for seg in segments if seg[1] == max_len]
    
    # 3. If multiple intervals share the max length, prefer the one closest to the center of parameter space
    center_idx = len(x_values) / 2.0
    best_segment = min(candidates, key=lambda seg: abs((seg[0] + seg[1] / 2.0) - center_idx))
    
    # Take the midpoint index of the best interval
    best_mid_idx = int(best_segment[0] + best_segment[1] // 2)
    
    return float(x_values[best_mid_idx])

# =========================================================================
# Pipeline Entrance: Main Pipeline Assembly
# =========================================================================

def adaptive_cold_stream_pipeline(subhalo_id, snapNum, catalogs, cutout_dir, output_dir):
    """
    Sequential multi-scale adaptive cold stream identification master pipeline.
    Includes pre-processing, Phase 1 (topology scan), Phase 2 (merge scan), and final data I/O.
    """
    saveFilename = os.path.join(output_dir, f"cold_streams_snap{snapNum:03d}_sh{subhalo_id}.hdf5")

    if os.path.isfile(saveFilename):
        print(f"[{subhalo_id}] Existing cold stream data found: {saveFilename}, skipping.")
        objects, props = {}, {}
        with h5py.File(saveFilename, 'r') as f:
            if 'objects' in f:
                for key in f['objects']:
                    objects[key] = f['objects'][key][()]
            if 'props' in f:
                for key in f['props']:
                    props[key] = f['props'][key][()]
        return objects, props
    print(f"\n{'='*65}")
    print(f"Starting Adaptive Cold Stream Pipeline | Snap: {snapNum} | Subhalo: {subhalo_id}")
    print(f"{'='*65}")
    pipeline_start = time.time()

    # ---------------------------------------------------------------------
    # Phase 0: Pre-processing (data pre-processing and global cache)
    # ---------------------------------------------------------------------
    # Establish immutable global cache state; must not be overwritten in subsequent steps
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
    # Phase 1: Topology (edge tolerance parameter scan)
    # ---------------------------------------------------------------------
    print("\n--- Phase 1: Topology Scanning ---")
    # Coarse scan across the full parameter range, then fine scan around the best value
    edge_tol_coarse = np.linspace(1.10, 1.50, 8)
    phase1_counts_coarse = []
    for tol in edge_tol_coarse:
        _, cnt = api.identify_stream_components(gas_data, final_valid_mask, BoxSize, edge_tolerance=tol)
        phase1_counts_coarse.append(cnt)

    best_edge_tol_coarse = find_plateau_1d(edge_tol_coarse, phase1_counts_coarse)
    if best_edge_tol_coarse is None:
        best_edge_tol_coarse = 1.25
        print(f"[{subhalo_id}] Phase 1 coarse scan returned no plateau, using default 1.250")
    else:
        print(f"[{subhalo_id}] Phase 1 Coarse Best edge_tolerance: {best_edge_tol_coarse:.3f}")
        # Fine scan: 10 points within ±10% of the coarse best value, clamped to [1.10, 1.50]
        lo = max(1.10, best_edge_tol_coarse * 0.90)
        hi = min(1.50, best_edge_tol_coarse * 1.10)
        edge_tol_fine = np.linspace(lo, hi, 10)
        phase1_counts_fine = []
        for tol in edge_tol_fine:
            _, cnt = api.identify_stream_components(gas_data, final_valid_mask, BoxSize, edge_tolerance=tol)
            phase1_counts_fine.append(cnt)
        best_edge_tol = find_plateau_1d(edge_tol_fine, phase1_counts_fine)
        if best_edge_tol is None:
            best_edge_tol = best_edge_tol_coarse
            print(f"[{subhalo_id}] Phase 1 fine scan returned no plateau, falling back to coarse: {best_edge_tol:.3f}")
        else:
            print(f"[{subhalo_id}] Phase 1 Locked Best edge_tolerance: {best_edge_tol:.3f}")

    # Solidify state: run a real computation with the best parameter and save identity
    identity, count = api.identify_stream_components(
        gas_data, final_valid_mask, BoxSize, edge_tolerance=best_edge_tol
    )
    
    # Short-circuit: skip remaining phases if count <= 1
    if count <= 1:
        print(f"[{subhalo_id}] Phase 1 count <= 1, triggering short-circuit to IO.")
        return api.extract_stream_properties(gas_data, subhalo_info, identity, count, catalogs, output_dir, snapNum)

    # ---------------------------------------------------------------------
    # Phase 2: Merging (fragment merge scan)
    # ---------------------------------------------------------------------
    print("\n--- Phase 2: Merging Scanning ---")
    # Coarse scan across the full parameter range, then fine scan around the best value
    merge_fracs_coarse = np.linspace(0.005, 0.05, 6)
    phase2_counts_coarse = []
    for frac in merge_fracs_coarse:
        _, cnt = api.merge_fragmented_streams(
            gas_data, subhalo_info, identity.copy(), count, BoxSize, merge_fraction=frac
        )
        phase2_counts_coarse.append(cnt)

    best_merge_frac_coarse = find_plateau_1d(merge_fracs_coarse, phase2_counts_coarse)
    if best_merge_frac_coarse is None:
        best_merge_frac_coarse = 0.02
        print(f"[{subhalo_id}] Phase 2 coarse scan returned no plateau, using default 0.020")
    else:
        print(f"[{subhalo_id}] Phase 2 Coarse Best merge_fraction: {best_merge_frac_coarse:.4f}")
        # Fine scan: 8 points within ±20% of the coarse best value, clamped to [0.005, 0.05]
        lo = max(0.005, best_merge_frac_coarse * 0.80)
        hi = min(0.05, best_merge_frac_coarse * 1.20)
        merge_fracs_fine = np.linspace(lo, hi, 8)
        phase2_counts_fine = []
        for frac in merge_fracs_fine:
            _, cnt = api.merge_fragmented_streams(
                gas_data, subhalo_info, identity.copy(), count, BoxSize, merge_fraction=frac
            )
            phase2_counts_fine.append(cnt)
        best_merge_frac = find_plateau_1d(merge_fracs_fine, phase2_counts_fine)
        if best_merge_frac is None:
            best_merge_frac = best_merge_frac_coarse
            print(f"[{subhalo_id}] Phase 2 fine scan returned no plateau, falling back to coarse: {best_merge_frac:.4f}")
        else:
            print(f"[{subhalo_id}] Phase 2 Locked Best merge_fraction: {best_merge_frac:.4f}")

    # Solidify state
    identity, count = api.merge_fragmented_streams(
        gas_data, subhalo_info, identity, count, BoxSize, merge_fraction=best_merge_frac
    )
    
    # Short-circuit: skip remaining phases if count <= 1
    if count <= 1:
        print(f"[{subhalo_id}] Phase 2 count <= 1, triggering short-circuit to IO.")
        return api.extract_stream_properties(gas_data, subhalo_info, identity, count, catalogs, output_dir, snapNum)

    # ---------------------------------------------------------------------
    # Final Step: Data I/O (Logging & IO)
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
# Script entry point
# =========================================================================
if __name__ == "__main__":
    base_path   = '/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_data'
    cutout_dir  = '/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/cutouts'
    output_dir  = '/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/streams'
    snap_num    = 99
    subhalo_id  = 0

    print("Loading global catalogs... (this may take a while)")
    catalogs = api.load_catalogs(base_path, snap_num)
    print("Catalogs loaded successfully.")

    adaptive_cold_stream_pipeline(
        subhalo_id=subhalo_id,
        snapNum=snap_num,
        catalogs=catalogs,
        cutout_dir=cutout_dir,
        output_dir=output_dir,
    )