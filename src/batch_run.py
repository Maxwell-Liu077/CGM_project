import os
import re
import traceback
import numpy as np
import illustris_python as il
from API_methods import load_catalogs
from adaptive_pipeline import adaptive_cold_stream_pipeline

# =========================================================================
# Filter function (from notebooks/TEST.ipynb)
# =========================================================================
def Filter(basePath, snapNum, h=0.6774,
           log_mstar_bounds=None,
           log_mbh_bounds=None,
           sfr_bounds=None,
           log_ssfr_bounds=None,
           log_mhalo_bounds=None):

    halo_fields = ['GroupFirstSub']
    if log_mhalo_bounds is not None:
        halo_fields.append('GroupMass')

    halos = il.groupcat.loadHalos(basePath, snapNum, fields=halo_fields)

    if not isinstance(halos, dict):
        halos = {halo_fields[0]: halos}

    valid_halo_mask = halos['GroupFirstSub'] != -1
    subhalo_ids = halos['GroupFirstSub'][valid_halo_mask]

    final_mask = np.ones(len(subhalo_ids), dtype=bool)

    if log_mhalo_bounds is not None:
        mhalo_sim = halos['GroupMass'][valid_halo_mask]
        min_sim = (10**log_mhalo_bounds[0]) * h / 1e10
        max_sim = (10**log_mhalo_bounds[1]) * h / 1e10
        final_mask &= (mhalo_sim >= min_sim) & (mhalo_sim <= max_sim)

    subhalo_fields = set()
    if log_mstar_bounds or log_ssfr_bounds:
        subhalo_fields.add('SubhaloMassType')
    if log_mbh_bounds:
        subhalo_fields.add('SubhaloBHMass')
    if sfr_bounds or log_ssfr_bounds:
        subhalo_fields.add('SubhaloSFR')

    if not subhalo_fields:
        return subhalo_ids[final_mask]

    subhalo_fields_list = list(subhalo_fields)
    subhalos = il.groupcat.loadSubhalos(basePath, snapNum, fields=subhalo_fields_list)

    if not isinstance(subhalos, dict):
        subhalos = {subhalo_fields_list[0]: subhalos}

    if log_mstar_bounds is not None:
        mstar_sim = subhalos['SubhaloMassType'][subhalo_ids, 4]
        min_sim = (10**log_mstar_bounds[0]) * h / 1e10
        max_sim = (10**log_mstar_bounds[1]) * h / 1e10
        final_mask &= (mstar_sim >= min_sim) & (mstar_sim <= max_sim)

    if log_mbh_bounds is not None:
        mbh_sim = subhalos['SubhaloBHMass'][subhalo_ids]
        min_sim = (10**log_mbh_bounds[0]) * h / 1e10
        max_sim = (10**log_mbh_bounds[1]) * h / 1e10
        final_mask &= (mbh_sim >= min_sim) & (mbh_sim <= max_sim)

    if sfr_bounds is not None:
        sfr = subhalos['SubhaloSFR'][subhalo_ids]
        final_mask &= (sfr >= sfr_bounds[0]) & (sfr <= sfr_bounds[1])

    if log_ssfr_bounds is not None:
        mstar_physical = subhalos['SubhaloMassType'][subhalo_ids, 4] * 1e10 / h
        sfr = subhalos['SubhaloSFR'][subhalo_ids]

        nonzero_mask = (mstar_physical > 0) & (sfr > 0)
        ssfr = np.zeros_like(sfr)
        ssfr[nonzero_mask] = sfr[nonzero_mask] / mstar_physical[nonzero_mask]

        log_ssfr = np.full_like(ssfr, -np.inf)
        log_ssfr[nonzero_mask] = np.log10(ssfr[nonzero_mask])

        final_mask &= (log_ssfr >= log_ssfr_bounds[0]) & (log_ssfr <= log_ssfr_bounds[1])

    return subhalo_ids[final_mask]

# =========================================================================
# Global configuration
# =========================================================================
BASE_PATH  = '/public/home/zju_visitor/LiuYuanhao/tng_data/output'
CUTOUT_DIR = '/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/cutouts'
OUTPUT_DIR = '/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/streams'
SNAP_NUM   = 33

FILTER_KWARGS = {
    'h': 0.6774,
    'log_mstar_bounds': (10.0, 11.0),
    'log_mbh_bounds': None,
    'sfr_bounds': None,
    'log_ssfr_bounds': (-10, -9),
    'log_mhalo_bounds': None,
}

# =========================================================================
# Utility functions
# =========================================================================
def run_single_subhalo(subhalo_id, snapNum, catalogs, cutout_dir, output_dir):
    target_file = os.path.join(cutout_dir, f"cutout_snap{snapNum:03d}_sh{subhalo_id}.hdf5")

    if not os.path.exists(target_file):
        print(f"[Warning] Cutout file not found: {target_file}, skipping subhalo {subhalo_id}")
        return False, 0

    print(f"\n>>> Processing subhalo {subhalo_id} (adaptive pipeline)...")

    try:
        objects, props = adaptive_cold_stream_pipeline(
            subhalo_id=subhalo_id,
            snapNum=snapNum,
            catalogs=catalogs,
            cutout_dir=cutout_dir,
            output_dir=output_dir,
        )

        stream_count = objects.get('count', 0) if isinstance(objects, dict) else 0

        if stream_count > 0:
            print(f"[Success] Subhalo {subhalo_id} detected {stream_count} cold streams")
        else:
            print(f"[Info] Subhalo {subhalo_id} processed, no cold streams detected")

        return True, stream_count

    except Exception as e:
        print(f"[Error] Failed to process subhalo {subhalo_id}: {e}")
        traceback.print_exc()
        return False, 0

# =========================================================================
# Main
# =========================================================================
def main():
    print(f"Loading global catalogs for snapshot {SNAP_NUM}...")
    catalogs = load_catalogs(BASE_PATH, SNAP_NUM)
    print("Catalogs loaded successfully")

    # Get subhalo IDs via Filter
    print(f"\nApplying galaxy selection criteria...")
    subhalo_ids = Filter(BASE_PATH, SNAP_NUM, **FILTER_KWARGS)
    print(f"Selected {len(subhalo_ids)} central galaxies after filtering")

    # Cross-check with available cutout files
    available_pattern = re.compile(rf"cutout_snap{SNAP_NUM:03d}_sh(\d+)\.hdf5$")
    available_ids = set()
    for fn in os.listdir(CUTOUT_DIR):
        m = available_pattern.search(fn)
        if m:
            available_ids.add(int(m.group(1)))

    subhalo_ids = sorted([sid for sid in subhalo_ids if sid in available_ids])
    total = len(subhalo_ids)
    print(f"Matched cutout files: {total} subhalos to process")

    if total == 0:
        print(f"\n[Info] No matching cutout files found in {CUTOUT_DIR}")
        return

    print(f"\n{'='*60}")
    print(f"Starting batch processing of {total} subhalos")
    print(f"{'='*60}")

    success_count = 0
    stream_detection_count = 0
    failed_subhalos = []

    for index, subhalo_id in enumerate(subhalo_ids, 1):
        print(f"\n--- Progress: {index}/{total} (Subhalo ID: {subhalo_id}) ---")
        success, stream_count = run_single_subhalo(
            subhalo_id=subhalo_id,
            snapNum=SNAP_NUM,
            catalogs=catalogs,
            cutout_dir=CUTOUT_DIR,
            output_dir=OUTPUT_DIR,
        )

        if success:
            success_count += 1
            if stream_count > 0:
                stream_detection_count += 1
        else:
            failed_subhalos.append(subhalo_id)

    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total subhalos processed:       {total}")
    print(f"Successfully processed:         {success_count}")
    print(f"Failed processing:              {len(failed_subhalos)}")
    print(f"Subhalos with cold streams:     {stream_detection_count}")
    print(f"Success rate:                   {success_count/total*100:.1f}%")

    if failed_subhalos:
        print(f"\nFailed subhalo IDs: {failed_subhalos}")

    print(f"\n{'='*60}")
    print("All cold stream detection tasks completed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
