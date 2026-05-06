import os
import argparse
import requests
import numpy as np
import illustris_python as il
from tqdm import tqdm

# =========================================================================
# Configuration
# =========================================================================
CUTOUT_OUTPUT_DIR = "/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/cutouts"
BASE_PATH = "/public/home/zju_visitor/LiuYuanhao/tng_data/output"
SNAP_NUM = 33

API_KEY = "efc5aaa1f009fd7189ca64c564681762"
headers = {"api-key": API_KEY}
base_url = "https://www.tng-project.org/api/"
sim_name = "TNG50-1"

# Default filter kwargs (matching batch_run.py defaults)
DEFAULT_FILTER_KWARGS = {
    "h": 0.6774,
    "log_mstar_bounds": (10.0, 11.0),
    "log_mbh_bounds": None,
    "sfr_bounds": None,
    "log_ssfr_bounds": (-10, -9),
    "log_mhalo_bounds": None,
}


# =========================================================================
# Filter function (shared with batch_run.py)
# =========================================================================
def Filter(basePath, snapNum, h=0.6774,
           log_mstar_bounds=None,
           log_mbh_bounds=None,
           sfr_bounds=None,
           log_ssfr_bounds=None,
           log_mhalo_bounds=None):

    halo_fields = ["GroupFirstSub"]
    if log_mhalo_bounds is not None:
        halo_fields.append("GroupMass")

    halos = il.groupcat.loadHalos(basePath, snapNum, fields=halo_fields)

    if not isinstance(halos, dict):
        halos = {halo_fields[0]: halos}

    valid_halo_mask = halos["GroupFirstSub"] != -1
    subhalo_ids = halos["GroupFirstSub"][valid_halo_mask]

    final_mask = np.ones(len(subhalo_ids), dtype=bool)

    if log_mhalo_bounds is not None:
        mhalo_sim = halos["GroupMass"][valid_halo_mask]
        min_sim = (10**log_mhalo_bounds[0]) * h / 1e10
        max_sim = (10**log_mhalo_bounds[1]) * h / 1e10
        final_mask &= (mhalo_sim >= min_sim) & (mhalo_sim <= max_sim)

    subhalo_fields = set()
    if log_mstar_bounds or log_ssfr_bounds:
        subhalo_fields.add("SubhaloMassType")
    if log_mbh_bounds:
        subhalo_fields.add("SubhaloBHMass")
    if sfr_bounds or log_ssfr_bounds:
        subhalo_fields.add("SubhaloSFR")

    if not subhalo_fields:
        return subhalo_ids[final_mask]

    subhalo_fields_list = list(subhalo_fields)
    subhalos = il.groupcat.loadSubhalos(basePath, snapNum, fields=subhalo_fields_list)

    if not isinstance(subhalos, dict):
        subhalos = {subhalo_fields_list[0]: subhalos}

    if log_mstar_bounds is not None:
        mstar_sim = subhalos["SubhaloMassType"][subhalo_ids, 4]
        min_sim = (10**log_mstar_bounds[0]) * h / 1e10
        max_sim = (10**log_mstar_bounds[1]) * h / 1e10
        final_mask &= (mstar_sim >= min_sim) & (mstar_sim <= max_sim)

    if log_mbh_bounds is not None:
        mbh_sim = subhalos["SubhaloBHMass"][subhalo_ids]
        min_sim = (10**log_mbh_bounds[0]) * h / 1e10
        max_sim = (10**log_mbh_bounds[1]) * h / 1e10
        final_mask &= (mbh_sim >= min_sim) & (mbh_sim <= max_sim)

    if sfr_bounds is not None:
        sfr = subhalos["SubhaloSFR"][subhalo_ids]
        final_mask &= (sfr >= sfr_bounds[0]) & (sfr <= sfr_bounds[1])

    if log_ssfr_bounds is not None:
        mstar_physical = subhalos["SubhaloMassType"][subhalo_ids, 4] * 1e10 / h
        sfr = subhalos["SubhaloSFR"][subhalo_ids]

        nonzero_mask = (mstar_physical > 0) & (sfr > 0)
        ssfr = np.zeros_like(sfr)
        ssfr[nonzero_mask] = sfr[nonzero_mask] / mstar_physical[nonzero_mask]

        log_ssfr = np.full_like(ssfr, -np.inf)
        log_ssfr[nonzero_mask] = np.log10(ssfr[nonzero_mask])

        final_mask &= (log_ssfr >= log_ssfr_bounds[0]) & (log_ssfr <= log_ssfr_bounds[1])

    return subhalo_ids[final_mask]


# =========================================================================
# Download logic
# =========================================================================
def download_stream_environment(snapNum, subhalo_id):
    os.makedirs(CUTOUT_OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(CUTOUT_OUTPUT_DIR, "cutout_snap{:03d}_sh{}.hdf5".format(snapNum, subhalo_id))
    temp_path = save_path + ".part"

    if os.path.exists(save_path):
        print(f"文件已存在，跳过下载: {save_path}")
        return save_path

    try:
        all_grnrs = il.groupcat.loadSubhalos(BASE_PATH, snapNum, fields=["SubhaloGrNr"])
        parent_halo_id = all_grnrs[subhalo_id]
    except Exception as e:
        print(f"读取本地星表失败，请检查 {BASE_PATH} 下是否存在 groups_{snapNum:03d} 文件夹！")
        raise e

    gas_fields = "Coordinates,Masses,Density,InternalEnergy,ElectronAbundance,StarFormationRate,Velocities,GFM_Metallicity"
    query_url = f"{base_url}{sim_name}/snapshots/{snapNum}/halos/{parent_halo_id}/cutout.hdf5?gas={gas_fields}"

    print(f"正在从云端拉取母晕 (Halo ID: {parent_halo_id}) 的包围盒...")

    try:
        with requests.get(query_url, headers=headers, stream=True, timeout=(10, 300)) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))

            with open(temp_path, "wb") as f, tqdm(
                desc=f"ShID {subhalo_id:5d}",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                leave=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

        os.rename(temp_path, save_path)
        print(f"成功，数据保存在: {save_path}\n")

    except Exception as err:
        print(f"下载失败 Subhalo {subhalo_id}: {err}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise err

    return save_path


def get_existing_cutout_ids(snapNum, cutout_dir):
    """Scan cutout directory for already downloaded subhalo IDs."""
    import re
    pattern = re.compile(rf"cutout_snap{snapNum:03d}_sh(\d+)\.hdf5$")
    existing_ids = set()
    if os.path.exists(cutout_dir):
        for fn in os.listdir(cutout_dir):
            m = pattern.match(fn)
            if m:
                existing_ids.add(int(m.group(1)))
    return existing_ids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download TNG cutout files for selected subhalos. "
                    "Supports both Filter-based selection and manual ID input."
    )
    parser.add_argument("--ids", type=int, nargs="+", default=None,
                        help="Manually specify subhalo IDs to download")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable automatic Filter; only use --ids")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip already-downloaded cutouts (default: True)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Override --skip-existing, download all")
    # Filter parameter overrides
    parser.add_argument("--log-mstar", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="log10 stellar mass bounds, e.g. --log-mstar 10 11")
    parser.add_argument("--log-ssfr", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="log10 sSFR bounds, e.g. --log-ssfr -10 -9")
    parser.add_argument("--log-mhalo", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="log10 halo mass bounds, e.g. --log-mhalo 12 13")
    parser.add_argument("--sfr", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="SFR bounds, e.g. --sfr 0 10")
    parser.add_argument("--log-mbh", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="log10 black hole mass bounds")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    skip_existing = args.skip_existing and not args.no_skip_existing

    # ---- Determine target subhalo IDs ----
    target_ids = set()

    if not args.no_filter:
        filter_kwargs = DEFAULT_FILTER_KWARGS.copy()
        if args.log_mstar is not None:
            filter_kwargs["log_mstar_bounds"] = tuple(args.log_mstar)
        if args.log_ssfr is not None:
            filter_kwargs["log_ssfr_bounds"] = tuple(args.log_ssfr)
        if args.log_mhalo is not None:
            filter_kwargs["log_mhalo_bounds"] = tuple(args.log_mhalo)
        if args.sfr is not None:
            filter_kwargs["sfr_bounds"] = tuple(args.sfr)
        if args.log_mbh is not None:
            filter_kwargs["log_mbh_bounds"] = tuple(args.log_mbh)

        print(f"Applying Filter with args: {filter_kwargs}")
        filtered_ids = Filter(BASE_PATH, SNAP_NUM, **filter_kwargs)
        target_ids.update(filtered_ids)
        print(f"Filter selected {len(filtered_ids)} subhalos")

    if args.ids is not None:
        target_ids.update(args.ids)
        print(f"Added {len(args.ids)} manual IDs, total unique targets: {len(target_ids)}")

    if not target_ids:
        print("[Error] No subhalo IDs selected. Use --ids or configure Filter parameters.")
        exit(1)

    # ---- Remove already-downloaded IDs ----
    if skip_existing:
        existing = get_existing_cutout_ids(SNAP_NUM, CUTOUT_OUTPUT_DIR)
        to_download = sorted(target_ids - existing)
        skipped = target_ids & existing
        print(f"Already downloaded: {len(skipped)}, to download: {len(to_download)}")
    else:
        to_download = sorted(target_ids)

    if not to_download:
        print("All target cutouts already exist, nothing to download.")
        exit(0)

    # ---- Batch download ----
    print(f"\n{'='*60}")
    print(f"准备批量下载 {len(to_download)} 个目标星系的局部环境")
    print(f"{'='*60}\n")

    success = 0
    failed = []
    for shID in to_download:
        try:
            download_stream_environment(SNAP_NUM, shID)
            success += 1
        except Exception:
            failed.append(shID)

    print(f"\n{'='*60}")
    print(f"下载完毕: 成功 {success}/{len(to_download)}")
    if failed:
        print(f"失败的 subhalo IDs: {failed}")
    print(f"{'='*60}")