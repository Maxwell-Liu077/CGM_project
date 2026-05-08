# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

This is a **CGM (Circumgalactic Medium) cold stream detection** research project analyzing gas dynamics in the IllustrisTNG cosmological simulation (TNG50-1). The pipeline identifies cold gas streams flowing into galaxies by filtering gas particles based on thermodynamic/kinematic properties, reconstructing their 3D topology via Delaunay triangulation, and merging fragments into coherent structures.

## Directory Structure

```
src/
  API_methods.py          # Core computational library: data loading, spatial/physical filtering,
                          # Delaunay topology reconstruction, connected component identification (BFS),
                          # fragment merging (KDTree), and property extraction
  adaptive_pipeline.py    # Main adaptive pipeline: Phase 0 (pre-processing) -> Phase 1 (topology
                          # scan with plateau selection) -> Phase 2 (merge scan with plateau selection) -> I/O
  batch_run.py            # Batch processor: applies galaxy selection Filter, iterates over subhalos,
                          # runs adaptive pipeline per subhalo. Currently stops after 1 subhalo (test mode).
  download_cutouts.py     # Downloads gas particle cutouts from TNG API for selected subhalos
  plot_cold_streams_2d_opt.py  # 2D projection visualization (XY/XZ/YZ) with journal-quality output
  plot_cold_streams_3d_opt.py  # 3D visualization of cold streams
  HDBSCAN_MERGING_PLAN.md # Design doc for planned Phase 3: 6D phase-space HDBSCAN clustering
data/                     # Raw data (too large for git, excluded in .gitignore)
output/                   # HDF5 output files with detected cold stream properties
figures/                  # Generated PDF/PNG figures
notebooks/                # Jupyter notebooks for exploration and testing
```

## Key Architecture

### Pipeline Flow

1. **Download** (`download_cutouts.py`): Fetches gas particle cutouts from TNG50-1 API for subhalos matching selection criteria
2. **Filter** (`batch_run.py:Filter`): Selects central galaxies by stellar mass, sSFR, halo mass, etc.
3. **Adaptive Pipeline** (`adaptive_pipeline.py`):
   - **Phase 0**: Spatial mask + satellite removal + physical condition filtering (temperature, density, radial velocity)
   - **Phase 1**: Delaunay triangulation topology scan with coarse-to-fine `edge_tolerance` parameter selection via plateau detection
   - **Phase 2**: Fragment merging via KDTree centroid distance with coarse-to-fine `merge_fraction` parameter selection
   - **I/O**: Extract and save stream properties (mass, volume, density, temperature, metallicity, velocity) to HDF5
4. **Visualization** (`plot_cold_streams_*.py`): 2D/3D publication-quality figures

### Core Modules

- **`API_methods.py`**: The foundational library. Contains all computational primitives:
  - `load_catalogs()` / `load_cutout()` — data loading via `illustris_python` and HDF5
  - `stream_spatial_mask()` — spatial filtering + satellite removal
  - `generate_physical_mask()` — thermodynamic (T: 5e3-2.5e5 K, nH: 1e-4 to 1.0) and kinematic filtering
  - `identify_stream_components()` — Delaunay triangulation + BFS connected components (numba-accelerated)
  - `merge_fragmented_streams()` — KDTree-based centroid merging with union-find
  - `extract_stream_properties()` — HDF5 output with per-stream physical properties

- **`adaptive_pipeline.py`**: Orchestrates the pipeline with adaptive parameter selection. Uses `find_plateau_1d()` to scan parameter space and pick the most stable component count.

### Planned: Phase 3 HDBSCAN

`HDBSCAN_MERGING_PLAN.md` describes a planned addition of 6D phase-space HDBSCAN clustering to merge fragments that are spatially dispersed but velocity-coherent. Not yet implemented.

## External Dependencies

- **Python packages**: `numpy`, `scipy`, `h5py`, `numba`, `illustris_python`, `matplotlib`, `requests`, `tqdm`
- **Planned**: `hdbscan` (for Phase 3, not yet implemented)
- **TNG simulation data**: Located at `/public/home/zju_visitor/LiuYuanhao/tng_data/output` (remote server)
- **Cutout/Output directories**: `/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/` (remote server)

## Running the Code

```bash
# Run the full pipeline for a single subhalo (default test mode stops after 1)
cd src && python batch_run.py

# Run adaptive pipeline standalone
cd src && python adaptive_pipeline.py

# Download cutouts for filter-selected subhalos
cd src && python download_cutouts.py

# Download specific subhalos manually
cd src && python download_cutouts.py --ids 59075 59076

# 2D visualization (real data)
python src/plot_cold_streams_2d_opt.py --subhalo 59075 --snap 33

# 2D visualization (synthetic demo data)
python src/plot_cold_streams_2d_opt.py --demo
```

## Important Notes

- `batch_run.py` line 194 has a test-mode limit (`if index >= 1: break`). Remove this line to process all subhalos.
- All hard-coded paths point to a remote server (`/public/home/zju_visitor/...`). Adjust paths for local development.
- Data files (cutouts, catalogs, output) are excluded from git via `.gitignore` — they must be downloaded or copied separately.
- The `download_cutouts.py` file contains a TNG API key (`efc5aaa1...`). Treat as sensitive.
- Python files in `src/` use relative imports (e.g., `from API_methods import ...`), so run from within the `src/` directory.
