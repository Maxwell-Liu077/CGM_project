import os
import time
import numpy as np
import h5py  
import illustris_python as il
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
from numba import jit

try:
    import hdbscan
except ImportError:
    hdbscan = None

# =========================================================================
# Core computation modules
# =========================================================================

def periodic_displacement(pos1, pos2, boxSize):
    """Calculate shortest displacement vector under periodic boundary conditions"""
    return (pos1 - pos2 + 0.5 * boxSize) % boxSize - 0.5 * boxSize

def periodic_distance(pos1, pos2, boxSize):
    """Compute distance under periodic boundary conditions"""
    dx = periodic_displacement(pos1, pos2, boxSize)
    return np.linalg.norm(dx, axis=-1)

def calculate_virial_velocity(m_vir, r_vir, a):
    """Calculate virial velocity in physical units (km/s)"""
    G = 43007.1 
    r_phys = r_vir * a  
    return np.sqrt(G * m_vir / r_phys)

def calculate_adaptive_edge_tolerance(m_vir, base_tol=1.25, min_tol=1.05, max_tol=1.45):
    """Dynamically compute edge tolerance based on halo mass"""
    m_ref = 1.0  
    scaling_factor = 0.1 * np.log10(m_vir / m_ref)
    adaptive_tol = base_tol + scaling_factor
    return float(np.clip(adaptive_tol, min_tol, max_tol))

# =========================================================================
# Data loading functions
# =========================================================================
def load_cutout(file_path, fields=None):
    """Load gas particle data from cutout file"""
    gas_data = {}
    with h5py.File(file_path, 'r') as f:
        if 'PartType0' not in f:
            return None
        
        keys_to_load = fields if fields else f['PartType0'].keys()
        for key in keys_to_load:
            if key in f['PartType0']:
                gas_data[key] = f[f'PartType0/{key}'][()]
    return gas_data

def load_catalogs(basePath, snapNum):
    """Preload global catalog information for snapshot"""
    header = il.groupcat.loadHeader(basePath, snapNum)

    catalogs = {
        'grnrs': il.groupcat.loadSubhalos(basePath, snapNum, fields=['SubhaloGrNr']),
        'r200c': il.groupcat.loadHalos(basePath, snapNum, fields=['Group_R_Crit200']),
        'm_vir': il.groupcat.loadHalos(basePath, snapNum, fields=['Group_M_Crit200']),
        'sub_pos': il.groupcat.loadSubhalos(basePath, snapNum, fields=['SubhaloPos']),
        'sub_vel': il.groupcat.loadSubhalos(basePath, snapNum, fields=['SubhaloVel']),
        'sub_rad': il.groupcat.loadSubhalos(basePath, snapNum, fields=['SubhaloHalfmassRad']),
        'a': header['Time'],
        'BoxSize': header['BoxSize']
    }
    return catalogs

# =========================================================================
# Spatial filtering
# =========================================================================
def get_spatial_mask(subhalo_id, snapNum, catalogs, 
                     cutout_dir='/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/cutouts', rmin_fac=0.15):
    """Compute spatial mask for galaxy based on actual bounding box extent"""
    file_path = os.path.join(cutout_dir, f"cutout_snap{snapNum:03d}_sh{subhalo_id}.hdf5")
    
    if not os.path.exists(file_path):
        print(f"[Warning] Cutout file not found: {file_path}, skipping")
        return None, None, None 

    parent_halo_id = catalogs['grnrs'][subhalo_id]
    r200c = catalogs['r200c'][parent_halo_id]
    m_vir = catalogs['m_vir'][parent_halo_id]
    center_pos = catalogs['sub_pos'][subhalo_id]
    center_vel = catalogs['sub_vel'][subhalo_id]
    BoxSize = catalogs['BoxSize']
    a = catalogs['a']

    gas_data = load_cutout(file_path)
    
    if gas_data is None or 'Coordinates' not in gas_data:
        print("[Warning] Cutout contains no gas particles, skipping")
        return None, None, None
        
    pos = gas_data['Coordinates']
    dx = periodic_displacement(pos, center_pos, BoxSize)
    
    L_x = dx[:, 0].max() - dx[:, 0].min()
    L_y = dx[:, 1].max() - dx[:, 1].min()
    L_z = dx[:, 2].max() - dx[:, 2].min()

    r_min = rmin_fac * r200c
    r_search_max = np.sqrt((L_x / 2)**2 + (L_y / 2)**2 + (L_z / 2)**2)

    dist = np.linalg.norm(dx, axis=-1)
    spatial_mask = (dist >= r_min)
    
    subhalo_info = {
        "subhalo_id": subhalo_id,
        "center_pos": center_pos,
        "center_vel": center_vel,
        "r200c": r200c, 
        "m_vir": m_vir,
        "r_search_max": r_search_max,      
        "dist": dist,
        "a": a
    }

    return spatial_mask, gas_data, subhalo_info

def remove_satellites(pos, spatial_mask, subhalo_id, center_pos, r_max, catalogs, sat_fac=2.0):
    """Remove satellite galaxy contamination"""
    all_sub_pos = catalogs['sub_pos']
    all_sub_rad = catalogs['sub_rad']
    BoxSize = catalogs['BoxSize']

    sub_dists = periodic_distance(all_sub_pos, center_pos, BoxSize)
    nearby_sub_ids = np.where(sub_dists <= r_max)[0]
    
    global_gas_indices = np.where(spatial_mask)[0]
    if len(global_gas_indices) == 0:
        return 0, 0
        
    global_gas_pos = pos[global_gas_indices]
    safe_gas_pos = global_gas_pos % BoxSize
    tree = cKDTree(safe_gas_pos, boxsize=BoxSize)
    
    removed_sub_count = 0
    to_remove_indices_set = set()
    
    for sid in nearby_sub_ids:
        if sid == subhalo_id:
            continue
            
        sat_pos = all_sub_pos[sid] % BoxSize
        sat_radius = all_sub_rad[sid] * sat_fac
        
        if sat_radius <= 0:
            continue
            
        idx_in_radius = tree.query_ball_point(sat_pos, r=sat_radius)
        
        if idx_in_radius:
            to_remove_indices_set.update(idx_in_radius)
            removed_sub_count += 1
            
    removed_gas_count = len(to_remove_indices_set)
    
    if removed_gas_count > 0:
        remove_global_idx = global_gas_indices[list(to_remove_indices_set)]
        spatial_mask[remove_global_idx] = False
            
    return removed_sub_count, removed_gas_count

def stream_spatial_mask(subhalo_id, snapNum, catalogs, 
                        cutout_dir='/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/cutouts', rmin_fac=0.15):
    """Apply spatial filtering and satellite removal"""
    spatial_mask, gas_data, subhalo_info = get_spatial_mask(
        subhalo_id, snapNum, catalogs, cutout_dir, rmin_fac
    )

    if spatial_mask is None:
        return None, None, None

    removed_sub_count, removed_gas_count = remove_satellites(
        pos=gas_data['Coordinates'], 
        spatial_mask=spatial_mask, 
        subhalo_id=subhalo_id, 
        center_pos=subhalo_info['center_pos'], 
        r_max=subhalo_info['r_search_max'], 
        catalogs=catalogs,
        sat_fac=2.0
    )

    print(f"Removed {removed_sub_count} satellite galaxies and {removed_gas_count} gas cells")

    subhalo_info['distance'] = subhalo_info['dist'][spatial_mask]
    del subhalo_info['dist']

    return spatial_mask, gas_data, subhalo_info

# =========================================================================
# Physical condition filtering
# =========================================================================
def generate_physical_mask(gas_data, spatial_mask, subhalo_info, BoxSize, h=0.6774):
    """Apply thermodynamic and kinematic filtering on spatial candidates"""
    print(f"Computing physical properties for subhalo [{subhalo_info['subhalo_id']}]")
    
    a = subhalo_info['a']
    gas_data['PhysicalVelocity'] = gas_data['Velocities'] / np.sqrt(a)
    gas_data['PhysicalVolume'] = (gas_data['Masses'] / gas_data['Density']) * (a**3)

    rho_phys = gas_data['Density'] / (a**3)
    UnitMass_in_g = 1.989e43 / h
    UnitLength_in_cm = 3.085678e21 / h
    rho_cgs = rho_phys * (UnitMass_in_g / UnitLength_in_cm**3)
    gas_data['nH'] = (rho_cgs * 0.76) / 1.672622e-24

    gamma, k_B, m_p, XH = 5.0/3.0, 1.380650e-16, 1.672622e-24, 0.76
    xe = gas_data['ElectronAbundance']
    ie = gas_data['InternalEnergy']
    meanmolwt = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * xe) * m_p
    temp = ie * 1e10 * (gamma - 1.0) * meanmolwt / k_B
    
    sfr = gas_data['StarFormationRate']
    temp[sfr > 0.0] = 1e3
    gas_data['Temperature'] = temp

    candidate_indices = np.where(spatial_mask)[0]
    num_candidates = len(candidate_indices)
    
    if num_candidates == 0:
        print("No spatial candidates found, skipping physical filtering")
        return np.zeros_like(spatial_mask, dtype=bool)

    pos = gas_data['Coordinates'][candidate_indices]
    nH_local = gas_data['nH'][candidate_indices]
    temp_local = gas_data['Temperature'][candidate_indices]
    vel_phys_local = gas_data['PhysicalVelocity'][candidate_indices]

    mask_temp_local = (temp_local >= 5e3) & (temp_local <= 2.5e5)
    mask_dens_local = (nH_local >= 1e-4) & (nH_local <= 1.0)

    center_vel = subhalo_info['center_vel']
    center_pos = subhalo_info['center_pos']
    
    dv = vel_phys_local - center_vel
    dx = periodic_displacement(pos, center_pos, BoxSize)
    
    dist = np.linalg.norm(dx, axis=-1)
    dist = np.clip(dist, 1e-5, None) 
    
    v_rad_local = np.sum(dv * dx, axis=1) / dist
    v_tot_local = np.linalg.norm(dv, axis=-1)
    v_tot_local = np.clip(v_tot_local, 1e-5, None)
    
    v_vir = calculate_virial_velocity(subhalo_info['m_vir'], subhalo_info['r200c'], a)
    
    mask_vrad_limit_local = v_rad_local < (-0.2 * v_vir)
    mask_vratio_local = (np.abs(v_rad_local) / v_tot_local) > 0.8
    
    valid_local_mask = (mask_temp_local & mask_dens_local & mask_vrad_limit_local & mask_vratio_local)
    
    print(f"[{subhalo_info['subhalo_id']}] Physical filtering complete: retained {np.sum(valid_local_mask)} cold stream cells (V_vir={v_vir:.2f} km/s)")
    
    final_valid_mask = np.zeros_like(spatial_mask, dtype=bool)
    final_valid_mask[candidate_indices] = valid_local_mask
    
    return final_valid_mask

# =========================================================================
# Topology reconstruction and connectivity analysis
# =========================================================================

def build_local_topology(gas_data, valid_indices, BoxSize, edge_tolerance):
    """Build 3D topology with Delaunay triangulation and edge pruning"""
    valid_pos = gas_data['Coordinates'][valid_indices]
    n_valid = len(valid_pos)
    
    valid_mass = gas_data['Masses'][valid_indices]
    valid_rho = gas_data['Density'][valid_indices]
    valid_vol = valid_mass / valid_rho
    
    if n_valid < 4:
        num_ngb = np.full(n_valid, n_valid - 1, dtype=np.int32)
        offset_ngb = np.arange(0, n_valid * (n_valid - 1), n_valid - 1, dtype=np.int32)
        ngb_inds = np.array([j for i in range(n_valid) for j in range(n_valid) if i != j], dtype=np.int32)
        return num_ngb, ngb_inds, offset_ngb
        
    tri = Delaunay(valid_pos)
    
    edges = np.vstack([
        tri.simplices[:, [0, 1]], tri.simplices[:, [1, 0]],
        tri.simplices[:, [0, 2]], tri.simplices[:, [2, 0]],
        tri.simplices[:, [0, 3]], tri.simplices[:, [3, 0]],
        tri.simplices[:, [1, 2]], tri.simplices[:, [2, 1]],
        tri.simplices[:, [1, 3]], tri.simplices[:, [3, 1]],
        tri.simplices[:, [2, 3]], tri.simplices[:, [3, 2]],
    ])
    
    pos_A = valid_pos[edges[:, 0]]
    pos_B = valid_pos[edges[:, 1]]
    dist_AB = periodic_distance(pos_A, pos_B, BoxSize)
    
    rad = (3.0 * valid_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    rad_A = rad[edges[:, 0]]
    rad_B = rad[edges[:, 1]]
    
    valid_edges_mask = dist_AB < edge_tolerance * (rad_A + rad_B)
    real_edges = edges[valid_edges_mask]
    
    if len(real_edges) == 0:
        return np.zeros(n_valid, dtype=np.int32), np.array([], dtype=np.int32), np.zeros(n_valid, dtype=np.int32)
    

    row = real_edges[:, 0]
    col = real_edges[:, 1]
    data = np.ones(len(row), dtype=bool)
    
    adj_matrix = csr_matrix((data, (row, col)), shape=(n_valid, n_valid))
    
    offset_ngb = adj_matrix.indptr.astype(np.int32)
    ngb_inds = adj_matrix.indices.astype(np.int32)
    num_ngb = np.diff(offset_ngb).astype(np.int32)
    
    return num_ngb, ngb_inds, offset_ngb

@jit(nopython=True, nogil=True)
def _find_components_bfs(n_valid, num_ngb, offset_ngb, ngb_inds):
    """BFS algorithm for connected component identification"""
    labels = np.full(n_valid, -1, dtype=np.int32)
    current_label = 0
    queue = np.zeros(n_valid, dtype=np.int32)
    
    for i in range(n_valid):
        if labels[i] == -1:
            labels[i] = current_label
            queue[0] = i
            head = 0
            tail = 1
            
            while head < tail:
                node = queue[head]
                head += 1
                
                start = offset_ngb[node]
                end = start + num_ngb[node]
                
                for j in range(start, end):
                    neighbor = ngb_inds[j]
                    
                    if labels[neighbor] == -1:
                        labels[neighbor] = current_label
                        queue[tail] = neighbor
                        tail += 1
            
            current_label += 1
            
    return labels, current_label

def identify_stream_components(gas_data, final_valid_mask, BoxSize, edge_tolerance):
    """Identify connected stream components using topology reconstruction"""
    print("Performing cold stream topology reconstruction")
    start_time = time.time()
    
    valid_indices = np.where(final_valid_mask)[0]
    n_valid = len(valid_indices)
    
    identity = np.full(final_valid_mask.shape, -1, dtype=np.int32)
    
    if n_valid == 0:
        print("No valid cold stream cells found, skipping topology reconstruction")
        return identity, 0
        
    print(f"Processing {n_valid} cold stream cells with adaptive edge tolerance: {edge_tolerance:.3f}")
    
    t0 = time.time()
    num_ngb, ngb_inds, offset_ngb = build_local_topology(
        gas_data, valid_indices, BoxSize, edge_tolerance=edge_tolerance
    )
    print(f"Topology construction complete: {time.time()-t0:.3f}s (nodes: {n_valid}, edges: {len(ngb_inds)//2})")
    
    t1 = time.time()
    local_labels, num_components = _find_components_bfs(n_valid, num_ngb, offset_ngb, ngb_inds)
    identity[valid_indices] = local_labels
    
    print(f"Topology reconstruction complete: identified {num_components} independent stream components")
    print(f"BFS traversal time: {time.time()-t1:.4f}s. Total module time: {time.time()-start_time:.3f}s")
    
    return identity, num_components

# =========================================================================
# Morphological and advanced filtering
# =========================================================================

def _get_surviving_gas_props(gas_data, valid_indices):
    """Extract physical properties of surviving gas cells"""
    pos = gas_data['Coordinates'][valid_indices]
    mass = gas_data['Masses'][valid_indices]
    vol = gas_data['PhysicalVolume'][valid_indices]
    return pos, mass, vol

def merge_fragmented_streams(gas_data, subhalo_info, identity, count, BoxSize, merge_fraction=0.02):
    """Merge spatially close stream fragments using centroid-based KDTree query."""
    if count <= 1:
        return identity, count

    print(f"[{subhalo_info['subhalo_id']}] Merging fragmented streams, initial components: {count}")

    valid_mask = identity >= 0
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return identity, 0

    pos = gas_data['Coordinates'][valid_indices]
    local_identities = identity[valid_indices]

    r200c = subhalo_info['r200c']
    tol = merge_fraction * r200c
    print(f"Merge tolerance: {tol:.2f} kpc/h ({merge_fraction*100:.1f}% R200c)")

    # Compute centroid per component
    centers = np.zeros((count, 3))
    for i in range(count):
        centers[i] = np.mean(pos[local_identities == i] % BoxSize, axis=0)

    # Single KDTree + query_pairs: O(count log count) instead of O(count^2 * n_particles)
    tree = cKDTree(centers, boxsize=BoxSize)
    pairs = tree.query_pairs(r=tol)

    # Union-Find
    parent = np.arange(count)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)

    # Relabel components
    new_labels = np.zeros(count, dtype=np.int32)
    mapping = {}
    next_label = 0
    for i in range(count):
        root = find(i)
        if root not in mapping:
            mapping[root] = next_label
            next_label += 1
        new_labels[i] = mapping[root]

    n_components = next_label

    if n_components == count:
        print("No fragments merged")
        return identity, count
    else:
        print(f"Merging complete: {count} fragments merged into {n_components} continuous structures")

    identity[valid_indices] = new_labels[local_identities]

    return identity, n_components

def extract_stream_properties(gas_data, subhalo_info, identity, count, catalogs, output_dir, snapNum):
    """Extract and save physical properties of validated cold streams"""
    os.makedirs(output_dir, exist_ok=True)
    saveFilename = os.path.join(output_dir, f"cold_streams_snap{snapNum:03d}_sh{subhalo_info['subhalo_id']}.hdf5")

    if count == 0:
        print(f"[{subhalo_info['subhalo_id']}] No cold streams detected, writing placeholder file")
        with h5py.File(saveFilename, "w") as f:
            f["objects/count"] = 0
            f["props/dummy"] = 0
        return {"count": 0}, {"dummy": 0}

    print(f"[{subhalo_info['subhalo_id']}] Extracting properties for {count} cold streams")
    start_time = time.time()
    BoxSize = catalogs['BoxSize']

    valid_indices = np.where(identity >= 0)[0]
    local_identities = identity[valid_indices]

    lengths = np.bincount(local_identities, minlength=count)
    sort_idx = np.argsort(local_identities, kind="mergesort")
    cell_inds = valid_indices[sort_idx]

    offsets = np.zeros(count, dtype="int32")
    offsets[1:] = np.cumsum(lengths)[:-1]

    pos = gas_data['Coordinates'][cell_inds]
    mass = gas_data['Masses'][cell_inds]
    vol = gas_data['PhysicalVolume'][cell_inds]
    dens = gas_data['nH'][cell_inds]
    temp = gas_data['Temperature'][cell_inds]
    vel_phys = gas_data['PhysicalVelocity'][cell_inds]
    metal = gas_data['GFM_Metallicity'][cell_inds] / 0.0127

    center_pos = subhalo_info['center_pos']
    center_vel = subhalo_info['center_vel']

    vrel = vel_phys - center_vel
    dx = periodic_displacement(pos, center_pos, BoxSize)
    dist_all = np.linalg.norm(dx, axis=-1)
    dist_all = np.clip(dist_all, 1e-5, None)
    vrad = np.sum(vrel * dx, axis=1) / dist_all

    props = {
        "count": np.array([count], dtype="int32"),
        "vol": np.zeros(count, dtype="float32"),
        "mass": np.zeros(count, dtype="float32"),
        "dens_mean": np.zeros(count, dtype="float32"),
        "temp_mean": np.zeros(count, dtype="float32"),
        "metal_mean": np.zeros(count, dtype="float32"),
        "vrad_mean": np.zeros(count, dtype="float32"),
        "cen": np.zeros((count, 3), dtype="float32"),
        "cen_masswt": np.zeros((count, 3), dtype="float32"),
        "vrel_masswt": np.zeros((count, 3), dtype="float32"),
        "vrel_denswt": np.zeros((count, 3), dtype="float32"),
    }

    for i in range(count):
        loc = slice(offsets[i], offsets[i] + lengths[i])

        props["vol"][i] = vol[loc].sum()
        props["mass"][i] = mass[loc].sum()
        props["dens_mean"][i] = dens[loc].mean()
        props["temp_mean"][i] = temp[loc].mean()
        props["metal_mean"][i] = metal[loc].mean()
        props["vrad_mean"][i] = vrad[loc].mean()

        avg_dx = np.average(dx[loc, :], axis=0)
        avg_dx_masswt = np.average(dx[loc, :], axis=0, weights=mass[loc])

        props["cen"][i, :] = (center_pos + avg_dx) % BoxSize
        props["cen_masswt"][i, :] = (center_pos + avg_dx_masswt) % BoxSize

        props["vrel_masswt"][i, :] = np.average(vrel[loc, :], axis=0, weights=mass[loc])
        props["vrel_denswt"][i, :] = np.average(vrel[loc, :], axis=0, weights=dens[loc])

    props["distance"] = periodic_distance(props["cen_masswt"], center_pos, BoxSize)

    objects = {
        "count": count,
        "lengths": lengths,
        "offsets": offsets,
        "cell_inds": cell_inds,
    }

    with h5py.File(saveFilename, "w") as f:
        f.attrs["SubhaloID"] = subhalo_info['subhalo_id']
        f.attrs["CenterPos"] = center_pos
        f.attrs["CenterVel"] = center_vel
        f.attrs["R200c"] = subhalo_info['r200c']
        f.attrs["ScaleFactor"] = catalogs['a']
        f.attrs["Mvir"] = subhalo_info['m_vir']

        for key in objects:
            f["objects/%s" % key] = objects[key]
        for key in props:
            f["props/%s" % key] = props[key]

    print(f"Data saved to: {saveFilename}")
    print(f"Property extraction complete: {time.time()-start_time:.2f}s")

    return objects, props

def _select_plateau(x_values, y_values):
    """Select the most stable x-value from a 1D scan via plateau detection."""
    from scipy import signal
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    if len(x_values) < 3:
        return float(x_values[len(x_values) // 2]) if len(x_values) > 0 else None
    y_filtered = signal.medfilt(y_values, kernel_size=3)
    segments = []
    start_idx = 0
    for val in y_filtered:
        if not segments or segments[-1][2] != val:
            segments.append([start_idx, 1, val])
        else:
            segments[-1][1] += 1
        start_idx += 1
    max_len = max(seg[1] for seg in segments)
    candidates = [seg for seg in segments if seg[1] == max_len]
    center_idx = len(x_values) / 2.0
    best_segment = min(candidates, key=lambda seg: abs((seg[0] + seg[1] / 2.0) - center_idx))
    best_mid_idx = int(best_segment[0] + best_segment[1] // 2)
    return float(x_values[best_mid_idx])

# =========================================================================
# HDBSCAN 6D phase-space merging (Phase 3)
# =========================================================================

def _build_hdbscan_features(gas_data, identity, count, BoxSize):
    """Build 6D phase-space feature matrix for stream fragments."""
    valid_mask = identity >= 0
    valid_indices = np.where(valid_mask)[0]
    pos = gas_data['Coordinates'][valid_indices]
    vel = gas_data['PhysicalVelocity'][valid_indices]
    mass = gas_data['Masses'][valid_indices]
    local_identities = identity[valid_indices]

    features = np.zeros((count, 6))
    for i in range(count):
        member = local_identities == i
        member_pos = pos[member]
        ref = member_pos[0]
        dx = periodic_displacement(member_pos, ref, BoxSize)
        features[i, :3] = (ref + dx.mean(axis=0)) % BoxSize
        features[i, 3:] = np.average(vel[member], axis=0, weights=mass[member])

    return features, valid_indices, local_identities

def _scale_features(features, subhalo_info, v_vir=None):
    """Normalize position by R200c, velocity by virial velocity."""
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

def _run_hdbscan_at_epsilon(features_scaled, count, epsilon, min_cluster_size=3, min_samples=1):
    """Run HDBSCAN at a fixed epsilon, return (new_labels, n_components)."""
    if hdbscan is None:
        raise ImportError("hdbscan is required. Install with: pip install hdbscan")

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

    return new_labels, next_label

def merge_streams_hdbscan(
    gas_data,
    subhalo_info,
    identity,
    count,
    BoxSize,
    v_vir=None,
    min_cluster_size=3,
    min_samples=1,
    eps_coarse_min=0.02,
    eps_coarse_max=0.50,
    eps_coarseN=8,
    eps_fineN=10,
    eps_fine_frac=0.20,
    convergence_threshold=10,
):
    """
    Merge stream fragments using HDBSCAN in 6D phase space
    with adaptive epsilon selection via coarse-to-fine plateau scanning.
    """
    if count <= 1:
        return identity, count

    print(f"[{subhalo_info['subhalo_id']}] Phase 3: HDBSCAN adaptive epsilon scan, initial components: {count}")

    features, valid_indices, local_identities = _build_hdbscan_features(
        gas_data, identity, count, BoxSize
    )
    features_scaled, v_vir = _scale_features(features, subhalo_info, v_vir)

    # Coarse scan
    eps_coarse = np.linspace(eps_coarse_min, eps_coarse_max, eps_coarseN)
    counts_coarse = []
    labels_list_coarse = []
    for eps in eps_coarse:
        lbls, cnt = _run_hdbscan_at_epsilon(
            features_scaled, count, eps, min_cluster_size, min_samples
        )
        counts_coarse.append(cnt)
        labels_list_coarse.append(lbls)

    # Filter to convergence zone
    conv_mask = [c <= convergence_threshold for c in counts_coarse]
    if not any(conv_mask):
        best_idx = int(np.argmin(counts_coarse))
        best_eps = float(eps_coarse[best_idx])
        best_labels = labels_list_coarse[best_idx]
        best_count = counts_coarse[best_idx]
        print(f"  [epsilon scan] No convergence (count<={convergence_threshold}) found; "
              f"using min-count epsilon={best_eps:.3f} (count={best_count})")
    else:
        eps_conv = eps_coarse[conv_mask]
        counts_conv = [c for c, m in zip(counts_coarse, conv_mask) if m]
        labels_conv = [l for l, m in zip(labels_list_coarse, conv_mask) if m]

        best_eps = _select_plateau(eps_conv, counts_conv)
        if best_eps is None:
            best_eps = float(eps_conv[len(eps_conv) // 2])

        # Fine scan
        lo = max(eps_coarse_min, best_eps * (1 - eps_fine_frac))
        hi = min(eps_coarse_max, best_eps * (1 + eps_fine_frac))
        eps_fine = np.linspace(lo, hi, eps_fineN)
        counts_fine = []
        labels_fine = []
        for eps in eps_fine:
            lbls, cnt = _run_hdbscan_at_epsilon(
                features_scaled, count, eps, min_cluster_size, min_samples
            )
            counts_fine.append(cnt)
            labels_fine.append(lbls)

        conv_mask_fine = [c <= convergence_threshold for c in counts_fine]
        if any(conv_mask_fine):
            eps_fine_conv = eps_fine[conv_mask_fine]
            counts_fine_conv = [c for c, m in zip(counts_fine, conv_mask_fine) if m]
            labels_fine_conv = [l for l, m in zip(labels_fine, conv_mask_fine) if m]

            best_eps_fine = _select_plateau(eps_fine_conv, counts_fine_conv)
            if best_eps_fine is not None:
                best_eps = best_eps_fine
                idx = int(np.argmin(np.abs(eps_fine_conv - best_eps_fine)))
                best_labels = labels_fine_conv[idx]
                best_count = counts_fine_conv[idx]
                print(f"  [epsilon scan] Locked epsilon={best_eps:.4f} (count={best_count})")
            else:
                best_idx = int(np.argmin(counts_fine_conv))
                best_eps = float(eps_fine_conv[best_idx])
                best_labels = labels_fine_conv[best_idx]
                best_count = counts_fine_conv[best_idx]
                print(f"  [epsilon scan] No fine plateau; using min-count epsilon={best_eps:.3f} (count={best_count})")
        else:
            best_idx = int(np.argmin(counts_fine))
            best_eps = float(eps_fine[best_idx])
            best_labels = labels_fine[best_idx]
            best_count = counts_fine[best_idx]
            print(f"  [epsilon scan] No fine convergence; using min-count epsilon={best_eps:.3f} (count={best_count})")

    identity[valid_indices] = best_labels[local_identities]
    print(f"[{subhalo_info['subhalo_id']}] Phase 3 complete: {count} fragments merged into {best_count} structures (epsilon={best_eps:.4f})")

    return identity, best_count

def analyze_cold_streams_pipeline(subhalo_id, snapNum, catalogs, 
                                  cutout_dir='/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/cutouts',
                                  output_dir='/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/streams',
                                  rmin_fac=0.15, 
                                  base_edge_tolerance=1.25, min_edge_tolerance=1.05, max_edge_tolerance=1.45,
                                  merge_fraction=0.01): 
    """Main pipeline for cold stream identification and analysis"""
    os.makedirs(output_dir, exist_ok=True)
    saveFilename = os.path.join(output_dir, f"cold_streams_snap{snapNum:03d}_sh{subhalo_id}.hdf5")
    
    if os.path.isfile(saveFilename):
        print(f"[{subhalo_id}] Existing cold stream data found: {saveFilename}, skipping computation")
        objects, props = {}, {}
        with h5py.File(saveFilename, "r") as f:
            if "objects" in f:
                for key in f["objects"]: objects[key] = f["objects"][key][()]
            if "props" in f:
                for key in f["props"]: props[key] = f["props"][key][()]
        return objects, props

    print(f"\n{'='*65}")
    print(f"Starting cold stream analysis | Snapshot: {snapNum} | Subhalo: {subhalo_id}")
    print(f"{'='*65}")
    pipeline_start = time.time()

    spatial_mask, gas_data, subhalo_info = stream_spatial_mask(
        subhalo_id, snapNum, catalogs, cutout_dir, rmin_fac
    )
    
    if spatial_mask is None:
        print(f"[{subhalo_id}] Pipeline terminated: invalid spatial mask or missing data")
        return {}, {}

    final_valid_mask = generate_physical_mask(
        gas_data, spatial_mask, subhalo_info, catalogs['BoxSize']
    )

    m_vir = subhalo_info['m_vir']
    adaptive_edge_tolerance = calculate_adaptive_edge_tolerance(
        m_vir, base_tol=base_edge_tolerance, min_tol=min_edge_tolerance, max_tol=max_edge_tolerance
    )
    print(f"[{subhalo_id}] Halo mass M_vir = {m_vir:.2e}, adaptive edge tolerance: {adaptive_edge_tolerance:.3f}")

    identity, count = identify_stream_components(
        gas_data, final_valid_mask, catalogs['BoxSize'], edge_tolerance=adaptive_edge_tolerance
    )

    identity, count = merge_fragmented_streams(
        gas_data, subhalo_info, identity, count, catalogs['BoxSize'], merge_fraction=merge_fraction
    )

    objects, props = extract_stream_properties(
        gas_data, subhalo_info, identity, count, catalogs, output_dir, snapNum
    )

    del gas_data 

    print(f"{'-'*65}")
    print(f"[{subhalo_id}] Cold stream analysis complete: total time {time.time()-pipeline_start:.2f}s")
    print(f"{'='*65}\n")
    
    return objects, props