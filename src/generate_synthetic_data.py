"""
用合成数据生成冷流可视化演示图

生成方法：
- 模拟 2 条冷流气体管，沿不同方向流入 Subhalo
- 气体管由带有一定弥散的三维高斯点组成
- 温度沿管从外到内递增
"""

import os
import sys
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

np.random.seed(42)

# ============================================================================
# 合成数据参数
# ============================================================================
center_pos = np.array([17500.0, 17500.0, 17500.0])
r200c = 250.0  # kpc/h
boxSize = 35000.0

stream_defs = [
    {
        'name': 'Stream 1',
        'n_cells': 800,
        # 从 (-x) 方向流入, 沿 x 轴
        'center': np.array([-0.6 * r200c, 0.05 * r200c, 0.08 * r200c]),
        'spread': np.array([0.5 * r200c, 0.04 * r200c, 0.04 * r200c]),
        'temp_range': (8e3, 1.5e5),
        'mass_range': (1e6, 5e6),
    },
    {
        'name': 'Stream 2',
        'n_cells': 600,
        # 从 (+y, +z) 方向流入, 弯曲
        'center': np.array([0.05 * r200c, 0.55 * r200c, -0.15 * r200c]),
        'spread': np.array([0.04 * r200c, 0.45 * r200c, 0.05 * r200c]),
        'temp_range': (6e3, 2.0e5),
        'mass_range': (2e6, 8e6),
    },
]

# ============================================================================
# 生成合成数据
# ============================================================================
stream_dir = os.path.join(os.path.dirname(__file__), 'synthetic_data')
os.makedirs(stream_dir, exist_ok=True)

subhalo_id = 59075
snapNum = 33

# 生成流数据文件
stream_file = os.path.join(stream_dir, f"cold_streams_snap{snapNum:03d}_sh{subhalo_id}.hdf5")

all_positions = []
all_masses = []
all_temps = []
offsets = []
lengths = []
cell_inds = []  # 合成全局索引

for si, sdef in enumerate(stream_defs):
    n = sdef['n_cells']
    pos = np.random.randn(n, 3) * sdef['spread'] + sdef['center']
    # 转回绝对坐标
    pos_abs = pos + center_pos
    # 处理周期性边界
    pos_abs = pos_abs % boxSize

    # 温度：距离中心越远温度越低，加入随机噪声
    dist_from_center = np.linalg.norm(pos, axis=1)
    max_dist = dist_from_center.max()
    temp_frac = np.clip(1.0 - dist_from_center / max_dist, 0, 1)
    temps = sdef['temp_range'][0] + temp_frac * (sdef['temp_range'][1] - sdef['temp_range'][0])
    temps *= (1 + 0.15 * np.random.randn(n))  # 加噪声
    temps = np.clip(temps, 5e3, 2.5e5)

    masses = np.random.uniform(sdef['mass_range'][0], sdef['mass_range'][1], n)

    start_idx = sum(lengths[:si])
    offsets.append(start_idx)
    lengths.append(n)

    all_positions.append(pos_abs)
    all_masses.append(masses)
    all_temps.append(temps)

all_positions = np.concatenate(all_positions)
all_masses = np.concatenate(all_masses)
all_temps = np.concatenate(all_temps)

with h5py.File(stream_file, 'w') as f:
    f.attrs['SubhaloID'] = subhalo_id
    f.attrs['CenterPos'] = center_pos
    f.attrs['CenterVel'] = np.array([0.0, 0.0, 0.0])
    f.attrs['R200c'] = r200c
    f.attrs['ScaleFactor'] = 1.0
    f.attrs['Mvir'] = 1.0

    f['objects/count'] = len(stream_defs)
    f['objects/lengths'] = np.array(lengths, dtype='int32')
    f['objects/offsets'] = np.array(offsets, dtype='int32')

    # cell_inds: 连续的索引数组
    cell_inds_arr = np.array([], dtype='int32')
    for i, n in enumerate(lengths):
        cell_inds_arr = np.concatenate([cell_inds_arr, np.arange(offsets[i], offsets[i] + n, dtype='int32')])
    f['objects/cell_inds'] = cell_inds_arr

    # props
    f['props/count'] = np.array([len(stream_defs)], dtype='int32')
    f['props/axis_ratio_ab'] = np.array([2.5, 3.0], dtype='float32')
    f['props/axis_ratio_ac'] = np.array([4.0, 5.5], dtype='float32')

# 生成合成 cutout 文件
cutout_file = os.path.join(stream_dir, f"cutout_snap{snapNum:03d}_sh{subhalo_id}.hdf5")

n_total = len(all_positions)
# 生成内部能量和电子丰度（反推温度）
# T = (gamma-1) * u * 1e10 * mu * m_p / k_B
# u = T * k_B / ((gamma-1) * 1e10 * mu * m_p)
k_B = 1.38064852e-16
m_p = 1.6726219e-24
gamma = 5.0 / 3.0
XH = 0.76
ne_all = np.random.uniform(0.01, 0.1, n_total)
mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * ne_all)
u_all = all_temps * k_B / ((gamma - 1.0) * 1e10 * mu * m_p)

with h5py.File(cutout_file, 'w') as f:
    pt0 = f.create_group('PartType0')
    pt0['Coordinates'] = all_positions
    pt0['Masses'] = all_masses.astype('float32')
    pt0['InternalEnergy'] = u_all.astype('float32')
    pt0['ElectronAbundance'] = ne_all.astype('float32')

print(f"Synthetic data written to:\n  {stream_file}\n  {cutout_file}")
print(f"Streams: {len(stream_defs)}, Total cells: {n_total}")
for i, sdef in enumerate(stream_defs):
    print(f"  {sdef['name']}: {sdef['n_cells']} cells")
