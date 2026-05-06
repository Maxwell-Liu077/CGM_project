"""
冷流 3D 可视化脚本

Usage:
    python plot_cold_streams_3d.py --subhalo 59075 --snap 33
    python plot_cold_streams_3d.py --subhalo 59075 --snap 33 --output_dir ./figures
"""

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
import matplotlib.cm as cm
from scipy.spatial import ConvexHull


# ============================================================================
# 绘图样式配置 (与 Visual.ipynb 保持一致)
# ============================================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 500


# ============================================================================
# 工具函数
# ============================================================================
def periodic_displacement(pos, center, BoxSize):
    """周期性边界条件下的位移"""
    return (pos - center + 0.5 * BoxSize) % BoxSize - 0.5 * BoxSize


def compute_temperature(u, ne, XH=0.76):
    """由内能和电子丰度计算气体温度 (K)"""
    k_B = 1.38064852e-16
    m_p = 1.6726219e-24
    gamma = 5.0 / 3.0
    u_cgs = u * 1e10
    mu = 4.0 / (1.0 + 3.0 * XH + 4.0 * XH * ne)
    T = (gamma - 1.0) * u_cgs * mu * m_p / k_B
    return T


def load_stream_data(subhalo_id, snapNum, cutout_dir, stream_dir):
    """读取冷流结果和原始 cutout 数据"""
    cutout_file = os.path.join(cutout_dir, f"cutout_snap{snapNum:03d}_sh{subhalo_id}.hdf5")
    stream_file = os.path.join(stream_dir, f"cold_streams_snap{snapNum:03d}_sh{subhalo_id}.hdf5")

    if not os.path.exists(cutout_file):
        raise FileNotFoundError(f"Cutout file not found: {cutout_file}")
    if not os.path.exists(stream_file):
        raise FileNotFoundError(f"Stream file not found: {stream_file}")

    data = {}
    with h5py.File(stream_file, 'r') as f:
        count = f['objects/count'][()]
        if count == 0:
            return None

        data['count'] = count
        data['cell_inds'] = f['objects/cell_inds'][()]
        data['lengths'] = f['objects/lengths'][()]
        data['offsets'] = f['objects/offsets'][()]
        data['center_pos'] = f.attrs['CenterPos']
        data['r200c'] = f.attrs['R200c']
        # 读取形态学轴比 (可能不存在)
        if 'props/axis_ratio_ab' in f:
            data['ab_ratios'] = f['props/axis_ratio_ab'][()]
        else:
            data['ab_ratios'] = np.zeros(count)
        if 'props/axis_ratio_ac' in f:
            data['ac_ratios'] = f['props/axis_ratio_ac'][()]
        else:
            data['ac_ratios'] = np.zeros(count)

    with h5py.File(cutout_file, 'r') as f:
        pos_all = f['PartType0/Coordinates'][()]
        mass_all = f['PartType0/Masses'][()]

        if 'PartType0/InternalEnergy' in f and 'PartType0/ElectronAbundance' in f:
            u_all = f['PartType0/InternalEnergy'][()]
            ne_all = f['PartType0/ElectronAbundance'][()]
            temp_all = compute_temperature(u_all, ne_all)
        else:
            raise KeyError("Cutout 中缺乏计算温度所需的 InternalEnergy / ElectronAbundance 字段。")

        data['stream_pos'] = pos_all[data['cell_inds']]
        data['stream_mass'] = mass_all[data['cell_inds']]
        data['stream_temp'] = temp_all[data['cell_inds']]

    return data


# ============================================================================
# 3D 冷流可视化
# ============================================================================
def plot_cold_streams_3d(data, snapNum, subhalo_id, boxSize=35000.0,
                         save_path=None, show_hull=True, hull_alpha=0.08):
    """
    生成 3D 冷流散点图 (温度着色 + virial 半径球壳)
    与 Visual.ipynb 中 plot_cold_streams 一致

    Parameters
    ----------
    show_hull : bool
        是否用浅蓝色半透明凸包面包裹冷流结构
    hull_alpha : float
        包裹面的透明度 (0~1)
    """
    dx = periodic_displacement(data['stream_pos'], data['center_pos'], boxSize)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    norm = LogNorm(vmin=5e3, vmax=2.5e5)
    cmap = plt.get_cmap('magma')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i in range(data['count']):
        loc = slice(data['offsets'][i], data['offsets'][i] + data['lengths'][i])
        clump_dx = dx[loc]
        clump_temp = data['stream_temp'][loc]

        ab = data['ab_ratios'][i]
        ac = data['ac_ratios'][i]
        label_str = f"Stream {i+1} (a/b={ab:.1f}, a/c={ac:.1f})" if ab > 0 and ac > 0 else f"Stream {i+1} (a/b={ab:.1f})" if ab > 0 else f"Stream {i+1}"

        ax.scatter(
            clump_dx[:, 0], clump_dx[:, 1], clump_dx[:, 2],
            s=1.2, alpha=0.5, c=clump_temp,
            cmap=cmap, norm=norm,
            label=label_str, edgecolors='none', rasterized=True,
        )

    # 每条 Stream 分别绘制凸包包裹面
    if show_hull:
        for i in range(data['count']):
            loc = slice(data['offsets'][i], data['offsets'][i] + data['lengths'][i])
            pts = dx[loc]
            if len(pts) >= 4:
                try:
                    hull = ConvexHull(pts)
                    ax.plot_trisurf(
                        pts[:, 0], pts[:, 1], pts[:, 2],
                        triangles=hull.simplices,
                        color='lightblue',
                        alpha=hull_alpha,
                        edgecolor='lightblue',
                        linewidth=0.5,
                        linestyle='--',
                        antialiased=True,
                    )
                except Exception:
                    pass

    rvir = data['r200c']
    shells = [
        {'r': 0.15 * rvir, 'color': 'darkred',  'ls': '-', 'alpha': 0.15, 'label': r'$0.15 R_{\rm vir}$'},
        {'r': rvir,        'color': 'darkblue', 'ls': '-', 'alpha': 0.10, 'label': r'$R_{\rm vir}$'},
    ]
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    for shell in shells:
        x_s = shell['r'] * np.cos(u) * np.sin(v)
        y_s = shell['r'] * np.sin(u) * np.sin(v)
        z_s = shell['r'] * np.cos(v)
        ax.plot_wireframe(
            x_s, y_s, z_s,
            color=shell['color'], alpha=shell['alpha'],
            linewidth=0.6, linestyle=shell['ls'], label=shell['label'],
        )

    limit = rvir
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])

    ax.set_xlabel('X (kpc/h)', labelpad=10)
    ax.set_ylabel('Y (kpc/h)', labelpad=10)
    ax.set_zlabel('Z (kpc/h)', labelpad=10)
    ax.set_title(f"Cold Streams 3D | Snap: {snapNum} | Subhalo: {subhalo_id}", pad=20, weight='bold')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(color='gray', linestyle=':', alpha=0.2)

    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
    cbar.set_label(r'Gas Temperature $T$ [K]', rotation=270, labelpad=25)
    cbar.locator = LogLocator(base=10.0, numticks=5)
    cbar.formatter = LogFormatterSciNotation(10.0)
    cbar.update_ticks()

    ax.legend(loc='upper left', fontsize=8, framealpha=0.8)

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"3D figure saved to: {save_path}")
    else:
        plt.show()

    return fig, ax


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="冷流 3D 可视化工具")
    parser.add_argument("--subhalo", type=int, default=59075, help="Subhalo ID (default: 59075)")
    parser.add_argument("--snap", type=int, default=33, help="Snapshot number (default: 33)")
    parser.add_argument("--cutout_dir", type=str,
                        default="/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/cutouts",
                        help="Cutout 文件目录")
    parser.add_argument("--stream_dir", type=str,
                        default="/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/streams",
                        help="冷流结果文件目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出图片目录 (默认: 当前目录)")
    parser.add_argument("--box_size", type=float, default=35000.0, help="模拟盒子大小 (kpc/h)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "."
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data for Subhalo {args.subhalo}, Snap {args.snap}...")
    data = load_stream_data(args.subhalo, args.snap, args.cutout_dir, args.stream_dir)

    if data is None:
        print(f"Subhalo {args.subhalo} (Snap {args.snap}) 没有识别到冷流。")
        exit(0)

    print(f"Found {data['count']} cold stream(s), {len(data['cell_inds'])} gas cells total.")

    save_path = os.path.join(args.output_dir, f"cold_streams_3d_sh{args.subhalo}_snap{args.snap}.pdf")
    plot_cold_streams_3d(data, args.snap, args.subhalo, args.box_size, save_path=save_path)

    print("Done.")
