"""
冷流 2D 投影可视化脚本

Usage:
    python plot_cold_streams_2d.py --subhalo 59075 --snap 33
    python plot_cold_streams_2d.py --subhalo 59075 --snap 33 --output_dir ./figures
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
# 2D 投影图 (XY / XZ / YZ 三个面)
# ============================================================================
def plot_cold_streams_2d(data, snapNum, subhalo_id, boxSize=35000.0, save_path=None,
                         show_hull=True, hull_linewidth=0.8):
    """
    生成 2D 投影图: XY, XZ, YZ 三个平面
    每个面板中气体单元按温度着色，叠加 virial 半径圆环

    Parameters
    ----------
    show_hull : bool
        是否用浅蓝色虚线凸包轮廓包裹冷流结构
    hull_linewidth : float
        包裹线的线宽
    """
    dx = periodic_displacement(data['stream_pos'], data['center_pos'], boxSize)
    rvir = data['r200c']

    norm = LogNorm(vmin=5e3, vmax=2.5e5)
    cmap = plt.get_cmap('magma')

    planes = [
        {'axes': (0, 1), 'label_x': 'X', 'label_y': 'Y', 'title': 'XY Plane'},
        {'axes': (0, 2), 'label_x': 'X', 'label_y': 'Z', 'title': 'XZ Plane'},
        {'axes': (1, 2), 'label_x': 'Y', 'label_y': 'Z', 'title': 'YZ Plane'},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, plane in zip(axes, planes):
        ix, iy = plane['axes']

        for i in range(data['count']):
            loc = slice(data['offsets'][i], data['offsets'][i] + data['lengths'][i])
            clump_temp = data['stream_temp'][loc]

            ab = data['ab_ratios'][i]
            ac = data['ac_ratios'][i]
            label_str = f"Stream {i+1} (a/b={ab:.1f}, a/c={ac:.1f})" if ab > 0 and ac > 0 else f"Stream {i+1} (a/b={ab:.1f})" if ab > 0 else f"Stream {i+1}"

            ax.scatter(
                dx[loc, ix], dx[loc, iy],
                s=1.2, alpha=0.5, c=clump_temp,
                cmap=cmap, norm=norm,
                label=label_str, edgecolors='none', rasterized=True,
            )

            # 每条 Stream 分别绘制凸包包裹线 (2D 投影)
            if show_hull and len(dx[loc]) >= 3:
                try:
                    pts_2d = dx[loc, [ix, iy]]
                    hull2d = ConvexHull(pts_2d)
                    hull_pts = pts_2d[hull2d.vertices]
                    hull_closed = np.vstack([hull_pts, hull_pts[0]])
                    ax.plot(hull_closed[:, 0], hull_closed[:, 1],
                            color='lightblue', linestyle='--', linewidth=hull_linewidth,
                            alpha=0.85, zorder=3)
                except Exception:
                    pass

        # virial 半径圆环
        circle_vir = plt.Circle((0, 0), rvir, color='darkblue', fill=False,
                                linestyle='-', linewidth=0.8, alpha=0.4, label=r'$R_{\rm vir}$')
        circle_inner = plt.Circle((0, 0), 0.15 * rvir, color='darkred', fill=False,
                                  linestyle='-', linewidth=0.8, alpha=0.4, label=r'$0.15 R_{\rm vir}$')
        ax.add_patch(circle_vir)
        ax.add_patch(circle_inner)

        ax.set_xlim(-rvir, rvir)
        ax.set_ylim(-rvir, rvir)
        ax.set_xlabel(f"{plane['label_x']} (kpc/h)")
        ax.set_ylabel(f"{plane['label_y']} (kpc/h)")
        ax.set_title(plane['title'], weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.3)

    # 只在第一个面板显示 legend
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:data['count']], labels[:data['count']],
                   loc='upper left', fontsize=7, framealpha=0.8)

    # 共享 colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04, shrink=0.8)
    cbar.set_label(r'Gas Temperature $T$ [K]', rotation=270, labelpad=20)
    cbar.locator = LogLocator(base=10.0, numticks=5)
    cbar.formatter = LogFormatterSciNotation(10.0)
    cbar.update_ticks()

    fig.suptitle(f"Cold Streams 2D Projection | Snap: {snapNum} | Subhalo: {subhalo_id}",
                 y=1.02, weight='bold', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"2D projection saved to: {save_path}")
    else:
        plt.show()

    return fig, axes


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="冷流 2D 投影可视化工具")
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

    save_path = os.path.join(args.output_dir, f"cold_streams_2d_sh{args.subhalo}_snap{args.snap}.pdf")
    plot_cold_streams_2d(data, args.snap, args.subhalo, args.box_size, save_path=save_path)

    print("Done.")
