"""
冷流 2D 投影可视化 — Scientific Visualization Skill 优化版

优化点:
  - 双栏宽度 (7.08 inch), 面板间距优化
  - 期刊级排版: sans-serif 字体, 统一字号
  - 色盲安全配色: Okabe-Ito 用于 legend, magma 用于温度映射
  - 凸包轮廓使用 steelblue (而非 lightblue), 提高对比度
  - 去除多余网格线, 减少 chart junk
  - 导出 PDF + PNG 双格式

Usage:
    python plot_cold_streams_2d_opt.py --subhalo 59075 --snap 33
    python plot_cold_streams_2d_opt.py --demo  # 使用合成数据演示
"""

import os
import sys
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
import matplotlib.cm as cm
from scipy.spatial import ConvexHull

# ── 引入 Scientific Visualization Skill 的辅助模块 ──
SKILL_DIR = os.path.expanduser("~/.claude/skills/scientific-visualization")
sys.path.insert(0, os.path.join(SKILL_DIR, "scripts"))
from style_presets import apply_publication_style
from figure_export import save_publication_figure

# ── 期刊级样式 ──
apply_publication_style('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['image.cmap'] = 'magma'
plt.rcParams['mathtext.fontset'] = 'dejavusans'


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
        data['ab_ratios'] = f['props/axis_ratio_ab'][()] if 'props/axis_ratio_ab' in f else np.zeros(count)
        data['ac_ratios'] = f['props/axis_ratio_ac'][()] if 'props/axis_ratio_ac' in f else np.zeros(count)

    with h5py.File(cutout_file, 'r') as f:
        pos_all = f['PartType0/Coordinates'][()]
        mass_all = f['PartType0/Masses'][()]
        if 'PartType0/InternalEnergy' not in f or 'PartType0/ElectronAbundance' not in f:
            raise KeyError("Cutout 中缺乏计算温度所需的 InternalEnergy / ElectronAbundance 字段。")
        u_all = f['PartType0/InternalEnergy'][()]
        ne_all = f['PartType0/ElectronAbundance'][()]
        temp_all = compute_temperature(u_all, ne_all)

        data['stream_pos'] = pos_all[data['cell_inds']]
        data['stream_mass'] = mass_all[data['cell_inds']]
        data['stream_temp'] = temp_all[data['cell_inds']]

    return data


def generate_synthetic_data(n_streams=2, n_cells_per_stream=2000, rvir=800.0, boxSize=35000.0):
    """生成合成冷流数据用于演示"""
    np.random.seed(42)
    center = np.array([boxSize / 2, boxSize / 2, boxSize / 2])
    data = {'count': n_streams, 'r200c': rvir, 'center_pos': center}
    data['ab_ratios'] = np.array([0.3, 0.5])
    data['ac_ratios'] = np.array([0.2, 0.4])

    t = np.linspace(0.1, 1.0, n_cells_per_stream)
    stream1_x = -rvir * 0.8 + 1.2 * rvir * t + np.random.randn(n_cells_per_stream) * 40
    stream1_y = np.random.randn(n_cells_per_stream) * 60 * (1 - 0.5 * t)
    stream1_z = rvir * 0.3 * np.sin(2 * np.pi * t) + np.random.randn(n_cells_per_stream) * 40

    theta = np.linspace(0, 3 * np.pi, n_cells_per_stream)
    r2 = rvir * (1.0 - 0.6 * t)
    stream2_x = r2 * np.cos(theta) + np.random.randn(n_cells_per_stream) * 50
    stream2_y = r2 * np.sin(theta) + np.random.randn(n_cells_per_stream) * 50
    stream2_z = rvir * (0.5 - t) * 0.4 + np.random.randn(n_cells_per_stream) * 35

    all_pos = np.vstack([
        np.column_stack([stream1_x, stream1_y, stream1_z]),
        np.column_stack([stream2_x, stream2_y, stream2_z]),
    ])
    temp1 = 5e3 + 1.5e5 * np.random.beta(2, 5, n_cells_per_stream)
    temp2 = 5e3 + 1.8e5 * np.random.beta(2, 4, n_cells_per_stream)
    all_temp = np.concatenate([temp1, temp2])

    data['cell_inds'] = np.arange(len(all_pos))
    data['lengths'] = np.array([n_cells_per_stream, n_cells_per_stream])
    data['offsets'] = np.array([0, n_cells_per_stream])
    data['stream_pos'] = all_pos + center
    data['stream_mass'] = np.ones(len(all_pos)) * 1e6
    data['stream_temp'] = all_temp

    return data


# ============================================================================
# 2D 优化主函数
# ============================================================================
def plot_cold_streams_2d(data, snapNum, subhalo_id, boxSize=35000.0,
                         save_path=None, show_hull=True, hull_linewidth=1.0):
    """
    期刊级 2D 冷流投影图 (XY / XZ / YZ)

    优化:
      - 双栏宽度 (7.08 inch), 3 面板均匀分布
      - 去除顶部/右侧 spines, 减少 chart junk
      - 凸包使用 steelblue 实色 (对比度优于 lightblue)
      - Virial 环使用色盲安全色 (blue/orange)
      - 共享 colorbar 统一比例
      - 面板标签 (A/B/C) 用于论文引用
    """
    dx = periodic_displacement(data['stream_pos'], data['center_pos'], boxSize)
    rvir = data['r200c']

    norm = LogNorm(vmin=5e3, vmax=2.5e5)
    cmap = plt.get_cmap('magma')

    okabe_ito_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
                        '#0072B2', '#D55E00', '#CC79A7']

    planes = [
        {'axes': (0, 1), 'label_x': r'X (kpc/$h$)', 'label_y': r'Y (kpc/$h$)', 'panel': 'A'},
        {'axes': (0, 2), 'label_x': r'X (kpc/$h$)', 'label_y': r'Z (kpc/$h$)', 'panel': 'B'},
        {'axes': (1, 2), 'label_x': r'Y (kpc/$h$)', 'label_y': r'Z (kpc/$h$)', 'panel': 'C'},
    ]

    # 双栏宽度, 高度约 1:1 比例 (禁用 constrained_layout 以使用 subplots_adjust)
    plt.rcParams['figure.constrained_layout.use'] = False
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.4))
    fig.subplots_adjust(wspace=0.25, bottom=0.15, top=0.88)

    for ax, plane in zip(axes, planes):
        ix, iy = plane['axes']

        for i in range(data['count']):
            loc = slice(data['offsets'][i], data['offsets'][i] + data['lengths'][i])
            clump_temp = data['stream_temp'][loc]
            color = okabe_ito_colors[i % len(okabe_ito_colors)]

            label_str = f"Stream {i+1}"

            ax.scatter(
                dx[loc, ix], dx[loc, iy],
                s=1.5, alpha=0.55, c=clump_temp,
                cmap=cmap, norm=norm,
                label=label_str, edgecolors='none', rasterized=True,
            )

            # 凸包包裹线
            if show_hull and len(dx[loc]) >= 3:
                try:
                    pts_2d = dx[loc, [ix, iy]]
                    hull2d = ConvexHull(pts_2d)
                    hull_pts = pts_2d[hull2d.vertices]
                    hull_closed = np.vstack([hull_pts, hull_pts[0]])
                    ax.plot(hull_closed[:, 0], hull_closed[:, 1],
                            color='#4682B4', linestyle='--', linewidth=hull_linewidth,
                            alpha=0.7, zorder=3)
                except Exception:
                    pass

        # Virial 半径环 — 使用色盲安全色
        circle_vir = plt.Circle((0, 0), rvir, color='#0072B2', fill=False,
                                linestyle='-', linewidth=0.8, alpha=0.35, zorder=4)
        circle_inner = plt.Circle((0, 0), 0.15 * rvir, color='#D55E00', fill=False,
                                  linestyle='-', linewidth=0.8, alpha=0.35, zorder=4)
        ax.add_patch(circle_vir)
        ax.add_patch(circle_inner)

        ax.set_xlim(-rvir, rvir)
        ax.set_ylim(-rvir, rvir)
        ax.set_xlabel(plane['label_x'], fontsize=8)
        ax.set_ylabel(plane['label_y'], fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

        # 去除顶部/右侧 spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 面板标签
        ax.text(-0.08, 1.02, plane['panel'], transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top')

    # Legend (只含 stream 标签)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=okabe_ito_colors[i % len(okabe_ito_colors)],
                   markersize=5, alpha=0.7, label=f"Stream {i+1}")
        for i in range(data['count'])
    ]
    legend_elements.append(plt.Line2D([0], [0], color='#0072B2', linewidth=1.2, label=r'$R_{\rm vir}$'))
    legend_elements.append(plt.Line2D([0], [0], color='#D55E00', linewidth=1.2, label=r'$0.15\,R_{\rm vir}$'))
    axes[0].legend(handles=legend_elements, loc='upper left', fontsize=7,
                   framealpha=0.85, frameon=True, edgecolor='lightgray')

    # 共享 colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04, shrink=0.85)
    cbar.set_label(r'Gas Temperature $T$ (K)', rotation=270, labelpad=12, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.locator = LogLocator(base=10.0, numticks=5)
    cbar.formatter = LogFormatterSciNotation(10.0)
    cbar.update_ticks()

    # 导出
    if save_path:
        base = os.path.splitext(save_path)[0]
        save_publication_figure(fig, base, formats=['pdf', 'png'], dpi=300)
    else:
        plt.show()

    plt.close(fig)
    return fig


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="冷流 2D 投影可视化 (优化版)")
    parser.add_argument("--subhalo", type=int, default=59075)
    parser.add_argument("--snap", type=int, default=33)
    parser.add_argument("--cutout_dir", type=str,
                        default="/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/cutouts")
    parser.add_argument("--stream_dir", type=str,
                        default="/public/home/zju_visitor/LiuYuanhao/cold_stream/simulation_results/streams")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--box_size", type=float, default=35000.0)
    parser.add_argument("--demo", action="store_true", help="使用合成数据演示")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(output_dir, exist_ok=True)

    if args.demo:
        print("Generating synthetic cold stream data for demo...")
        data = generate_synthetic_data()
        print(f"  {data['count']} streams, {len(data['cell_inds'])} cells total")
        save_path = os.path.join(output_dir, "cold_streams_2d_demo.pdf")
        plot_cold_streams_2d(data, snapNum=33, subhalo_id=0, boxSize=args.box_size, save_path=save_path)
    else:
        print(f"Loading data for Subhalo {args.subhalo}, Snap {args.snap}...")
        data = load_stream_data(args.subhalo, args.snap, args.cutout_dir, args.stream_dir)
        if data is None:
            print(f"No cold streams found for Subhalo {args.subhalo} (Snap {args.snap}).")
            exit(0)
        print(f"Found {data['count']} cold stream(s), {len(data['cell_inds'])} gas cells total.")
        save_path = os.path.join(output_dir, f"cold_streams_2d_sh{args.subhalo}_snap{args.snap}.pdf")
        plot_cold_streams_2d(data, args.snap, args.subhalo, args.box_size, save_path=save_path)

    print("Done.")
