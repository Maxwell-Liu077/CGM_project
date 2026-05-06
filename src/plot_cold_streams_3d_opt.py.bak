"""
冷流 3D 可视化 — Scientific Visualization Skill 优化版

优化点:
  - 期刊级排版 (double-column 尺寸, Arial sans-serif 字体)
  - 色盲友好配色 (magma 已是 perceptually uniform, 保留)
  - 更高分辨率的 virial 球壳 (40j×20j vs 20j×10j)
  - 透明凸包 + 线框风格, 减少视觉遮挡
  - 导出 PDF + PNG 双格式, 300 DPI

Usage:
    python plot_cold_streams_3d_opt.py --subhalo 59075 --snap 33
    python plot_cold_streams_3d_opt.py --demo  # 使用合成数据演示
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

# 保留天文领域常用的 serif 字体用于数学公式
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

    all_pos, all_temp, all_offsets, all_lengths = [], [], [], []
    offset = 0

    # Stream 1: 沿 XZ 平面进入的准直流
    t = np.linspace(0.1, 1.0, n_cells_per_stream)
    stream1_x = -rvir * 0.8 + 1.2 * rvir * t + np.random.randn(n_cells_per_stream) * 40
    stream1_y = np.random.randn(n_cells_per_stream) * 60 * (1 - 0.5 * t)
    stream1_z = rvir * 0.3 * np.sin(2 * np.pi * t) + np.random.randn(n_cells_per_stream) * 40
    temp1 = 5e3 + 1.5e5 * np.random.beta(2, 5, n_cells_per_stream)

    # Stream 2: 从另一侧螺旋进入
    theta = np.linspace(0, 3 * np.pi, n_cells_per_stream)
    r2 = rvir * (1.0 - 0.6 * t)
    stream2_x = r2 * np.cos(theta) + np.random.randn(n_cells_per_stream) * 50
    stream2_y = r2 * np.sin(theta) + np.random.randn(n_cells_per_stream) * 50
    stream2_z = rvir * (0.5 - t) * 0.4 + np.random.randn(n_cells_per_stream) * 35
    temp2 = 5e3 + 1.8e5 * np.random.beta(2, 4, n_cells_per_stream)

    all_pos = np.vstack([
        np.column_stack([stream1_x, stream1_y, stream1_z]),
        np.column_stack([stream2_x, stream2_y, stream2_z]),
    ])
    all_temp = np.concatenate([temp1, temp2])

    data['cell_inds'] = np.arange(len(all_pos))
    data['lengths'] = np.array([n_cells_per_stream, n_cells_per_stream])
    data['offsets'] = np.array([0, n_cells_per_stream])
    data['stream_pos'] = all_pos + center
    data['stream_mass'] = np.ones(len(all_pos)) * 1e6
    data['stream_temp'] = all_temp

    return data


# ============================================================================
# 3D 优化主函数
# ============================================================================
def plot_cold_streams_3d(data, snapNum, subhalo_id, boxSize=35000.0,
                         save_path=None, show_hull=True, hull_alpha=0.06):
    """
    期刊级 3D 冷流散点图

    优化:
      - 双栏尺寸 (7.08 inch / 180 mm)
      - 高分辨率球壳 (40j×20j)
      - 浅蓝色半透明凸包 (降低 alpha, 减少遮挡)
      - Okabe-Ito 色盲安全 legend
      - 矢量 PDF + PNG 双导出
    """
    dx = periodic_displacement(data['stream_pos'], data['center_pos'], boxSize)

    # 双栏宽度, 高度按黄金比例
    fig = plt.figure(figsize=(7.08, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    norm = LogNorm(vmin=5e3, vmax=2.5e5)
    cmap = plt.get_cmap('magma')

    okabe_ito_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
                        '#0072B2', '#D55E00', '#CC79A7']

    for i in range(data['count']):
        loc = slice(data['offsets'][i], data['offsets'][i] + data['lengths'][i])
        clump_dx = dx[loc]
        clump_temp = data['stream_temp'][loc]
        color = okabe_ito_colors[i % len(okabe_ito_colors)]

        ab = data['ab_ratios'][i]
        ac = data['ac_ratios'][i]
        if ab > 0 and ac > 0:
            label_str = f"Stream {i+1} (a/b={ab:.2f}, a/c={ac:.2f})"
        elif ab > 0:
            label_str = f"Stream {i+1} (a/b={ab:.2f})"
        else:
            label_str = f"Stream {i+1}"

        ax.scatter(
            clump_dx[:, 0], clump_dx[:, 1], clump_dx[:, 2],
            s=1.5, alpha=0.6, c=clump_temp,
            cmap=cmap, norm=norm,
            label=label_str, edgecolors='none', rasterized=True,
        )

    # 凸包包裹面 — 降低透明度, 避免遮挡内部结构
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
                        color='lightblue', alpha=hull_alpha,
                        edgecolor='steelblue', linewidth=0.4,
                        antialiased=True,
                    )
                except Exception:
                    pass

    rvir = data['r200c']

    # 高分辨率 virial 球壳
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    shells = [
        {'r': 0.15 * rvir, 'color': '#D55E00', 'alpha': 0.15, 'label': r'$0.15\,R_{\rm vir}$'},
        {'r': rvir,        'color': '#0072B2', 'alpha': 0.10, 'label': r'$R_{\rm vir}$'},
    ]
    for shell in shells:
        x_s = shell['r'] * np.cos(u) * np.sin(v)
        y_s = shell['r'] * np.sin(u) * np.sin(v)
        z_s = shell['r'] * np.cos(v)
        ax.plot_wireframe(
            x_s, y_s, z_s,
            color=shell['color'], alpha=shell['alpha'],
            linewidth=0.5, label=shell['label'],
        )

    limit = rvir
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])

    ax.set_xlabel(r'X (kpc/$h$)', labelpad=8, fontsize=9)
    ax.set_ylabel(r'Y (kpc/$h$)', labelpad=8, fontsize=9)
    ax.set_zlabel(r'Z (kpc/$h$)', labelpad=8, fontsize=9)

    ax.tick_params(labelsize=7)

    # 清除面板背景
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(color='lightgray', linestyle=':', alpha=0.3, linewidth=0.5)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.08, shrink=0.7)
    cbar.set_label(r'Gas Temperature $T$ (K)', rotation=270, labelpad=15, fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    cbar.locator = LogLocator(base=10.0, numticks=5)
    cbar.formatter = LogFormatterSciNotation(10.0)
    cbar.update_ticks()

    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.85, frameon=True,
              edgecolor='lightgray')

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
    parser = argparse.ArgumentParser(description="冷流 3D 可视化 (优化版)")
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
        save_path = os.path.join(output_dir, "cold_streams_3d_demo.pdf")
        plot_cold_streams_3d(data, snapNum=33, subhalo_id=0, boxSize=args.box_size, save_path=save_path)
    else:
        print(f"Loading data for Subhalo {args.subhalo}, Snap {args.snap}...")
        data = load_stream_data(args.subhalo, args.snap, args.cutout_dir, args.stream_dir)
        if data is None:
            print(f"No cold streams found for Subhalo {args.subhalo} (Snap {args.snap}).")
            exit(0)
        print(f"Found {data['count']} cold stream(s), {len(data['cell_inds'])} gas cells total.")
        save_path = os.path.join(output_dir, f"cold_streams_3d_sh{args.subhalo}_snap{args.snap}.pdf")
        plot_cold_streams_3d(data, args.snap, args.subhalo, args.box_size, save_path=save_path)

    print("Done.")
