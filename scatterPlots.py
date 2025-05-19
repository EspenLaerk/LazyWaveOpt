# Generate the full script to produce all scatter plots and corresponding density variants

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import os
from config import MIN_BEND_RADIUS_LIMIT, MIN_SEABED_CLEARANCE, MIN_SURFACE_CLEARANCE
import re
import matplotlib.ticker as mticker
from scipy.stats import binned_statistic_2d
from scipy.spatial import ConvexHull
import numpy as np

def save_or_show(fig, save_path):
    plt.tight_layout()
    save_path = os.path.join('plots', os.path.basename(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.close(fig)

# Load the data
df = pd.read_csv(os.path.join("output", "solutions_filtered.csv"))
df_valid = df[df['is_valid']].copy()

# Colour definitions for valid/invalid
valid_colour = '#008000'   # More saturated green
invalid_colour = '#FFB3B3' # Light red

def to_camel_case(s):
    parts = s.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('/', ' ').replace('-', ' ').replace('.', ' ').split()
    return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])

def to_camel_case_label(label):
    # Remove units in parentheses or brackets, and special characters
    label = re.sub(r'\(.*?\)', '', label)
    label = re.sub(r'\[.*?\]', '', label)
    label = re.sub(r'[^a-zA-Z0-9 ]', '', label)
    parts = label.strip().split()
    return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:]) if parts else ''

def make_plot_filename_from_labels(xlabel, ylabel, suffix=None, prefix=None):
    base = f"{to_camel_case_label(xlabel)}_vs_{to_camel_case_label(ylabel)}"
    if suffix:
        base = f"{base}_{suffix}"
    if prefix:
        return f"{prefix}-{base}"
    return base

# Scatter plot with valid/invalid
def scatter_plot(x, y, xlabel, ylabel, title, hlines=None, vlines=None, prefix=None):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    valid = df['is_valid']
    # Plot invalid first (background)
    plt.scatter(df.loc[~valid, x], df.loc[~valid, y], c=invalid_colour, edgecolors='none', s=30, alpha=1.0, marker='o', label='Invalid')
    # Plot valid on top (foreground)
    plt.scatter(df.loc[valid, x], df.loc[valid, y], c=valid_colour, edgecolors='none', s=40, alpha=0.9, marker='o', label='Valid')
    if hlines:
        for val, label in hlines:
            plt.axhline(val, linestyle='--', color='grey')
            plt.text(df[x].min(), val + 1, label, fontsize=8, color='grey')
    if vlines:
        for val, label in vlines:
            plt.axvline(val, linestyle='--', color='grey')
            plt.text(val + 1, df[y].min(), label, fontsize=8, color='grey', rotation=90)
    # No legend, no title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    # Ensure integer ticks for Number of Modules
    if xlabel.lower().startswith('number of modules'):
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if ylabel.lower().startswith('number of modules'):
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_aspect('auto')
    plt.tight_layout()
    save_or_show(plt.gcf(), make_plot_filename_from_labels(xlabel, ylabel, prefix=prefix))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    return xlim, ylim

# Single-colour neutral scatter plot (valid only)
def scatter_neutral(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.scatter(df_valid[x], df_valid[y], c=valid_colour, edgecolors='none', s=40, alpha=0.9, marker='o')
    # No title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    # Ensure integer ticks for Number of Modules
    if xlabel.lower().startswith('number of modules'):
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if ylabel.lower().startswith('number of modules'):
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_or_show(plt.gcf(), make_plot_filename_from_labels(xlabel, ylabel, 'neutral'))

# Density plot for valid/invalid, matching scatter aspect/limits
def density_plot(x, y, xlabel, ylabel, title, xlim=None, ylim=None, prefix=None):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    valid = df['is_valid']
    # Use config limits for clipping
    clip_x = (MIN_BEND_RADIUS_LIMIT, df[x].max()) if x == 'min_bend_radius' else \
             (MIN_SEABED_CLEARANCE, df[x].max()) if x == 'min_seabed_clearance' else \
             (MIN_SURFACE_CLEARANCE, df[x].max()) if x == 'min_surface_clearance' else \
             (df[x].min(), df[x].max())
    clip_y = (MIN_BEND_RADIUS_LIMIT, df[y].max()) if y == 'min_bend_radius' else \
             (MIN_SEABED_CLEARANCE, df[y].max()) if y == 'min_seabed_clearance' else \
             (MIN_SURFACE_CLEARANCE, df[y].max()) if y == 'min_surface_clearance' else \
             (df[y].min(), df[y].max())
    clip = (clip_x, clip_y)
    # Valid (filled)
    sns.kdeplot(
        x=df.loc[valid, x],
        y=df.loc[valid, y],
        color=valid_colour,
        fill=True,
        thresh=0.05,
        levels=100,
        alpha=0.7,
        label='Valid',
        clip=clip,
        bw_adjust=0.7,
        warn_singular=False
    )
    # Valid (outline)
    sns.kdeplot(
        x=df.loc[valid, x],
        y=df.loc[valid, y],
        color='black',
        fill=False,
        thresh=0.05,
        levels=5,
        linewidths=1.2,
        clip=clip,
        bw_adjust=0.7,
        warn_singular=False
    )
    # Invalid
    sns.kdeplot(
        x=df.loc[~valid, x],
        y=df.loc[~valid, y],
        color=invalid_colour,
        fill=True,
        thresh=0.05,
        levels=100,
        alpha=1.0,
        label='Invalid',
        bw_adjust=0.7,
        clip=((df[x].min(), df[x].max()), (df[y].min(), df[y].max())),
        warn_singular=False
    )
    # Add black dashed lines at the clipping thresholds
    if x == 'min_bend_radius':
        plt.axvline(MIN_BEND_RADIUS_LIMIT, color='black', linewidth=1.5, linestyle='--')
    if y == 'min_bend_radius':
        plt.axhline(MIN_BEND_RADIUS_LIMIT, color='black', linewidth=1.5, linestyle='--')
    if x == 'min_seabed_clearance':
        plt.axvline(MIN_SEABED_CLEARANCE, color='black', linewidth=1.5, linestyle='--')
    if y == 'min_seabed_clearance':
        plt.axhline(MIN_SEABED_CLEARANCE, color='black', linewidth=1.5, linestyle='--')
    if x == 'min_surface_clearance':
        plt.axvline(MIN_SURFACE_CLEARANCE, color='black', linewidth=1.5, linestyle='--')
    if y == 'min_surface_clearance':
        plt.axhline(MIN_SURFACE_CLEARANCE, color='black', linewidth=1.5, linestyle='--')
    # No title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    # Ensure integer ticks for Number of Modules
    if xlabel.lower().startswith('number of modules'):
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if ylabel.lower().startswith('number of modules'):
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_aspect('auto')
    # No legend
    plt.tight_layout()
    save_or_show(plt.gcf(), make_plot_filename_from_labels(xlabel, ylabel, 'density', prefix=prefix))

def line_plot_num_modules_vs_fitness():
    means = df_valid.groupby('num_modules')['fitness'].mean()
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.plot(means.index, means.values, color=valid_colour, linewidth=2, marker='o')
    plt.xlabel('Number of Modules')
    plt.ylabel('Fitness')
    plt.grid(True)
    # Ensure integer ticks for Number of Modules
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_or_show(plt.gcf(), make_plot_filename_from_labels('Number of Modules', 'Fitness', 'line'))

def tension_heatmap(x, y, xlabel, ylabel, bins=30, prefix=None):
    # Only valid solutions
    xvals = df_valid[x]
    yvals = df_valid[y]
    tension = df_valid['max_tension']
    stat, xedges, yedges, binnumber = binned_statistic_2d(
        xvals, yvals, tension, statistic='mean', bins=bins
    )
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    mesh = ax.pcolormesh(xedges, yedges, stat.T, cmap='viridis', shading='auto')
    # Overlay valid scatter points for context
    plt.scatter(xvals, yvals, c=valid_colour, edgecolors='none', s=20, alpha=0.5, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar(mesh)
    cbar.set_label('Maximum Tension (kN)')
    plt.grid(True)
    # Ensure integer ticks for Number of Modules
    if xlabel.lower().startswith('number of modules'):
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if ylabel.lower().startswith('number of modules'):
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_or_show(plt.gcf(), make_plot_filename_from_labels(xlabel, ylabel, 'tension', prefix=prefix))

def plot_totalNetBuoyancySol_vs_minimumSurfaceClearance_highlight():
    """Create a scatter plot with highlighted solutions for total net buoyancy vs surface clearance."""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # Plot invalid points first (background)
    valid = df['is_valid']
    plt.scatter(df.loc[~valid, 'total_buoyancy_SoL'], 
                df.loc[~valid, 'min_surface_clearance'], 
                c=invalid_colour, edgecolors='none', s=30, alpha=1.0, marker='o')
    
    # Plot valid points
    plt.scatter(df.loc[valid, 'total_buoyancy_SoL'], 
                df.loc[valid, 'min_surface_clearance'], 
                c=valid_colour, edgecolors='none', s=40, alpha=0.9, marker='o')
    
    # Identify three points: leftmost, rightmost, and lower-corner (at right edge, lowest buoyancy at that Y)
    leftmost = df_valid.loc[df_valid['total_buoyancy_SoL'].idxmin()]
    rightmost = df_valid.loc[df_valid['total_buoyancy_SoL'].idxmax()]
    # Lower-corner: min total net buoyancy among points with surface clearance close to rightmost's surface clearance
    delta = 1.0  # meters tolerance
    candidates = df_valid[np.abs(df_valid['min_surface_clearance'] - rightmost['min_surface_clearance']) < delta]
    if not candidates.empty:
        lower_corner = candidates.loc[candidates['total_buoyancy_SoL'].idxmin()]
    else:
        lower_corner = rightmost  # fallback

    # Print detailed information about each highlighted point
    print("\n=== Highlighted Edge Cases ===")
    print("\n1. Leftmost Point (Minimum Total Net Buoyancy):")
    print(f"Solution ID: {leftmost.name}")
    print(f"Ballast Mass: {leftmost['bm_mass']} kg")
    print(f"Number of Modules: {leftmost['num_modules']}")
    print(f"Start Arc Length: {leftmost['start_arc_length']} m")
    print(f"Total Buoyancy (SoL): {leftmost['total_buoyancy_SoL']} kg")
    print(f"Min Surface Clearance: {leftmost['min_surface_clearance']} m")
    print(f"Fitness: {leftmost['fitness']}")
    
    print("\n2. Rightmost Point (Maximum Total Net Buoyancy):")
    print(f"Solution ID: {rightmost.name}")
    print(f"Ballast Mass: {rightmost['bm_mass']} kg")
    print(f"Number of Modules: {rightmost['num_modules']}")
    print(f"Start Arc Length: {rightmost['start_arc_length']} m")
    print(f"Total Buoyancy (SoL): {rightmost['total_buoyancy_SoL']} kg")
    print(f"Min Surface Clearance: {rightmost['min_surface_clearance']} m")
    print(f"Fitness: {rightmost['fitness']}")
    
    print("\n3. Lower-Corner Point (Right Edge, Minimum Buoyancy at that Y):")
    print(f"Solution ID: {lower_corner.name}")
    print(f"Ballast Mass: {lower_corner['bm_mass']} kg")
    print(f"Number of Modules: {lower_corner['num_modules']}")
    print(f"Start Arc Length: {lower_corner['start_arc_length']} m")
    print(f"Total Buoyancy (SoL): {lower_corner['total_buoyancy_SoL']} kg")
    print(f"Min Surface Clearance: {lower_corner['min_surface_clearance']} m")
    print(f"Fitness: {lower_corner['fitness']}")
    print("\n===========================\n")

    # Plot only the three highlighted points (without annotation labels)
    for pt in [leftmost, rightmost, lower_corner]:
        plt.scatter(pt['total_buoyancy_SoL'], 
                    pt['min_surface_clearance'],
                    c=valid_colour,
                    edgecolors='black',
                    s=100,
                    linewidth=2,
                    marker='o')

    # Add labels and grid
    plt.xlabel('Total Net Buoyancy SoL (kg)')
    plt.ylabel('Minimum Surface Clearance (m)')
    plt.grid(True)
    
    # Save the plot
    save_or_show(plt.gcf(), 'totalNetBuoyancySol_vs_minimumSurfaceClearance_highlight.png')

# Define pairs for plotting (replace max_tension with num_modules or start_arc_length)
plot_pairs = [
    ('total_buoyancy_SoL', 'min_surface_clearance', 'Total Net Buoyancy SoL vs Surface Clearance',
     'Total Net Buoyancy SoL (kg)', 'Minimum Surface Clearance (m)', {}),

    ('total_buoyancy_SoL', 'min_seabed_clearance', 'Total Net Buoyancy SoL vs Seabed Clearance',
     'Total Net Buoyancy SoL (kg)', 'Minimum Seabed Clearance (m)', {}),

    ('total_buoyancy_SoL', 'min_bend_radius', 'Total Net Buoyancy SoL vs Minimum Bend Radius',
     'Total Net Buoyancy SoL (kg)', 'Minimum Bend Radius (m)', {}),

    ('min_bend_radius', 'num_modules', 'Minimum Bend Radius vs Number of Modules',
     'Minimum Bend Radius (m)', 'Number of Modules', {}),

    ('total_buoyancy_SoL', 'num_modules', 'Total Net Buoyancy SoL vs Number of Modules',
     'Total Net Buoyancy SoL (kg)', 'Number of Modules', {}),

    ('start_arc_length', 'bm_mass', 'Start Arc Length vs BM Mass',
     'Start Arc Length (m)', 'Buoyancy Module Mass SoL (kg)', {}),

    ('num_modules', 'fitness', 'Number of Modules vs Fitness',
     'Number of Modules', 'Fitness', {}),

    ('start_arc_length', 'fitness', 'Start Arc Length vs Fitness',
     'Start Arc Length (m)', 'Fitness', {}),

    ('min_seabed_clearance', 'min_bend_radius', 'Seabed Clearance vs Minimum Bend Radius',
     'Minimum Seabed Clearance (m)', 'Minimum Bend Radius (m)', {}),

    ('min_seabed_clearance', 'num_modules', 'Seabed Clearance vs Number of Modules',
     'Minimum Seabed Clearance (m)', 'Number of Modules', {})
]

# 1: Neutral scatter for MBR vs Number of Modules (undistinguished max values)
scatter_neutral('min_bend_radius', 'num_modules',
                'Minimum Bend Radius (m)', 'Number of Modules',
                make_plot_filename_from_labels('Minimum Bend Radius (m)', 'Number of Modules', 'neutral'))

# Loop to generate scatter and density plots for each defined pair
for idx, (x, y, title, xlabel, ylabel, extras) in enumerate(plot_pairs):
    prefix = chr(ord('A') + idx)
    if (x, y) == ('num_modules', 'fitness'):
        line_plot_num_modules_vs_fitness()
        continue
    xlim, ylim = scatter_plot(x, y, xlabel, ylabel, title, **extras, prefix=prefix)
    density_plot(x, y, xlabel, ylabel, title, xlim=xlim, ylim=ylim, prefix=prefix)
    tension_heatmap(x, y, xlabel, ylabel, prefix=prefix)

# Generate the highlighted solutions plot
plot_totalNetBuoyancySol_vs_minimumSurfaceClearance_highlight()
