import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV, using the second row as the header
csv_path = 'Edge-cases/edgeCases.csv'
df = pd.read_csv(csv_path, header=1)

# The columns are: Arc length (m), X, Z, X, Z, X, Z
# The first set after Arc length is Leftmost, then Rightmost, then Lower-Corner
arc = df['Arc length (m)'].values
x_left = df['X'].values
z_left = df['Z'].values
x_right = df['X.1'].values
z_right = df['Z.1'].values
x_corner = df['X.2'].values
z_corner = df['Z.2'].values

# BM parameters for each configuration
bm_params = [
    {'start_arc': 84, 'num_modules': 8, 'color': '#4FC3F7'},      # Leftmost
    {'start_arc': 127, 'num_modules': 15, 'color': '#FF8A65'},    # Rightmost
    {'start_arc': 46, 'num_modules': 14, 'color': '#81C784'},     # Lower-Corner
]
spacing = 5.0

fig, ax = plt.subplots(figsize=(12, 10))
colors = ['#4FC3F7', '#FF8A65', '#81C784']  # blue, red, green
ax.plot(x_left, z_left, '-', linewidth=2.5, color=colors[0])
ax.plot(x_right, z_right, '-', linewidth=2.5, color=colors[1])
ax.plot(x_corner, z_corner, '-', linewidth=2.5, color=colors[2])

def plot_bms_on_profile(ax, arc, x, z, start_arc, num_modules, spacing, color):
    bm_arcs = [start_arc + i * spacing for i in range(num_modules)]
    bm_x = np.interp(bm_arcs, arc, x)
    bm_z = np.interp(bm_arcs, arc, z)
    ax.scatter(bm_x, bm_z, s=50, marker='o', color=color, alpha=0.9, zorder=5, edgecolors='black', linewidth=0.5)

# Plot BM markers for each profile
plot_bms_on_profile(ax, arc, x_left, z_left, bm_params[0]['start_arc'], bm_params[0]['num_modules'], spacing, colors[0])
plot_bms_on_profile(ax, arc, x_right, z_right, bm_params[1]['start_arc'], bm_params[1]['num_modules'], spacing, colors[1])
plot_bms_on_profile(ax, arc, x_corner, z_corner, bm_params[2]['start_arc'], bm_params[2]['num_modules'], spacing, colors[2])

ax.set_aspect('equal')
ax.set_xlabel('Horizontal Position (m)')
ax.set_ylabel('Depth (m)')
ax.set_ylim(-100, 0)
ax.grid(True, alpha=0.3, linestyle='--')

x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
tick_step = 10
base_x_ticks = np.arange(np.floor(x_min/tick_step)*tick_step, np.ceil(x_max/tick_step)*tick_step+1, tick_step)
base_y_ticks = np.arange(np.floor(y_min/tick_step)*tick_step, np.ceil(y_max/tick_step)*tick_step+1, tick_step)
ax.set_xticks(base_x_ticks)
ax.set_xticklabels([f"{int(round(x))}" for x in base_x_ticks])
ax.set_yticks(base_y_ticks)
ax.set_yticklabels([f"{int(round(y))}" for y in base_y_ticks])

plt.tight_layout()
plt.savefig('Edge-cases/edge_cases_profiles.png', dpi=300, bbox_inches='tight')
plt.close(fig)
