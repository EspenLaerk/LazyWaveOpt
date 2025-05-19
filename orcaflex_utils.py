"""
OrcaFlex utilities for buoyancy module configuration
"""

import OrcFxAPI
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from config import (
    CABLE_NAMES, BM_NAMES, BM_EoL_MASS_FACTOR
)
import os
import sys
import math
from progress_tracker import optimization_state, OptimizationState
import config

# Create a logger for this module
logger = logging.getLogger(__name__)

# Use a fixed module volume for all calculations
FIXED_MODULE_VOLUME = 1.0  # m^3

def load_model(model_path):
    """
    Load an OrcaFlex model from a specified file path.
    
    Args:
        model_path (str): Path to the OrcaFlex model file
        
    Returns:
        OrcFxAPI.Model: The loaded OrcaFlex model
    """
    try:
        model = OrcFxAPI.Model(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def update_bm_mass(model, bm_name, mass):
    """
    Update the mass of a buoyancy module.
    
    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        bm_name (str): Name of the buoyancy module
        mass (float): New mass in kg
    """
    try:
        buoy = model[bm_name]
        current_mass = buoy.Mass
        
        if abs(current_mass - mass) > 0.01:
            buoy.Mass = mass
            logger.info(f"Updated {bm_name} mass to {mass} kg")
        else:
            logger.info(f"Mass of {bm_name} already set to {mass} kg")
    except Exception as e:
        logger.error(f"Failed to update {bm_name} mass: {e}")
        raise

def configure_buoyancy_modules(model, cable_name, bm_name, start_arc_length, num_modules, spacing=None):
    """
    Configure buoyancy modules on a cable.
    
    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        cable_name (str): Name of the cable
        bm_name (str): Name of the buoyancy module
        start_arc_length (float): Starting arc length for first module in meters
        num_modules (int): Number of modules to distribute
        spacing (float, optional): Spacing between modules in meters (default: config.BM_SPACING)
    """
    if spacing is None:
        spacing = getattr(config, 'BM_SPACING', 5.0)
    try:
        logger.info(f"Configuring {num_modules} modules on {cable_name} starting at {start_arc_length}m")
        
        # Get the cable object
        cable = model[cable_name]
        
        # Set the exact number of attachments we need (OrcaFlex uses zero-indexed attachments)
        cable.NumberOfAttachments = num_modules
        
        # Configure each module's position along the cable (zero-indexed)
        for i in range(num_modules):
            # Calculate position for this module
            position = start_arc_length + (i * spacing)
            
            # Ensure position is not negative to avoid OrcaFlex error
            if position < 0:
                logger.warning(f"Module position {position} on {cable_name} is negative. Setting to 0.01.")
                position = 0.01  # Set to a small positive value instead of 0 to avoid other potential issues
            
            # Set the arc length position (using zero-based index)
            cable.AttachmentZ[i] = position
            
        logger.info(f"Successfully configured {num_modules} modules on {cable_name}")
    except Exception as e:
        logger.error(f"Failed to configure modules on {cable_name}: {e}")
        raise

def run_static_analysis(model, timeout=None):
    """
    Run static analysis on the OrcaFlex model with optional timeout (in seconds).
    If timeout is exceeded, statics is cancelled using the progress handler.
    Returns True if statics succeeded, False if failed, or 'timeout' if cancelled by timeout.
    """
    start_time = time.time()
    cancelled = [False]
    def statics_progress_handler(model, progress):
        elapsed = time.time() - start_time
        if timeout is not None and elapsed > timeout:
            cancelled[0] = True
            logger.warning(f"Statics progress handler: cancelling statics after {elapsed:.2f}s (timeout={timeout}s)")
            return True  # Cancel statics
        return False
    model.staticsProgressHandler = statics_progress_handler
    try:
        logger.info(f"Calling model.CalculateStatics() with timeout={timeout}")
        model.CalculateStatics()
        if cancelled[0]:
            logger.warning("Statics was cancelled due to timeout.")
            return 'timeout'
        logger.info("Statics completed successfully.")
        return True
    except Exception as e:
        if cancelled[0]:
            logger.warning(f"Statics cancelled by progress handler: {e}")
            return 'timeout'
        logger.error(f"Static analysis failed: {e}")
        return False
    finally:
        if hasattr(model, 'staticsProgressHandler'):
            model.staticsProgressHandler = None

def save_model(model, file_path):
    """
    Save the OrcaFlex model to a file.
    
    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        file_path (str): Path where model should be saved
    """
    try:
        # Save the model
        model.SaveData(file_path)
        logger.info(f"Model saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

# ===== Plotting functions ===== #

def get_rangegraph_data(model, object_name, variable_name):
    """
    Retrieve range graph data for a specific variable from a cable.

    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        object_name (str): Name of the object (e.g., cable name)
        variable_name (str): Name of the variable to extract (e.g., "Z", "X", "bend Radius")

    Returns:
        dict: Dictionary with "X" (arc length) and "Mean" (variable values)
    """
    try:
        obj = model[object_name]
        range_graph = obj.RangeGraph(variable_name)

        # Convert to numpy arrays for easy processing
        x_values = np.array(range_graph.X)
        mean_values = np.array(range_graph.Mean)
        
        # Filter out invalid values (NaN, inf, or extremely large values)
        valid_indices = ~np.isnan(mean_values) & ~np.isinf(mean_values) & (mean_values < 1e6)

        # Apply the filter
        x_values = x_values[valid_indices]
        mean_values = mean_values[valid_indices]

        return {
            "X": x_values.tolist(),
            "Mean": mean_values.tolist()
        }
    except Exception as e:
        logger.error(f"Error getting range graph data for {object_name}.{variable_name}: {e}")
        return {"X": [], "Mean": []}

def get_bm_positions(arc_lengths, x_values, z_values, start_arc, num_modules, spacing):
    """
    Calculate the X and Z positions of buoyancy modules along a cable.

    Args:
        arc_lengths (list): Arc length values along the cable
        x_values (list): X position values along the cable
        z_values (list): Z position values along the cable
        start_arc (float): Starting arc length for the first module
        num_modules (int): Number of modules
        spacing (float): Spacing between modules

    Returns:
        tuple: (bm_x_positions, bm_z_positions) - lists of X and Z coordinates
    """
    bm_x_positions = []
    bm_z_positions = []
    
    # Generate arc lengths for each buoyancy module
    bm_arc_lengths = [start_arc + i * spacing for i in range(num_modules)]
    
    # Get the valid range of arc lengths
    min_arc = min(arc_lengths) if arc_lengths else 0
    max_arc = max(arc_lengths) if arc_lengths else 0
    
    # Find the position of each module through interpolation
    for bm_arc in bm_arc_lengths:
        # Skip if module is outside the cable's modeled range
        if bm_arc < min_arc or bm_arc > max_arc:
            continue
            
        # Find the segment where this module is located
        for i in range(len(arc_lengths) - 1):
            if arc_lengths[i] <= bm_arc <= arc_lengths[i+1]:
                # Linear interpolation
                segment_length = arc_lengths[i+1] - arc_lengths[i]
                if segment_length > 0:
                    t = (bm_arc - arc_lengths[i]) / segment_length
                    x = x_values[i] + t * (x_values[i+1] - x_values[i])
                    z = z_values[i] + t * (z_values[i+1] - z_values[i])
                    bm_x_positions.append(x)
                    bm_z_positions.append(z)
                break
    
    return bm_x_positions, bm_z_positions

def plot_cable_profiles(model, variable="Z", show_buoyancy_modules=True, save_path=None):
    """
    Plot cable profiles for all cables in the model.
    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        variable (str): Variable to plot (default is "Z" for depth)
        show_buoyancy_modules (bool): Whether to show buoyancy module positions
        save_path (str, optional): Path to save the plot image
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # High-contrast color pairs for each offset
    position_colors = {
        'near': '#4FC3F7',     # Light Blue (SoL)
        '': '#81C784',         # Light Green (SoL)
        'far': '#FF8A65'       # Light Red/Orange (SoL)
    }
    
    # EoL will use darker shades of the same colors
    eol_colors = {
        'near': '#0D47A1',     # Dark Blue (EoL)
        '': '#1B5E20',         # Dark Green (EoL)
        'far': '#B71C1C'       # Dark Red (EoL)
    }
    
    # Group cables by type
    sol_cables = [cable for cable in CABLE_NAMES if "SoL" in cable]
    eol_cables = [cable for cable in CABLE_NAMES if "EoL" in cable]
    
    # Process each cable
    all_cable_data = []
    
    # Process SoL cables
    for cable in sol_cables:
        # Determine position (near, default, or far)
        if "near" in cable.lower():
            position = "near"
        elif "far" in cable.lower():
            position = "far"
        else:
            position = ""
            
        # Get X and Z data
        z_data = get_rangegraph_data(model, cable, variable)
        x_data = get_rangegraph_data(model, cable, "X")
        
        # Extract values
        arc_lengths = z_data["X"]
        z_values = z_data["Mean"]
        x_values = x_data["Mean"]
        
        # Store data for later use
        all_cable_data.append({
            "cable": cable,
            "arc_lengths": arc_lengths,
            "x_values": x_values,
            "z_values": z_values,
            "type": "SoL",
            "position": position,
            "color": position_colors[position]
        })
    
    # Process EoL cables
    for cable in eol_cables:
        # Determine position (near, default, or far)
        if "near" in cable.lower():
            position = "near"
        elif "far" in cable.lower():
            position = "far"
        else:
            position = ""
            
        # Get X and Z data
        z_data = get_rangegraph_data(model, cable, variable)
        x_data = get_rangegraph_data(model, cable, "X")
        
        # Extract values
        arc_lengths = z_data["X"]
        z_values = z_data["Mean"]
        x_values = x_data["Mean"]
        
        # Store data for later use
        all_cable_data.append({
            "cable": cable,
            "arc_lengths": arc_lengths,
            "x_values": x_values,
            "z_values": z_values,
            "type": "EoL",
            "position": position,
            "color": eol_colors[position]
        })
    
    # Track most critical metrics across all cables
    critical_metrics = {
        "seabed": {"value": float('inf'), "arc": 0, "cable": None, "x": None, "z": None},
        "surface": {"value": float('inf'), "arc": 0, "cable": None, "x": None, "z": None},
        "tension": {"value": 0, "arc": 0, "cable": None, "x": None, "z": None, "loc": None},
        "radius": {"value": float('inf'), "arc": 0, "cable": None, "x": None, "z": None, "loc": None}
    }
    
    # Plot each cable and find its metrics
    for cable_data in all_cable_data:
        cable = cable_data["cable"]
        arc_lengths = cable_data["arc_lengths"]
        x_values = cable_data["x_values"]
        z_values = cable_data["z_values"]
        color = cable_data["color"]
        
        # --- TRUNCATE TO SUSPENDED LENGTH + 20m ---
        suspended_length = get_suspended_length(model, cable)
        max_plot_arc = suspended_length + 20.0
        # Find the last index to include
        plot_indices = [i for i, arc in enumerate(arc_lengths) if arc <= max_plot_arc]
        if plot_indices:
            last_idx = plot_indices[-1] + 1  # include up to and including this index
            arc_lengths_plot = arc_lengths[:last_idx]
            x_values_plot = x_values[:last_idx]
            z_values_plot = z_values[:last_idx]
        else:
            arc_lengths_plot = arc_lengths
            x_values_plot = x_values
            z_values_plot = z_values
        
        # Plot the cable profile
        if x_values_plot and z_values_plot:
            line, = ax.plot(x_values_plot, z_values_plot, '-', linewidth=2.5,
                          label=f"{cable}", color=color)
                          
            # Add buoyancy modules if requested
            if show_buoyancy_modules:
                # Get the actual cable object from the model
                cable_obj = model[cable]
                
                # Get the number of attachments (buoyancy modules)
                num_modules = cable_obj.NumberOfAttachments
                
                # Extract the BM positions directly from the model
                bm_arc_lengths = []
                for i in range(num_modules):
                    bm_arc_lengths.append(cable_obj.AttachmentZ[i])
                
                # Calculate BM positions using the actual configured arc lengths
                bm_x = []
                bm_z = []
                
                # Find the position of each module through interpolation
                for bm_arc in bm_arc_lengths:
                    # Skip if module is outside the cable's modeled range
                    if not arc_lengths_plot:
                        continue
                        
                    min_arc = min(arc_lengths_plot)
                    max_arc = max(arc_lengths_plot)
                    
                    if bm_arc < min_arc or bm_arc > max_arc:
                        continue
                        
                    # Find the segment where this module is located
                    for i in range(len(arc_lengths_plot) - 1):
                        if arc_lengths_plot[i] <= bm_arc <= arc_lengths_plot[i+1]:
                            # Linear interpolation
                            segment_length = arc_lengths_plot[i+1] - arc_lengths_plot[i]
                            if segment_length > 0:
                                t = (bm_arc - arc_lengths_plot[i]) / segment_length
                                x = x_values_plot[i] + t * (x_values_plot[i+1] - x_values_plot[i])
                                z = z_values_plot[i] + t * (z_values_plot[i+1] - z_values_plot[i])
                                bm_x.append(x)
                                bm_z.append(z)
                            break
                
                if bm_x and bm_z:
                    ax.scatter(bm_x, bm_z, s=50, marker='o', color=color, 
                             alpha=0.9, zorder=5, edgecolors='black', linewidth=0.5)
            
            # Function to find x,z coordinates at a specific arc length
            def find_point_at_arc(arc_length):
                if arc_length == 0.0:  # Handle special case for the hang-off point
                    if len(arc_lengths_plot) > 0 and len(x_values_plot) > 0 and len(z_values_plot) > 0:
                        # Return the values at the first point
                        return x_values_plot[0], z_values_plot[0]
                    return None, None
                
                if arc_length < 0.5 and len(arc_lengths_plot) > 0:  # Very close to hang-off
                    # Return the first point or interpolate between first and second
                    if len(arc_lengths_plot) > 1:
                        # Linear interpolation between first two points
                        t = arc_length / arc_lengths_plot[1] if arc_lengths_plot[1] > 0 else 0
                        x = x_values_plot[0] + t * (x_values_plot[1] - x_values_plot[0])
                        z = z_values_plot[0] + t * (z_values_plot[1] - z_values_plot[0])
                        return x, z
                    return x_values_plot[0], z_values_plot[0]
                
                # Find the closest arc length in our data
                closest_idx = min(range(len(arc_lengths_plot)), 
                                 key=lambda i: abs(arc_lengths_plot[i] - arc_length))
                
                # If too far from requested arc length, interpolate
                if abs(arc_lengths_plot[closest_idx] - arc_length) > 1.0 and closest_idx > 0:
                    # Find surrounding points
                    if arc_lengths_plot[closest_idx] > arc_length and closest_idx > 0:
                        idx1, idx2 = closest_idx - 1, closest_idx
                    else:
                        idx1, idx2 = closest_idx, closest_idx + 1
                        
                    if idx2 < len(arc_lengths_plot):
                        # Linear interpolation
                        arc1, arc2 = arc_lengths_plot[idx1], arc_lengths_plot[idx2]
                        x1, x2 = x_values_plot[idx1], x_values_plot[idx2]
                        z1, z2 = z_values_plot[idx1], z_values_plot[idx2]
                        
                        t = (arc_length - arc1) / (arc2 - arc1) if arc2 != arc1 else 0
                        x = x1 + t * (x2 - x1)
                        z = z1 + t * (z2 - z1)
                        return x, z
                
                # Return the coordinates at the closest index
                return x_values_plot[closest_idx], z_values_plot[closest_idx]
            
            # Get critical measurements for this cable
            seabed_clearance, seabed_arc = get_seabed_clearance(model, cable)
            surface_clearance, surface_arc = get_surface_clearance(model, cable)
            max_tension, tension_arc, tension_loc = get_max_tension(model, cable)
            min_radius, radius_arc, radius_loc = get_min_bend_radius(model, cable)
            
            # Update critical metrics if this cable has a more critical value
            # For seabed and surface clearance, smaller is more critical
            # For tension, larger is more critical
            # For bend radius, smaller is more critical
            
            if 0 < seabed_clearance < critical_metrics["seabed"]["value"]:
                x, z = find_point_at_arc(seabed_arc)
                critical_metrics["seabed"] = {
                    "value": seabed_clearance,
                    "arc": seabed_arc,
                    "cable": cable,
                    "x": x,
                    "z": z,
                    "color": color
                }
            
            if 0 < surface_clearance < critical_metrics["surface"]["value"]:
                x, z = find_point_at_arc(surface_arc)
                critical_metrics["surface"] = {
                    "value": surface_clearance,
                    "arc": surface_arc,
                    "cable": cable,
                    "x": x,
                    "z": z,
                    "color": color
                }
            
            if max_tension > critical_metrics["tension"]["value"]:
                x, z = find_point_at_arc(tension_arc)
                critical_metrics["tension"] = {
                    "value": max_tension,
                    "arc": tension_arc,
                    "cable": cable,
                    "x": x,
                    "z": z,
                    "loc": tension_loc,
                    "color": color
                }
            
            if 0 < min_radius < critical_metrics["radius"]["value"]:
                x, z = find_point_at_arc(radius_arc)
                critical_metrics["radius"] = {
                    "value": min_radius,
                    "arc": radius_arc,
                    "cable": cable,
                    "x": x,
                    "z": z,
                    "loc": radius_loc,
                    "color": color
                }
    
    # Set equal aspect ratio to ensure 1:1 scale (distances are visually accurate)
    ax.set_aspect('equal')
    
    # Add legend for cables only - moved to bottom left
    cable_legend = None
    if getattr(config, 'PLOT_SHOW_CABLE_LEGEND', True):
        cable_handles, cable_labels = ax.get_legend_handles_labels()
        cable_legend = ax.legend(handles=cable_handles, labels=cable_labels, 
                               loc='lower left', framealpha=0.9, title="Cables", 
                               fontsize=8, title_fontsize=9, frameon=True, 
                               edgecolor='gray', borderaxespad=0.5)
        ax.add_artist(cable_legend)  # Ensure cable legend is not overwritten

    # Show critical point markers on the plot
    critical_marker_styles = {
        'seabed': dict(marker='D', s=90, c=None, edgecolors='black', linewidths=1, zorder=10),
        'surface': dict(marker='s', s=90, c=None, edgecolors='black', linewidths=1, zorder=10),
        'tension': dict(marker='*', s=140, c=None, edgecolors='black', linewidths=1, zorder=11),
        'radius': dict(marker='p', s=110, c=None, edgecolors='black', linewidths=1, zorder=10),
    }
    for key, metric in critical_metrics.items():
        if metric['x'] is not None and metric['z'] is not None and metric['cable'] is not None:
            style = critical_marker_styles[key].copy()
            style['c'] = metric.get('color', 'gray')
            ax.scatter([metric['x']], [metric['z']], **style)

    # Add critical points legend (single-line box in upper left)
    from matplotlib.lines import Line2D
    critical_legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', 
               markersize=8, markeredgecolor='black', markeredgewidth=1, label='Min Seabed Clearance'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=8, markeredgecolor='black', markeredgewidth=1, label='Min Surface Clearance'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
               markersize=10, markeredgecolor='black', markeredgewidth=1, label='Max Tension'),
        Line2D([0], [0], marker='p', color='w', markerfacecolor='gray', 
               markersize=9, markeredgecolor='black', markeredgewidth=1, label='Min Bend Radius'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=6, markeredgecolor='black', markeredgewidth=0.5, label='Buoyancy Module')
    ]
    critical_legend = ax.legend(handles=critical_legend_elements, loc='upper left', framealpha=0.95, fontsize=9, ncol=len(critical_legend_elements), borderaxespad=0.5, handletextpad=1.2, title="Critical Points", title_fontsize=9)
    # Add after cable legend so both are visible
    
    # Remove the current configuration and metrics boxes, and replace with a single summary box below the surface clearance zone
    if getattr(config, 'PLOT_SHOW_METRICS_BOX', True):
        # Gather configuration parameters
        sol_cable = next((c for c in CABLE_NAMES if c == "SoL"), None)
        eol_cable = next((c for c in CABLE_NAMES if c == "EoL"), None)
        sol_bm_mass = model[BM_NAMES["SoL"]].Mass if sol_cable else None
        eol_bm_mass = model[BM_NAMES["EoL"]].Mass if eol_cable else None
        sol_num_modules = model[sol_cable].NumberOfAttachments if sol_cable else None
        eol_num_modules = model[eol_cable].NumberOfAttachments if eol_cable else None
        sol_start_arc = model[sol_cable].AttachmentZ[0] if (sol_cable and model[sol_cable].NumberOfAttachments > 0) else None
        eol_start_arc = model[eol_cable].AttachmentZ[0] if (eol_cable and model[eol_cable].NumberOfAttachments > 0) else None
        module_spacing = getattr(config, 'BM_SPACING', 5.0)
        mass_factor = BM_EoL_MASS_FACTOR
        # Compose summary box text
        summary_text = "Configuration\n"
        summary_text += "═══════════════\n"
        if sol_bm_mass is not None:
            summary_text += f"SoL BM mass: {sol_bm_mass:.2f} kg\n"
        if eol_bm_mass is not None:
            summary_text += f"EoL BM mass: {eol_bm_mass:.2f} kg\n"
        summary_text += f"EoL mass factor: {mass_factor}\n"
        if sol_num_modules is not None:
            summary_text += f"Number of modules: {sol_num_modules}\n"
        if module_spacing is not None:
            summary_text += f"Module spacing: {module_spacing:.2f} m\n"
        if sol_start_arc is not None:
            summary_text += f"Start arc length: {sol_start_arc:.2f} m\n"
        # Add total net buoyancy for SoL and EoL
        from orcaflex_utils import get_total_net_buoyancy
        from config import SEAWATER_DENSITY
        if sol_bm_mass is not None and sol_num_modules is not None:
            sol_net_buoy = get_total_net_buoyancy(sol_bm_mass, sol_num_modules, SEAWATER_DENSITY)
            summary_text += f"Total net buoyancy (SoL): {sol_net_buoy:.2f} kg\n"
        if eol_bm_mass is not None and eol_num_modules is not None:
            eol_net_buoy = get_total_net_buoyancy(eol_bm_mass, eol_num_modules, SEAWATER_DENSITY)
            summary_text += f"Total net buoyancy (EoL): {eol_net_buoy:.2f} kg\n"
        # Add a separator
        summary_text += "\nCritical Values\n"
        summary_text += "═══════════════\n"
        # Min seabed clearance
        if critical_metrics["seabed"]["value"] != float('inf'):
            m = critical_metrics["seabed"]
            summary_text += f"Min Seabed Clearance: {m['value']:.2f} m\n"
        # Min surface clearance
        if critical_metrics["surface"]["value"] != float('inf'):
            m = critical_metrics["surface"]
            summary_text += f"Min Surface Clearance: {m['value']:.2f} m\n"
        # Max tension
        if critical_metrics["tension"]["value"] > 0:
            m = critical_metrics["tension"]
            summary_text += f"Max Tension: {m['value']:.2f} kN\n"
        # Min bend radius
        if critical_metrics["radius"]["value"] != float('inf'):
            m = critical_metrics["radius"]
            summary_text += f"Min Bend Radius: {m['value']:.2f} m"
        # Place the box just below the surface clearance zone
        min_surface = getattr(config, 'MIN_SURFACE_CLEARANCE', 15.0)
        y_box = -min_surface - 2  # 2m margin below the zone
        summary_props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, edgecolor='gray')
        ax.text(0.98, y_box, summary_text, transform=ax.get_yaxis_transform(),
                fontsize=9, va='top', ha='right', 
                bbox=summary_props, family='monospace')

    # Add title and labels
    if getattr(config, 'PLOT_SHOW_TITLE', True):
        ax.set_title("Cable Profiles with Buoyancy Modules and Critical Points")
    ax.set_xlabel("Horizontal Position (m)")
    
    if variable == "Z":
        ax.set_ylabel("Depth (m)")
    else:
        ax.set_ylabel(variable)
    
    # Set Y-axis range from -100 to 0
    ax.set_ylim(-100, 0)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Use regular x and y ticks instead of adding the critical point ticks
    x_min, x_max = ax.get_xlim()
    base_x_ticks = np.arange(np.floor(x_min/10)*10, np.ceil(x_max/10)*10+1, 10)
    ax.set_xticks(base_x_ticks)
    ax.set_xticklabels([f"{int(round(x))}" for x in base_x_ticks])
    
    y_min, y_max = ax.get_ylim()
    base_y_ticks = np.arange(np.floor(y_min/10)*10, np.ceil(y_max/10)*10+1, 10)
    ax.set_yticks(base_y_ticks)
    ax.set_yticklabels([f"{int(round(y))}" for y in base_y_ticks])
    
    # Add shaded zones for min surface and seabed clearance
    min_surface = getattr(config, 'MIN_SURFACE_CLEARANCE', 15.0)
    min_seabed = getattr(config, 'MIN_SEABED_CLEARANCE', 15.0)
    ax.axhspan(0, -min_surface, facecolor='gray', alpha=0.18, zorder=0)
    # Seabed clearance zone: from (y_min) up to -(model.environment.WaterDepth - min_seabed) if possible, else -100+min_seabed
    # Assume water depth is 100 if not available
    try:
        water_depth = model.environment.WaterDepth
    except Exception:
        water_depth = 100
    seabed_zone_top = -water_depth + min_seabed
    ax.axhspan(seabed_zone_top, y_min, facecolor='gray', alpha=0.18, zorder=0)
    # Add labels to the right of each zone, vertically aligned with the limits and with a larger margin
    xlim = ax.get_xlim()
    x_text = xlim[1] - 0.01 * (xlim[1] - xlim[0])
    margin = 1.0  # meters
    # Surface clearance label (move just above the lower limit, inside the zone)
    ax.text(x_text, -min_surface + margin, f"Minimum Surface Clearance Zone ({min_surface:.1f} m)",
            va='bottom', ha='right', fontsize=9, color='dimgray', rotation=0)
    # Seabed clearance label (move just below the upper limit, inside the zone)
    ax.text(x_text, seabed_zone_top - margin, f"Minimum Seabed Clearance Zone ({min_seabed:.1f} m)",
            va='top', ha='right', fontsize=9, color='dimgray', rotation=0)
    
    # Finalize plot with tight layout but ensure the aspect ratio is maintained
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig

def plot_bend_radius(model, save_path=None):
    """
    Plot bend radius profiles for all cables.
    
    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        save_path (str, optional): Path to save the plot image
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for SoL and EoL cables
    sol_colors = ['blue', 'royalblue', 'cornflowerblue']
    eol_colors = ['red', 'indianred', 'lightcoral']
    
    # Group cables by type
    sol_cables = [cable for cable in CABLE_NAMES if "SoL" in cable]
    eol_cables = [cable for cable in CABLE_NAMES if "EoL" in cable]
    
    all_min_radius = []
    has_plot = False
    
    # Plot SoL cables
    for i, cable in enumerate(sol_cables):
        # Get bend radius data
        radius_data = get_rangegraph_data(model, cable, "bend Radius")
        
        # Extract values
        arc_lengths = radius_data["X"]
        radius_values = radius_data["Mean"]
        
        if radius_values:
            # Plot the bend radius profile
            line, = ax.plot(arc_lengths, radius_values, '-', linewidth=2,
                         label=f"{cable}", color=sol_colors[i % len(sol_colors)])
            has_plot = True
            
            # Record minimum value
            if radius_values:
                min_radius = min(radius_values)
                min_arc = arc_lengths[radius_values.index(min_radius)]
                all_min_radius.append((cable, min_radius, min_arc))
    
    # Plot EoL cables
    for i, cable in enumerate(eol_cables):
        # Get bend radius data
        radius_data = get_rangegraph_data(model, cable, "bend Radius")
        
        # Extract values
        arc_lengths = radius_data["X"]
        radius_values = radius_data["Mean"]
        
        if radius_values:
            # Plot the bend radius profile
            line, = ax.plot(arc_lengths, radius_values, '-', linewidth=2,
                         label=f"{cable}", color=eol_colors[i % len(eol_colors)])
            has_plot = True
            
            # Record minimum value
            if radius_values:
                min_radius = min(radius_values)
                min_arc = arc_lengths[radius_values.index(min_radius)]
                all_min_radius.append((cable, min_radius, min_arc))
    
    # Add information box with minimum bend radii
    if all_min_radius:
        info_text = f"Minimum Bend Radii\n"
        info_text += f"-----------------\n"
        
        # Sort by cable name
        all_min_radius.sort(key=lambda x: x[0])
        
        for cable, min_radius, min_arc in all_min_radius:
            info_text += f"{cable}: {min_radius:.2f} m at arc {min_arc:.1f} m\n"
    
        # Add information box to plot
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray')
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=9, va='bottom', ha='left', 
               bbox=props, family='monospace')
    
    # Add legend only if there are plots
    if has_plot:
        ax.legend(loc='upper right', framealpha=0.9)
    
    # Add title and labels
    ax.set_title("Cable Bend Radius Profiles")
    ax.set_xlabel("Arc Length (m)")
    ax.set_ylabel("Bend Radius (m)")
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Finalize plot
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Bend radius plot saved to {save_path}")
    
    return fig

# ===== Metric Extraction Functions ===== #

def get_seabed_clearance(model, cable_name):
    """
    Get the minimum clearance between the cable and the seabed, 
    focusing on sag sections rather than the touchdown point.

    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        cable_name (str): Name of the cable

    Returns:
        tuple: (min_clearance, arc_length) - Minimum seabed clearance in meters and its arc length
    """
    try:
        # Get the cable object
        cable = model[cable_name]
        
        # Get the Z positions and arc lengths of the cable
        z_data = get_rangegraph_data(model, cable_name, "Z")
        z_values = z_data["Mean"]
        arc_lengths = z_data["X"]
        
        if not z_values or len(z_values) < 3:
            logger.warning(f"Insufficient Z data found for {cable_name}")
            return (0.0, 0.0)
        
        # Get water depth
        water_depth = model.environment.WaterDepth
        
        # Identify the resting section at the end of the cable if any
        # (points very close to the seabed, within 0.5m)
        seabed_threshold = 0.5
        resting_indices = [
            i for i, z in enumerate(z_values) 
            if abs(z + water_depth) <= seabed_threshold
        ]
        
        # Exclude the resting section for local minima analysis
        if resting_indices:
            first_resting_index = resting_indices[0]
            filtered_indices = range(first_resting_index)
        else:
            filtered_indices = range(len(z_values))
        
        # Find local minima in the suspended section (these are the sag points)
        local_minima_indices = []
        for i in filtered_indices:
            if i > 0 and i < len(z_values) - 1:
                if z_values[i] < z_values[i-1] and z_values[i] < z_values[i+1]:
                    local_minima_indices.append(i)
        
        # If no local minima found, fall back to the absolute minimum
        if not local_minima_indices:
            if filtered_indices:
                min_index = min(filtered_indices, key=lambda i: z_values[i])
                local_minima_indices = [min_index]
            else:
                logger.warning(f"No valid points found for seabed clearance in {cable_name}")
                return (0.0, 0.0)
        
        # Calculate clearances at each local minimum
        clearances = []
        for index in local_minima_indices:
            # Seabed is at -water_depth, more negative Z values are deeper
            clearance = z_values[index] + water_depth
            clearances.append((clearance, arc_lengths[index], index))
        
        # Find the minimum clearance and its location
        min_clearance, min_arc_length, min_index = min(clearances, key=lambda x: x[0])
        
        logger.info(f"Minimum seabed clearance for {cable_name}: {min_clearance:.2f} m at arc length {min_arc_length:.2f} m")
        return (min_clearance, min_arc_length)

    except Exception as e:
        logger.error(f"Error getting seabed clearance for {cable_name}: {e}")
        return (0.0, 0.0)

def get_surface_clearance(model, cable_name):
    """
    Calculate the minimum clearance between the cable and the water surface.
    Finds local maxima of Z (closest to surface) and excludes hang-off points.

    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        cable_name (str): Name of the cable

    Returns:
        tuple: (min_clearance, arc_length) - Minimum surface clearance in meters and its arc length
    """
    try:
        # Get the cable object
        cable = model[cable_name]
        
        # Get Z coordinates along the cable
        data = get_rangegraph_data(model, cable_name, "Z")
        
        if not data["X"] or not data["Mean"]:
            logger.warning(f"No valid data points found for {cable_name}")
            return (0.0, 0.0)
        
        # Find the minimum clearance (maximum Z value, since Z is negative)
        z_values = data["Mean"]
        arc_lengths = data["X"]
        
        # Skip the first few points (hang-off/fixed points) - typically first 15m
        hang_off_threshold = 15.0  # Skip points within 15m of the beginning
        skip_indices = [i for i, arc in enumerate(arc_lengths) if arc < hang_off_threshold]
        
        # If all points are within the threshold, use all points (fallback)
        if len(skip_indices) == len(arc_lengths):
            skip_indices = []
        
        # Find local maxima (points closer to the surface than their neighbors)
        local_maxima_indices = []
        for i in range(1, len(z_values) - 1):
            # Skip hang-off points
            if i in skip_indices:
                continue
                
            # Check if this is a local maximum (Z values are negative, so we check for greater values)
            if z_values[i] > z_values[i-1] and z_values[i] > z_values[i+1]:
                local_maxima_indices.append(i)
        
        # If no local maxima found, fall back to the absolute maximum excluding hang-off points
        if not local_maxima_indices:
            # Get valid indices (excluding hang-off points)
            valid_indices = [i for i in range(len(z_values)) if i not in skip_indices]
            
            if valid_indices:
                # Find the point closest to surface (maximum Z)
                max_idx = max(valid_indices, key=lambda i: z_values[i])
                local_maxima_indices = [max_idx]
            else:
                logger.warning(f"No valid points found for surface clearance in {cable_name}")
                return (0.0, 0.0)
        
        # Find the smallest clearance among the local maxima
        clearances = []
        for idx in local_maxima_indices:
            # Surface clearance is the absolute value of Z
            clearance = abs(z_values[idx])
            clearances.append((clearance, arc_lengths[idx], idx))
        
        # Find the minimum surface clearance from the local maxima
        min_clearance, arc_length, _ = min(clearances, key=lambda x: x[0])
        
        logger.info(f"Minimum surface clearance for {cable_name}: {min_clearance:.2f} m at arc length {arc_length:.2f} m")
        return (min_clearance, arc_length)
    except Exception as e:
        logger.error(f"Error getting surface clearance for {cable_name}: {e}")
        return (0.0, 0.0)

def get_max_tension(model, cable_name):
    """
    Get the maximum tension in the cable and its location.

    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        cable_name (str): Name of the cable

    Returns:
        tuple: (max_tension, arc_length, location_desc) - Maximum tension in kN, its arc length, and location description
    """
    try:
        # Get tension data
        tension_data = get_rangegraph_data(model, cable_name, "Effective Tension")
        tension_values = tension_data["Mean"]
        arc_lengths = tension_data["X"]
        
        if not tension_values:
            logger.warning(f"No tension data found for {cable_name}")
            return (float('inf'), 0.0, "unknown")
        
        # Filter out invalid values
        valid_tensions = [(t, a) for t, a in zip(tension_values, arc_lengths) 
                           if not np.isnan(t) and not np.isinf(t)]
        
        if not valid_tensions:
            logger.warning(f"No valid tension values for {cable_name}")
            return (float('inf'), 0.0, "unknown")
        
        # Find the maximum tension and its location
        max_tension, max_arc_length = max(valid_tensions, key=lambda x: x[0])
        
        # Convert to kN if in N
        if max_tension > 1000:  # Assume it's in N if very large
            max_tension /= 1000
        
        # Add location description based on arc length (consistent with V1)
        if max_arc_length < 50:
            location = "near top connection"
        elif max_arc_length > 200:
            location = "near seabed"
        else:
            location = "in suspended section"
        
        # If max tension is at arc length 0, we need to handle it specially for plotting
        # Get the coordinates for the hang-off point (first point)
        if max_arc_length == 0.0 or max_arc_length < 1.0:
            # Get the x and z data for the first few points
            x_data = get_rangegraph_data(model, cable_name, "X")
            z_data = get_rangegraph_data(model, cable_name, "Z")
            
            # Use the first point if available
            if x_data["Mean"] and z_data["Mean"]:
                # Adjust the arc length slightly to ensure it's plotted
                max_arc_length = 0.1 if max_arc_length == 0.0 else max_arc_length
        
        logger.info(f"Maximum tension for {cable_name}: {max_tension:.2f} kN at {max_arc_length:.2f} m ({location})")
        return (max_tension, max_arc_length, location)
        
    except Exception as e:
        logger.error(f"Error getting max tension for {cable_name}: {e}")
        return (float('inf'), 0.0, "unknown")

def get_min_bend_radius(model, cable_name):
    """
    Get the minimum bend radius in the cable and its location.
    
    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        cable_name (str): Name of the cable
        
    Returns:
        tuple: (min_radius, arc_length, location_desc) - Minimum bend radius in meters, its arc length, and location description
    """
    try:
        # Get bend radius data
        radius_data = get_rangegraph_data(model, cable_name, "bend Radius")
        radius_values = radius_data["Mean"]
        arc_lengths = radius_data["X"]
        
        if not radius_values:
            logger.warning(f"No bend radius data found for {cable_name}")
            return (0.0, 0.0, "unknown")
        
        # Filter out invalid and zero/negative values
        valid_radii = [(r, a) for r, a in zip(radius_values, arc_lengths) 
                       if r > 0 and not np.isnan(r) and not np.isinf(r)]
        
        if not valid_radii:
            logger.warning(f"No valid bend radius values for {cable_name}")
            return (0.0, 0.0, "unknown")
        
        # Find the minimum radius and its location
        min_radius, min_arc_length = min(valid_radii, key=lambda x: x[0])
        
        # Add location description based on arc length (consistent with V1)
        if min_arc_length < 50:
            location = "near top connection"
        elif min_arc_length > 200:
            location = "near seabed"
        else:
            location = "in suspended section"
        
        logger.info(f"Minimum bend radius for {cable_name}: {min_radius:.2f} m at {min_arc_length:.2f} m ({location})")
        return (min_radius, min_arc_length, location)
        
    except Exception as e:
        logger.error(f"Error getting min bend radius for {cable_name}: {e}")
        return (0.0, 0.0, "unknown")

def get_suspended_length(model, cable_name, seabed_threshold=0.5):
    """
    Calculate the length of cable that is suspended (not resting on the seabed).
    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        cable_name (str): Name of the cable
        seabed_threshold (float): Z-difference to consider as 'on seabed'
    Returns:
        float: Suspended cable length in meters
    """
    cable = model[cable_name]
    z_data = get_rangegraph_data(model, cable_name, "Z")
    arc_lengths = z_data["X"]
    z_values = z_data["Mean"]
    if not arc_lengths or not z_values:
        return 0.0
    water_depth = model.environment.WaterDepth
    # Find the first index where the cable is considered on the seabed
    for i, z in enumerate(z_values):
        if abs(z + water_depth) <= seabed_threshold:
            return arc_lengths[i]
    # If never touches seabed, the whole cable is suspended
    return arc_lengths[-1] if arc_lengths else 0.0

def plot_cable_profiles_by_offset(model, output_dir, variable="Z", show_buoyancy_modules=True):
    """
    Generate and save individual cable profile plots for each offset (near, far, default),
    each showing only the SoL and EoL cable for that offset.
    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        output_dir (str): Directory to save the plots
        variable (str): Variable to plot (default is "Z" for depth)
        show_buoyancy_modules (bool): Whether to show buoyancy module positions
    """
    position_colors = {
        'near': '#4FC3F7',     # Light Blue (SoL)
        '': '#81C784',         # Light Green (SoL)
        'far': '#FF8A65'       # Light Red/Orange (SoL)
    }
    eol_colors = {
        'near': '#0D47A1',     # Dark Blue (EoL)
        '': '#1B5E20',         # Dark Green (EoL)
        'far': '#B71C1C'       # Dark Red (EoL)
    }
    offsets = ["near", "far", ""]
    offset_names = {"near": "near", "far": "far", "": "default"}
    for offset in offsets:
        fig, ax = plt.subplots(figsize=(12, 10))
        # Find SoL and EoL cable for this offset
        sol_cable = next((c for c in CABLE_NAMES if "SoL" in c and (offset in c.lower() if offset else ("near" not in c.lower() and "far" not in c.lower()))), None)
        eol_cable = next((c for c in CABLE_NAMES if "EoL" in c and (offset in c.lower() if offset else ("near" not in c.lower() and "far" not in c.lower()))), None)
        plotted = []
        for cable, color, label in [
            (sol_cable, position_colors[offset], "SoL"),
            (eol_cable, eol_colors[offset], "EoL")
        ]:
            if cable is None:
                continue
            z_data = get_rangegraph_data(model, cable, variable)
            x_data = get_rangegraph_data(model, cable, "X")
            arc_lengths = z_data["X"]
            z_values = z_data["Mean"]
            x_values = x_data["Mean"]
            if x_values and z_values:
                # --- TRUNCATE TO SUSPENDED LENGTH + 20m ---
                suspended_length = get_suspended_length(model, cable)
                max_plot_arc = suspended_length + 20.0
                plot_indices = [i for i, arc in enumerate(arc_lengths) if arc <= max_plot_arc]
                if plot_indices:
                    last_idx = plot_indices[-1] + 1
                    arc_lengths_plot = arc_lengths[:last_idx]
                    x_values_plot = x_values[:last_idx]
                    z_values_plot = z_values[:last_idx]
                else:
                    arc_lengths_plot = arc_lengths
                    x_values_plot = x_values
                    z_values_plot = z_values
                ax.plot(x_values_plot, z_values_plot, '-', linewidth=2.5, label=label, color=color)
                plotted.append(label)
                if show_buoyancy_modules:
                    cable_obj = model[cable]
                    num_modules = cable_obj.NumberOfAttachments
                    bm_arc_lengths = [cable_obj.AttachmentZ[i] for i in range(num_modules)]
                    bm_x = []
                    bm_z = []
                    for bm_arc in bm_arc_lengths:
                        if not arc_lengths_plot:
                            continue
                        min_arc = min(arc_lengths_plot)
                        max_arc = max(arc_lengths_plot)
                        if bm_arc < min_arc or bm_arc > max_arc:
                            continue
                        for i in range(len(arc_lengths_plot) - 1):
                            if arc_lengths_plot[i] <= bm_arc <= arc_lengths_plot[i+1]:
                                segment_length = arc_lengths_plot[i+1] - arc_lengths_plot[i]
                                if segment_length > 0:
                                    t = (bm_arc - arc_lengths_plot[i]) / segment_length
                                    x = x_values_plot[i] + t * (x_values_plot[i+1] - x_values_plot[i])
                                    z = z_values_plot[i] + t * (z_values_plot[i+1] - z_values_plot[i])
                                    bm_x.append(x)
                                    bm_z.append(z)
                                break
                    if bm_x and bm_z:
                        ax.scatter(bm_x, bm_z, s=50, marker='o', color=color, alpha=0.9, zorder=5, edgecolors='black', linewidth=0.5)
        ax.set_aspect('equal')
        if getattr(config, 'PLOT_SHOW_TITLE', True):
            ax.set_title(f"Cable Profile: {offset_names[offset].capitalize()} Offset")
        ax.set_xlabel("Horizontal Position (m)")
        if variable == "Z":
            ax.set_ylabel("Depth (m)")
        else:
            ax.set_ylabel(variable)
        ax.set_ylim(-100, 0)
        ax.grid(True, alpha=0.3, linestyle='--')
        # Set ticks to match main plot and ensure both axes have matching intervals
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        # Use the same step for both axes (10m)
        tick_step = 10
        base_x_ticks = np.arange(np.floor(x_min/tick_step)*tick_step, np.ceil(x_max/tick_step)*tick_step+1, tick_step)
        base_y_ticks = np.arange(np.floor(y_min/tick_step)*tick_step, np.ceil(y_max/tick_step)*tick_step+1, tick_step)
        ax.set_xticks(base_x_ticks)
        ax.set_xticklabels([f"{int(round(x))}" for x in base_x_ticks])
        ax.set_yticks(base_y_ticks)
        ax.set_yticklabels([f"{int(round(y))}" for y in base_y_ticks])
        min_surface = getattr(config, 'MIN_SURFACE_CLEARANCE', 15.0)
        min_seabed = getattr(config, 'MIN_SEABED_CLEARANCE', 15.0)
        ax.axhspan(0, -min_surface, facecolor='gray', alpha=0.18, zorder=0)
        try:
            water_depth = model.environment.WaterDepth
        except Exception:
            water_depth = 100
        seabed_zone_top = -water_depth + min_seabed
        ax.axhspan(seabed_zone_top, y_min, facecolor='gray', alpha=0.18, zorder=0)
        xlim = ax.get_xlim()
        x_text = xlim[1] - 0.01 * (xlim[1] - xlim[0])
        margin = 1.0  # meters
        ax.text(x_text, -min_surface + margin, f"Minimum Surface Clearance Zone ({min_surface:.1f} m)",
                va='bottom', ha='right', fontsize=9, color='dimgray', rotation=0)
        ax.text(x_text, seabed_zone_top - margin, f"Minimum Seabed Clearance Zone ({min_seabed:.1f} m)",
                va='top', ha='right', fontsize=9, color='dimgray', rotation=0)
        if plotted and getattr(config, 'PLOT_SHOW_CABLE_LEGEND', True):
            ax.legend(loc='lower left', framealpha=0.9, fontsize=10, frameon=True, edgecolor='gray', borderaxespad=0.5)
        ax.set_aspect('equal')
        plt.tight_layout()
        filename = f"cable_profiles_{offset_names[offset]}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Offset plot saved to {save_path}") 


def get_module_net_buoyancy_fixed_volume(mass, seawater_density=None, volume=FIXED_MODULE_VOLUME):
    """
    Calculate the net buoyancy of a single module with a fixed volume.
    Args:
        mass (float): Mass of the module in kg
        seawater_density (float, optional): Density of seawater in kg/m^3
        volume (float): Fixed volume in m^3
    Returns:
        float: Net buoyancy in kg (positive = upward force)
    """
    if seawater_density is None:
        seawater_density = config.SEAWATER_DENSITY
    return seawater_density * volume - mass

def get_total_net_buoyancy(bm_mass, num_modules, seawater_density=None):
    """
    Calculate the total net buoyancy for a solution (SoL by default) using fixed module volume.
    Args:
        bm_mass (float): Mass of a single module in kg
        num_modules (int): Number of modules
        seawater_density (float, optional): Density of seawater in kg/m^3
    Returns:
        float: Total net buoyancy in kg
    """
    net_buoyancy_per_module = get_module_net_buoyancy_fixed_volume(bm_mass, seawater_density)
    return net_buoyancy_per_module * num_modules

def get_total_net_buoyancy_sol(bm_mass, num_modules, seawater_density=None):
    """
    Net buoyancy for SoL modules (fixed volume, SoL mass).
    """
    return get_total_net_buoyancy(bm_mass, num_modules, seawater_density)

def get_total_net_buoyancy_eol(bm_mass, num_modules, seawater_density=None):
    """
    Net buoyancy for EoL modules (fixed volume, EoL mass).
    """
    eol_mass = bm_mass * BM_EoL_MASS_FACTOR
    net_buoyancy_per_module = get_module_net_buoyancy_fixed_volume(eol_mass, seawater_density)
    return net_buoyancy_per_module * num_modules 