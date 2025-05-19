"""
Test script for configuring buoyancy modules on cables
and optimizing the configuration using PyGAD
"""

import os
import argparse
import logging
import config
from config import (
    MODEL_PATH, CABLE_NAMES, BM_NAMES,
    BM_EoL_MASS_FACTOR,
    DEFAULT_WORKING_BM_MASS, DEFAULT_WORKING_START_ARC_LENGTH, DEFAULT_WORKING_NUM_MODULES,
    BM_SPACING, SEAWATER_DENSITY
)
from orcaflex_utils import (
    load_model, update_bm_mass, configure_buoyancy_modules, 
    run_static_analysis, save_model, plot_cable_profiles, plot_cable_profiles_by_offset,
    get_seabed_clearance, get_surface_clearance, get_max_tension, get_min_bend_radius, get_suspended_length,
    get_total_net_buoyancy
)
# Import optimization modules
# from optimization_module import run_optimization as run_single_process
from multiprocess_optimization_module import run_optimization as run_multiprocess
# Note: optimization_utils.py contains shared functions used by both optimization modules

# Check for environment variables that might override config values
def get_env_config():
    """Get configuration from environment variables if they exist"""
    config_updates = {}
    
    # Check for GA parameters
    if "POPULATION_SIZE" in os.environ:
        try:
            config_updates["POPULATION_SIZE"] = int(os.environ["POPULATION_SIZE"])
        except ValueError:
            pass
            
    if "NUM_GENERATIONS" in os.environ:
        try:
            config_updates["NUM_GENERATIONS"] = int(os.environ["NUM_GENERATIONS"])
        except ValueError:
            pass
    
    if "MUTATION_PROBABILITY" in os.environ:
        try:
            config_updates["MUTATION_PROBABILITY"] = float(os.environ["MUTATION_PROBABILITY"])
        except ValueError:
            pass
    
    if "CROSSOVER_PROBABILITY" in os.environ:
        try:
            config_updates["CROSSOVER_PROBABILITY"] = float(os.environ["CROSSOVER_PROBABILITY"])
        except ValueError:
            pass
    
    if "KEEP_PARENTS" in os.environ:
        try:
            config_updates["KEEP_PARENTS"] = int(os.environ["KEEP_PARENTS"])
        except ValueError:
            pass
    
    if "NUM_PROCESSES" in os.environ:
        try:
            config_updates["NUM_PROCESSES"] = int(os.environ["NUM_PROCESSES"])
        except ValueError:
            pass
    
    return config_updates

def setup_logging(console_output=True):
    """
    Configure logging to save to log.txt
    
    Args:
        console_output (bool): Whether to show logs in console as well as file
    """
    # Configure logging to overwrite the file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="log.txt",
        filemode="w",  # Use 'w' to overwrite, 'a' to append
        encoding="utf-8"
    )
    
    # Add console handler only if requested
    if console_output:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    
    # Log startup information
    logging.info("=" * 80)
    logging.info("Starting buoyancy module configuration")
    logging.info("=" * 80)
    logging.info(f"Model path: {MODEL_PATH}")
    logging.info(f"Module spacing: {BM_SPACING} m")
    logging.info("=" * 80)

def configure_model(bm_mass=None, start_arc_length=None, num_modules=None, module_spacing=None):
    """
    Configure buoyancy modules with specified parameters
    
    Args:
        bm_mass (float, optional): Mass of buoyancy module in kg
        start_arc_length (float, optional): Starting arc length in meters
        num_modules (int, optional): Number of modules to distribute
        module_spacing (float, optional): Spacing between modules in meters
        
    Returns:
        OrcFxAPI.Model: The configured model
    """
    # Use working configuration defaults if not specified
    if bm_mass is None:
        bm_mass = DEFAULT_WORKING_BM_MASS
    else:
        bm_mass = int(round(bm_mass))
        
    if start_arc_length is None:
        start_arc_length = DEFAULT_WORKING_START_ARC_LENGTH
    else:
        start_arc_length = int(round(start_arc_length))
        
    if num_modules is None:
        num_modules = DEFAULT_WORKING_NUM_MODULES
    else:
        num_modules = int(num_modules)
        
    if module_spacing is None:
        module_spacing = BM_SPACING
    
    # Log the configuration being used
    logging.info(f"SoL BM mass: {bm_mass} kg")
    logging.info(f"EoL BM mass: {bm_mass * BM_EoL_MASS_FACTOR} kg")
    logging.info(f"Start arc length: {start_arc_length} m")
    logging.info(f"Number of modules: {num_modules}")
    logging.info(f"Module spacing: {module_spacing} m")
        
    # Create a directory for output files
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = load_model(MODEL_PATH)
    
    # Separate SoL and EoL cables
    sol_cables = [cable for cable in CABLE_NAMES if "SoL" in cable]
    eol_cables = [cable for cable in CABLE_NAMES if "EoL" in cable]
    
    logging.info(f"SoL cables: {sol_cables}")
    logging.info(f"EoL cables: {eol_cables}")
    
    # Calculate EoL mass (5% more than SoL)
    eol_bm_mass = bm_mass * BM_EoL_MASS_FACTOR
    
    # Update buoyancy module masses
    update_bm_mass(model, BM_NAMES["SoL"], bm_mass)
    update_bm_mass(model, BM_NAMES["EoL"], eol_bm_mass)
    
    # Configure buoyancy modules on SoL cables
    for cable in sol_cables:
        configure_buoyancy_modules(
            model, cable, BM_NAMES["SoL"], 
            start_arc_length, num_modules, module_spacing
        )
    
    # Configure buoyancy modules on EoL cables
    for cable in eol_cables:
        configure_buoyancy_modules(
            model, cable, BM_NAMES["EoL"], 
            start_arc_length, num_modules, module_spacing
        )
    
    # Run a static analysis to update the model for plotting
    logging.info("Running static analysis...")
    success = run_static_analysis(model)
    
    if not success:
        logging.error("Static analysis failed - plots may not show accurate results")
    
    # Save the model
    save_model(model, MODEL_PATH)
    logging.info(f"Configuration complete and saved to {MODEL_PATH}")
    
    return model, success

def generate_plots(model, success):
    """
    Generate plots for the configured model
    
    Args:
        model (OrcFxAPI.Model): The OrcaFlex model
        success (bool): Whether static analysis was successful
    """
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    # Generate plots
    if success:
        logging.info("Generating cable profile plot...")
        cable_plot_path = os.path.join(output_dir, "cable_profiles.png")
        try:
            cable_fig = plot_cable_profiles(model, save_path=cable_plot_path)
            logging.info(f"Cable profile plot created and saved to {cable_plot_path}")
            plot_cable_profiles_by_offset(model, output_dir)
            logging.info("Per-offset cable profile plots created.")
        except Exception as e:
            logging.error(f"Error creating cable profile plot: {e}")

        # --- Write metrics.txt ---
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            # Configuration section
            f.write("CONFIGURATION PARAMETERS\n")
            f.write("=======================\n")
            main_sol = next((c for c in CABLE_NAMES if c == "SoL"), None)
            main_eol = next((c for c in CABLE_NAMES if c == "EoL"), None)
            sol_bm_mass = model[BM_NAMES["SoL"]].Mass if main_sol else None
            eol_bm_mass = model[BM_NAMES["EoL"]].Mass if main_eol else None
            sol_num_modules = model[main_sol].NumberOfAttachments if main_sol else None
            eol_num_modules = model[main_eol].NumberOfAttachments if main_eol else None
            sol_start_arc = model[main_sol].AttachmentZ[0] if (main_sol and model[main_sol].NumberOfAttachments > 0) else None
            eol_start_arc = model[main_eol].AttachmentZ[0] if (main_eol and model[main_eol].NumberOfAttachments > 0) else None
            module_spacing = BM_SPACING
            seawater_density = SEAWATER_DENSITY
            f.write(f"SoL BM mass: {sol_bm_mass:.2f} kg\n" if sol_bm_mass is not None else "")
            f.write(f"EoL BM mass: {eol_bm_mass:.2f} kg\n" if eol_bm_mass is not None else "")
            f.write(f"Number of modules (SoL): {sol_num_modules}\n" if sol_num_modules is not None else "")
            f.write(f"Number of modules (EoL): {eol_num_modules}\n" if eol_num_modules is not None else "")
            f.write(f"Start arc length (SoL): {sol_start_arc:.2f} m\n" if sol_start_arc is not None else "")
            f.write(f"Start arc length (EoL): {eol_start_arc:.2f} m\n" if eol_start_arc is not None else "")
            f.write(f"Module spacing: {module_spacing:.2f} m\n")
            f.write(f"Seawater density: {seawater_density:.2f} kg/m^3\n")
            f.write(f"EoL mass factor: {BM_EoL_MASS_FACTOR}\n")
            f.write("\n")
            # Total net buoyancy section
            if main_sol:
                sol_net_buoy = get_total_net_buoyancy(sol_bm_mass, sol_num_modules, SEAWATER_DENSITY)
                f.write(f"Total net buoyancy (SoL): {sol_net_buoy:.2f} kg\n")
            if main_eol:
                eol_net_buoy = get_total_net_buoyancy(eol_bm_mass, eol_num_modules, SEAWATER_DENSITY)
                f.write(f"Total net buoyancy (EoL): {eol_net_buoy:.2f} kg\n")
            f.write("\n")
            # Suspended lengths section
            f.write("SUSPENDED CABLE LENGTHS\n")
            f.write("=======================\n")
            for cable in CABLE_NAMES:
                suspended_length = get_suspended_length(model, cable)
                f.write(f"{cable}: {suspended_length:.2f} m\n")
            f.write("\n")
            # Critical values section
            f.write("CRITICAL VALUES AND LIMITS\n")
            f.write("==========================\n")
            # Find critical values across all cables
            min_seabed = float('inf')
            min_surface = float('inf')
            max_tension = 0.0
            min_radius = float('inf')
            for cable in CABLE_NAMES:
                seabed_clearance, _ = get_seabed_clearance(model, cable)
                surface_clearance, _ = get_surface_clearance(model, cable)
                tension, _, _ = get_max_tension(model, cable)
                radius, _, _ = get_min_bend_radius(model, cable)
                if 0 < seabed_clearance < min_seabed:
                    min_seabed = seabed_clearance
                if 0 < surface_clearance < min_surface:
                    min_surface = surface_clearance
                if tension > max_tension:
                    max_tension = tension
                if 0 < radius < min_radius:
                    min_radius = radius
            f.write(f"Min seabed clearance: {min_seabed:.2f} m (Limit: {getattr(config, 'MIN_SEABED_CLEARANCE', 0):.2f} m)\n")
            f.write(f"Min surface clearance: {min_surface:.2f} m (Limit: {getattr(config, 'MIN_SURFACE_CLEARANCE', 0):.2f} m)\n")
            f.write(f"Max tension: {max_tension:.2f} kN (Limit: {getattr(config, 'MAX_TENSION_LIMIT', 0):.2f} kN)\n")
            f.write(f"Min bend radius: {min_radius:.2f} m (Limit: {getattr(config, 'MIN_BEND_RADIUS_LIMIT', 0):.2f} m)\n")
            f.write("\n")
            # Per-cable metrics section
            f.write("PER-CABLE METRICS\n")
            f.write("=================\n")
            for cable in CABLE_NAMES:
                f.write(f"Cable: {cable}\n")
                seabed_clearance, seabed_arc = get_seabed_clearance(model, cable)
                surface_clearance, surface_arc = get_surface_clearance(model, cable)
                max_tension, tension_arc, tension_loc = get_max_tension(model, cable)
                min_radius, radius_arc, radius_loc = get_min_bend_radius(model, cable)
                suspended_length = get_suspended_length(model, cable)
                f.write(f"  Min seabed clearance: {seabed_clearance:.2f} m @ {seabed_arc:.1f} m\n")
                f.write(f"  Min surface clearance: {surface_clearance:.2f} m @ {surface_arc:.1f} m\n")
                f.write(f"  Max tension: {max_tension:.2f} kN @ {tension_arc:.1f} m ({tension_loc})\n")
                f.write(f"  Min bend radius: {min_radius:.2f} m @ {radius_arc:.1f} m ({radius_loc})\n")
                f.write(f"  Suspended length: {suspended_length:.2f} m\n")
                f.write("\n")
            f.write("(All values are based on the current configuration and static analysis results.)\n")
        logging.info(f"Metrics written to {metrics_path}")
    else:
        logging.warning("Skipping plots due to static analysis failure")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Configure and optimize buoyancy modules")
    parser.add_argument("--optimize", action="store_true", help="Run optimization")
    parser.add_argument("--multi", action="store_true", help="Use multiprocessing optimization")
    parser.add_argument("--mass", type=float, default=None, help="Buoyancy module mass in kg")
    parser.add_argument("--start", type=float, default=None, help="Starting arc length in meters")
    parser.add_argument("--num", type=int, default=None, help="Number of modules")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from current model")
    parser.add_argument("--single-analysis", action="store_true", help="Run a single static analysis with specified configuration")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Get environment config overrides
    env_config = get_env_config()
    if env_config:
        logging.info("Using configuration overrides from environment variables:")
        for key, value in env_config.items():
            logging.info(f"  {key}: {value}")
    
    if args.plot_only:
        # Load the model and run static analysis before plotting
        model = load_model(MODEL_PATH)
        logging.info("Running static analysis for plot...")
        success = run_static_analysis(model)
        generate_plots(model, success)
        logging.info("Plots generated from existing model")
        return
    
    if args.single_analysis:
        # Run a single static analysis with specified configuration
        logging.info("Running single static analysis with specified configuration...")
        model, success = configure_model(
            bm_mass=args.mass,
            start_arc_length=args.start,
            num_modules=args.num
        )
        if success:
            generate_plots(model, success)
        logging.info("Single static analysis completed")
        return
    
    if args.optimize:
        # Run optimization
        logging.info("Starting optimization process...")
        # Use file-only logging during optimization (multiprocess_optimization_module handles its own console output)
        # First remove any existing console handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)
        
        # Only use multiprocess optimizer
        logging.info("Using multiprocessing optimization")
        solution, fitness = run_multiprocess(config_overrides=env_config)

        if fitness != float('-inf'):
            # Load the best solution model
            try:
                model = load_model("BestSolution.dat")
                # Run static analysis on the model before plotting
                success = run_static_analysis(model)
                # Only generate cable profile plot for best solution
                if success:
                    generate_plots(model, success)
            except Exception as e:
                logging.error(f"Error loading best solution model: {e}")
                logging.info("Using solution parameters with original model instead...")
                model, success = configure_model(
                    bm_mass=int(round(solution["bm_mass"])),
                    start_arc_length=solution["start_arc_length"],
                    num_modules=solution["num_modules"]
                )
                # Only generate cable profile plot for best solution
                if success:
                    generate_plots(model, success)
        else:
            logging.warning("Optimization failed to find a valid solution")
            logging.info("Using working configuration for fallback plots...")
            model, success = configure_model()
            if success:
                generate_plots(model, success)
    else:
        # Configure with specified or default parameters
        model, success = configure_model(
            bm_mass=args.mass,
            start_arc_length=args.start,
            num_modules=args.num
        )
        
        # Generate plots
        generate_plots(model, success)
    
    logging.info("All tasks completed successfully")

if __name__ == "__main__":
    main()