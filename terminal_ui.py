#!/usr/bin/env python
"""
Terminal-based user interface for OrcaFlex Optimization Tool
"""

import os
import sys
import time
import subprocess
import re
import logging
from config import (
    PARAM_BOUNDS, SOL_PER_POP, NUM_GENERATIONS, 
    MUTATION_PROBABILITY, CROSSOVER_PROBABILITY, KEEP_PARENTS,
    NUM_PROCESSES, MODEL_PATH,
    MIN_SEABED_CLEARANCE, MIN_SURFACE_CLEARANCE, MAX_TENSION_LIMIT, MIN_BEND_RADIUS_LIMIT,
    SAFETY_WEIGHT, COST_WEIGHT, OPTIMIZATION_WEIGHT,
    STATIC_ANALYSIS_TIMEOUT
)
from graphics import clear_terminal, print_header, print_waterline, print_single_floater, print_three_floaters, BLUE, GREEN, WHITE, YELLOW, BOLD, BROWN, RED, CYAN, END
import config

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the application header"""
    clear_terminal()
    print_single_floater()
    print_waterline()
    print(f"\n  {BOLD}ORCAFLEX OPTIMIZATION TOOL{END}\n")
    print(f"  {GREEN}A genetic algorithm approach to optimize buoyancy module configuration{END}\n")


def print_menu(title, options, show_back=True):
    """
    Print a menu with options
    
    Args:
        title (str): Menu title
        options (list): List of option strings
        show_back (bool): Whether to show back/exit option
    """
    print(f"\n  {title}\n")
    
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    
    if show_back:
        print(f"\n  0. Back/Exit")


def get_choice(max_option):
    """
    Get user choice with validation
    
    Args:
        max_option (int): Maximum option number
        
    Returns:
        int: User's choice
    """
    while True:
        try:
            choice = input("\n  Enter your choice: ")
            choice = int(choice)
            if 0 <= choice <= max_option:
                return choice
            else:
                print(f"  Please enter a number between 0 and {max_option}")
        except ValueError:
            print("  Please enter a valid number")


def view_current_config():
    """Display current configuration from config.py"""
    # Reload the config to ensure we have the latest values
    reload_config()
    
    print_header()
    print("  CURRENT CONFIGURATION\n")
    
    print("  Model Configuration:")
    print(f"  - Model Path: {MODEL_PATH}")
    
    print("\n  Optimization Parameters:")
    print(f"  - Population Size: {SOL_PER_POP}")
    print(f"  - Number of Generations: {NUM_GENERATIONS}")
    print(f"  - Mutation Probability: {MUTATION_PROBABILITY}")
    print(f"  - Crossover Probability: {CROSSOVER_PROBABILITY}")
    print(f"  - Parents Kept (Elitism): {KEEP_PARENTS}")
    print(f"  - Number of Processes: {NUM_PROCESSES}")
    
    print("\n  Parameter Bounds:")
    print(f"  - Buoyancy Module Mass: {PARAM_BOUNDS['bm_mass'][0]} to {PARAM_BOUNDS['bm_mass'][1]} kg")
    print(f"  - Start Arc Length: {PARAM_BOUNDS['start_arc_length'][0]} to {PARAM_BOUNDS['start_arc_length'][1]} m")
    print(f"  - Number of Modules: {PARAM_BOUNDS['num_modules'][0]} to {PARAM_BOUNDS['num_modules'][1]}")
    
    input("\n  Press Enter to continue...")


def update_config_file(param_name, new_value):
    """
    Update a parameter in the config.py file
    
    Args:
        param_name (str): Name of the parameter to update
        new_value: New value for the parameter (proper type conversion should be done by caller)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the config file
        with open('config.py', 'r') as f:
            content = f.read()
            lines = content.splitlines()
        
        # Handle different parameter types
        if "PARAM_BOUNDS" in param_name and "['" in param_name:
            # For parameter bounds, we need to update the specific key
            # Extract the key from something like "PARAM_BOUNDS['bm_mass']"
            key = param_name.split("['")[1].split("']")[0] if "['" in param_name else None
            
            if key:
                # Format the new value as a proper Python list
                if isinstance(new_value, list) and len(new_value) == 2:
                    new_value_str = f"[{new_value[0]}, {new_value[1]}]"
                else:
                    return False
                
                # Find the PARAM_BOUNDS dictionary in the content
                param_bounds_found = False
                in_param_bounds = False
                updated = False
                new_lines = []
                
                # Process line by line for better control
                for line in lines:
                    # Check if we're entering the PARAM_BOUNDS dictionary
                    if "PARAM_BOUNDS = {" in line:
                        in_param_bounds = True
                        param_bounds_found = True
                        new_lines.append(line)
                        continue
                    
                    # Check if we're leaving the PARAM_BOUNDS dictionary
                    if in_param_bounds and "}" in line and not line.strip().startswith("#"):
                        in_param_bounds = False
                        if not updated:  # If we didn't find the key, add it before closing
                            indent = "    "  # Typical indentation
                            new_lines.append(f'{indent}"{key}": {new_value_str},')
                            updated = True
                        new_lines.append(line)
                        continue
                    
                    # If we're in PARAM_BOUNDS, check for the key
                    if in_param_bounds and f'"{key}"' in line:
                        # Found the key, update the line
                        parts = line.split(":")
                        if len(parts) >= 2:
                            # Preserve the indentation and key portion
                            indent_and_key = parts[0]
                            new_lines.append(f"{indent_and_key}: {new_value_str},")
                            updated = True
                            continue
                    
                    # For any other line, keep it as is
                    new_lines.append(line)
                
                # If PARAM_BOUNDS wasn't found, return False
                if not param_bounds_found:
                    return False
                
                # Write back the updated content
                with open('config.py', 'w') as f:
                    f.write('\n'.join(new_lines))
                
                return updated
            else:
                return False
        else:
            # For simple parameters, direct replacement line by line
            param_found = False
            new_lines = []
            
            # Convert value to appropriate string representation
            if isinstance(new_value, str):
                value_str = f'"{new_value}"'
            elif isinstance(new_value, bool):
                value_str = str(new_value)
            elif isinstance(new_value, (int, float)):
                value_str = str(new_value)
            else:
                # For complex types we'd need special handling
                return False
            
            # Process line by line
            for line in lines:
                # Check if this line contains the parameter definition
                if line.strip().startswith(param_name + " ="):
                    # Replace the line
                    new_lines.append(f"{param_name} = {value_str}")
                    param_found = True
                else:
                    # Keep the line unchanged
                    new_lines.append(line)
            
            # Only write the file if we found and updated the parameter
            if param_found:
                with open('config.py', 'w') as f:
                    f.write('\n'.join(new_lines))
                return True
            else:
                return False
            
    except Exception as e:
        logging.error(f"Error updating config: {e}")
        return False


def reload_config():
    """Reload the configuration from config.py"""
    try:
        # Use importlib to reload the config module
        import importlib
        import config
        importlib.reload(config)
        
        # Update global variables
        global PARAM_BOUNDS, SOL_PER_POP, NUM_GENERATIONS
        global MUTATION_PROBABILITY, CROSSOVER_PROBABILITY, KEEP_PARENTS
        global NUM_PROCESSES, MODEL_PATH
        global MIN_SEABED_CLEARANCE, MIN_SURFACE_CLEARANCE, MAX_TENSION_LIMIT, MIN_BEND_RADIUS_LIMIT
        global SAFETY_WEIGHT, COST_WEIGHT, OPTIMIZATION_WEIGHT, STATIC_ANALYSIS_TIMEOUT
        
        PARAM_BOUNDS = config.PARAM_BOUNDS
        SOL_PER_POP = config.SOL_PER_POP
        NUM_GENERATIONS = config.NUM_GENERATIONS
        MUTATION_PROBABILITY = config.MUTATION_PROBABILITY
        CROSSOVER_PROBABILITY = config.CROSSOVER_PROBABILITY
        KEEP_PARENTS = config.KEEP_PARENTS
        NUM_PROCESSES = config.NUM_PROCESSES
        MODEL_PATH = config.MODEL_PATH
        MIN_SEABED_CLEARANCE = config.MIN_SEABED_CLEARANCE
        MIN_SURFACE_CLEARANCE = config.MIN_SURFACE_CLEARANCE
        MAX_TENSION_LIMIT = config.MAX_TENSION_LIMIT
        MIN_BEND_RADIUS_LIMIT = config.MIN_BEND_RADIUS_LIMIT
        SAFETY_WEIGHT = config.SAFETY_WEIGHT
        COST_WEIGHT = config.COST_WEIGHT
        OPTIMIZATION_WEIGHT = config.OPTIMIZATION_WEIGHT
        STATIC_ANALYSIS_TIMEOUT = config.STATIC_ANALYSIS_TIMEOUT
        
        return True
    except Exception as e:
        logging.error(f"Error reloading config: {e}")
        return False


def edit_param_bounds():
    """Edit parameter bounds"""
    while True:
        print_header()
        print("  EDIT PARAMETER BOUNDS\n")
        
        options = [
            f"Buoyancy Module Mass: {PARAM_BOUNDS['bm_mass'][0]} to {PARAM_BOUNDS['bm_mass'][1]} kg",
            f"Start Arc Length: {PARAM_BOUNDS['start_arc_length'][0]} to {PARAM_BOUNDS['start_arc_length'][1]} m",
            f"Number of Modules: {PARAM_BOUNDS['num_modules'][0]} to {PARAM_BOUNDS['num_modules'][1]}"
        ]
        
        print_menu("Select parameter to edit:", options)
        choice = get_choice(len(options))
        
        if choice == 0:
            return
        
        param_names = ["bm_mass", "start_arc_length", "num_modules"]
        param_name = param_names[choice - 1]
        
        print_header()
        print(f"  EDIT {param_name.upper()} BOUNDS\n")
        print(f"  Current bounds: {PARAM_BOUNDS[param_name][0]} to {PARAM_BOUNDS[param_name][1]}")
        
        try:
            min_val = float(input(f"\n  Enter new minimum value: "))
            max_val = float(input(f"  Enter new maximum value: "))
            
            if min_val >= max_val:
                print("\n  Error: Minimum value must be less than maximum value")
                input("\n  Press Enter to continue...")
                continue
            
            # Update in-memory value for this session
            PARAM_BOUNDS[param_name] = [min_val, max_val]
            
            # Update the config.py file
            success = update_config_file(f"PARAM_BOUNDS['{param_name}']", [min_val, max_val])
            
            if success:
                print(f"\n  {param_name} bounds updated to: {min_val} to {max_val}")
                print("  Changes saved to config.py")
            else:
                print("\n  Warning: Changes are only in memory for this session")
                print("  Failed to update config.py file")
                
            input("\n  Press Enter to continue...")
            
        except ValueError:
            print("\n  Error: Please enter valid numbers")
            input("\n  Press Enter to continue...")


def edit_seawater_density():
    """Edit seawater density parameter"""
    reload_config()
    print_header()
    print("  EDIT SEAWATER DENSITY\n")
    print(f"  Current seawater density: {config.SEAWATER_DENSITY} kg/m^3")
    try:
        new_value = float(input("\n  Enter new seawater density (kg/m^3): "))
        if new_value <= 0:
            print("\n  Error: Value must be positive")
            input("\n  Press Enter to continue...")
            return
        # Update in-memory value for this session
        config.SEAWATER_DENSITY = new_value
        # Update the config.py file
        success = update_config_file("SEAWATER_DENSITY", new_value)
        if success:
            print(f"\n  Seawater density updated to: {new_value} kg/m^3")
            print("  Changes saved to config.py")
        else:
            print(f"\n  Seawater density updated to: {new_value} for this session only")
            print("  Warning: Failed to update config.py file")
        input("\n  Press Enter to continue...")
    except ValueError:
        print("\n  Error: Please enter a valid number")
        input("\n  Press Enter to continue...")


def edit_ga_params():
    """Edit genetic algorithm parameters"""
    import config
    reload_config()
    # List of GA parameters and their display names
    ga_params = [
        ("NUM_GENERATIONS", "Number of Generations", int),
        ("NUM_PARENTS_MATING", "Number of Parents Mating", int),
        ("SOL_PER_POP", "Solutions per Population", int),
        ("PARENT_SELECTION_TYPE", "Parent Selection Type", str),
        ("KEEP_PARENTS", "Parents Kept (Elitism)", int),
        ("CROSSOVER_TYPE", "Crossover Type", str),
        ("CROSSOVER_PROBABILITY", "Crossover Probability", float),
        ("MUTATION_TYPE", "Mutation Type", str),
        ("MUTATION_PERCENT_GENES", "Mutation Percent Genes", int),
        ("MUTATION_PROBABILITY", "Mutation Probability", float),
        ("MUTATION_BY_REPLACEMENT", "Mutation By Replacement", bool),
        ("GENE_TYPE", "Gene Type", str),
        ("INIT_RANGE_LOW", "Initial Range Low", float),
        ("INIT_RANGE_HIGH", "Initial Range High", float),
        ("ALLOW_DUPLICATE_GENES", "Allow Duplicate Genes", bool),
        ("STOP_CRITERIA", "Stop Criteria", str),
    ]
    while True:
        print_header()
        print("  EDIT GENETIC ALGORITHM PARAMETERS\n")
        options = [f"{display}: {getattr(config, param)}" for param, display, _ in ga_params]
        print_menu("Select parameter to edit:", options)
        choice = get_choice(len(options))
        if choice == 0:
            return
        param_name, display_name, param_type = ga_params[choice - 1]
        current_value = getattr(config, param_name)
        print_header()
        print(f"  EDIT {display_name}\n")
        print(f"  Current value: {current_value}")
        try:
            if param_type == int:
                new_value = int(input(f"\n  Enter new integer value: "))
            elif param_type == float:
                new_value = float(input(f"\n  Enter new float value: "))
            elif param_type == bool:
                val = input(f"\n  Enter new value (True/False): ").strip().lower()
                if val in ["true", "1", "yes", "y"]:
                    new_value = True
                elif val in ["false", "0", "no", "n"]:
                    new_value = False
                else:
                    raise ValueError("Value must be True or False")
            else:  # str
                new_value = input(f"\n  Enter new string value: ")
                if new_value.lower() == "none":
                    new_value = None
            # Special handling for GENE_TYPE
            if param_name == "GENE_TYPE":
                if new_value == "float":
                    new_value = float
                elif new_value == "int":
                    new_value = int
                elif new_value == "str":
                    new_value = str
                else:
                    raise ValueError("GENE_TYPE must be 'float', 'int', or 'str'")
            # Update config.py
            success = update_config_file(param_name, new_value)
            if success:
                print(f"\n  {display_name} updated to: {new_value}")
                print("  Changes saved to config.py")
            else:
                print(f"\n  {display_name} updated to: {new_value} for this session only")
                print("  Warning: Failed to update config.py file")
            reload_config()
            input("\n  Press Enter to continue...")
        except ValueError as e:
            print(f"\n  Error: {str(e) if str(e) else 'Please enter a valid value'}")
            input("\n  Press Enter to continue...")


def edit_execution_params():
    """Edit execution parameters including timeouts and number of processes"""
    import config
    global STATIC_ANALYSIS_TIMEOUT, NUM_PROCESSES
    while True:
        print_header()
        print("  EDIT EXECUTION PARAMETERS\n")
        print(f"  1. Static Analysis Timeout: {config.STATIC_ANALYSIS_TIMEOUT} seconds")
        print(f"  2. Multiprocess Timeout Multiplier: {getattr(config, 'MULTIPROCESS_TIMEOUT_MULTIPLIER', 2.0)}x")
        print(f"  3. Number of Processes (Workers): {config.NUM_PROCESSES}")
        print("\n  0. Back/Exit\n")
        choice = get_choice(3)
        if choice == 0:
            return
        elif choice == 1:
            print_header()
            print("  EDIT STATIC ANALYSIS TIMEOUT\n")
            print(f"  Current value: {config.STATIC_ANALYSIS_TIMEOUT} seconds")
            try:
                new_value = float(input("\n  Enter new timeout in seconds (e.g. 10): "))
                if new_value <= 0:
                    raise ValueError("Timeout must be positive")
                STATIC_ANALYSIS_TIMEOUT = new_value
                success = update_config_file("STATIC_ANALYSIS_TIMEOUT", new_value)
                if success:
                    print(f"\n  STATIC_ANALYSIS_TIMEOUT updated to: {new_value} seconds")
                    print("  Changes saved to config.py")
                else:
                    print(f"\n  STATIC_ANALYSIS_TIMEOUT updated to: {new_value} for this session only")
                    print("  Warning: Failed to update config.py file")
                reload_config()
                input("\n  Press Enter to continue...")
            except ValueError as e:
                print(f"\n  Error: {str(e) if str(e) else 'Please enter a valid value'}")
                input("\n  Press Enter to continue...")
        elif choice == 2:
            print_header()
            print("  EDIT MULTIPROCESS TIMEOUT MULTIPLIER\n")
            print(f"  Current value: {getattr(config, 'MULTIPROCESS_TIMEOUT_MULTIPLIER', 2.0)}x")
            try:
                new_value = float(input("\n  Enter new multiplier (e.g. 2 for 2x the single-process timeout): "))
                if new_value <= 0:
                    raise ValueError("Multiplier must be positive")
                success = update_config_file("MULTIPROCESS_TIMEOUT_MULTIPLIER", new_value)
                if success:
                    print(f"\n  MULTIPROCESS_TIMEOUT_MULTIPLIER updated to: {new_value}x")
                    print("  Changes saved to config.py")
                else:
                    print(f"\n  MULTIPROCESS_TIMEOUT_MULTIPLIER updated to: {new_value}x for this session only")
                    print("  Warning: Failed to update config.py file")
                reload_config()
                input("\n  Press Enter to continue...")
            except ValueError as e:
                print(f"\n  Error: {str(e) if str(e) else 'Please enter a valid value'}")
                input("\n  Press Enter to continue...")
        elif choice == 3:
            print_header()
            print("  EDIT NUMBER OF PROCESSES (WORKERS)\n")
            print(f"  Current value: {config.NUM_PROCESSES}")
            try:
                new_value = int(input("\n  Enter new number of processes (e.g. 4): "))
                if new_value <= 0:
                    raise ValueError("Number of processes must be positive")
                NUM_PROCESSES = new_value
                success = update_config_file("NUM_PROCESSES", new_value)
                if success:
                    print(f"\n  NUM_PROCESSES updated to: {new_value}")
                    print("  Changes saved to config.py")
                else:
                    print(f"\n  NUM_PROCESSES updated to: {new_value} for this session only")
                    print("  Warning: Failed to update config.py file")
                reload_config()
                input("\n  Press Enter to continue...")
            except ValueError as e:
                print(f"\n  Error: {str(e) if str(e) else 'Please enter a valid integer value'}")
                input("\n  Press Enter to continue...")


def edit_safety_params():
    """Edit safety parameters"""
    global MIN_SEABED_CLEARANCE, MIN_SURFACE_CLEARANCE, MAX_TENSION_LIMIT, MIN_BEND_RADIUS_LIMIT
    
    while True:
        print_header()
        print("  EDIT SAFETY PARAMETERS\n")
        
        options = [
            f"Minimum Seabed Clearance: {MIN_SEABED_CLEARANCE} m",
            f"Minimum Surface Clearance: {MIN_SURFACE_CLEARANCE} m",
            f"Maximum Tension Limit: {MAX_TENSION_LIMIT} kN",
            f"Minimum Bend Radius Limit: {MIN_BEND_RADIUS_LIMIT} m"
        ]
        
        print_menu("Select parameter to edit:", options)
        choice = get_choice(len(options))
        
        if choice == 0:
            return
        
        param_names = ["MIN_SEABED_CLEARANCE", "MIN_SURFACE_CLEARANCE", "MAX_TENSION_LIMIT", "MIN_BEND_RADIUS_LIMIT"]
        param_name = param_names[choice - 1]
        current_value = [MIN_SEABED_CLEARANCE, MIN_SURFACE_CLEARANCE, MAX_TENSION_LIMIT, MIN_BEND_RADIUS_LIMIT][choice - 1]
        
        print_header()
        print(f"  EDIT {param_name}\n")
        print(f"  Current value: {current_value}")
        
        try:
            new_value = float(input(f"\n  Enter new value: "))
            if new_value <= 0:
                raise ValueError("Value must be positive")
            
            # Update the parameter value in memory
            if param_name == "MIN_SEABED_CLEARANCE":
                MIN_SEABED_CLEARANCE = new_value
            elif param_name == "MIN_SURFACE_CLEARANCE":
                MIN_SURFACE_CLEARANCE = new_value
            elif param_name == "MAX_TENSION_LIMIT":
                MAX_TENSION_LIMIT = new_value
            elif param_name == "MIN_BEND_RADIUS_LIMIT":
                MIN_BEND_RADIUS_LIMIT = new_value
            
            # Update the config.py file
            success = update_config_file(param_name, new_value)
            
            if success:
                print(f"\n  {param_name} updated to: {new_value}")
                print("  Changes saved to config.py")
            else:
                print(f"\n  {param_name} updated to: {new_value} for this session only")
                print("  Warning: Failed to update config.py file")
                
            input("\n  Press Enter to continue...")
            
        except ValueError as e:
            print(f"\n  Error: {str(e) if str(e) else 'Please enter a valid value'}")
            input("\n  Press Enter to continue...")


def edit_fitness_weightings():
    """Edit fitness weightings (SAFETY_WEIGHT, COST_WEIGHT, OPTIMIZATION_WEIGHT)"""
    global SAFETY_WEIGHT, COST_WEIGHT, OPTIMIZATION_WEIGHT
    while True:
        print_header()
        print("  EDIT FITNESS WEIGHTINGS\n")
        print("  These weights determine the relative importance of each fitness component.")
        print("  For best results, weights should sum to 1.0.\n")
        print(f"  1. Safety Weight: {SAFETY_WEIGHT}")
        print(f"  2. Cost Weight: {COST_WEIGHT}")
        print(f"  3. Optimization Weight: {OPTIMIZATION_WEIGHT}")
        print("\n  0. Back/Exit\n")
        try:
            choice = int(input("  Select parameter to edit: "))
        except ValueError:
            print("\n  Error: Please enter a valid number")
            input("\n  Press Enter to continue...")
            continue
        if choice == 0:
            return
        param_names = ["SAFETY_WEIGHT", "COST_WEIGHT", "OPTIMIZATION_WEIGHT"]
        if 1 <= choice <= 3:
            param_name = param_names[choice - 1]
            current_value = [SAFETY_WEIGHT, COST_WEIGHT, OPTIMIZATION_WEIGHT][choice - 1]
            print_header()
            print(f"  EDIT {param_name}\n")
            print(f"  Current value: {current_value}")
            print("  Enter a new value between 0.0 and 1.0. For best results, all weights should sum to 1.0.\n")
            try:
                new_value = float(input("  Enter new value: "))
                if not 0.0 <= new_value <= 1.0:
                    raise ValueError("Value must be between 0.0 and 1.0")
                # Update in-memory value
                if param_name == "SAFETY_WEIGHT":
                    SAFETY_WEIGHT = new_value
                elif param_name == "COST_WEIGHT":
                    COST_WEIGHT = new_value
                elif param_name == "OPTIMIZATION_WEIGHT":
                    OPTIMIZATION_WEIGHT = new_value
                # Update config.py
                success = update_config_file(param_name, new_value)
                if success:
                    print(f"\n  {param_name} updated to: {new_value}")
                    print("  Changes saved to config.py")
                else:
                    print(f"\n  {param_name} updated to: {new_value} for this session only")
                    print("  Warning: Failed to update config.py file")
                # Reload config to update globals
                reload_config()
                input("\n  Press Enter to continue...")
            except ValueError as e:
                print(f"\n  Error: {str(e) if str(e) else 'Please enter a valid value'}")
                input("\n  Press Enter to continue...")
        else:
            print("\n  Invalid choice. Please select a valid option.")
            input("\n  Press Enter to continue...")


def edit_csv_logging_mode():
    """Edit CSV logging mode (Reset or Append)"""
    import config
    from importlib import reload
    print_header()
    print("  CSV LOGGING MODE\n")
    current_mode = "Append" if config.CSV_APPEND_MODE else "Reset"
    print(f"  CSV Logging Mode [Reset/Append] (current: {current_mode})\n")
    print("  Reset:  Start a new CSV file each run (better for reproducibility)")
    print("  Append: Add new results to the same CSV (good for data gathering)\n")
    print("  1. Reset (overwrite CSV at start of each run)")
    print("  2. Append (add to CSV across runs)")
    print("\n  0. Back/Exit\n")
    while True:
        try:
            choice = int(input("  Select logging mode: "))
        except ValueError:
            print("\n  Error: Please enter a valid number")
            continue
        if choice == 0:
            return
        elif choice == 1:
            new_mode = False
        elif choice == 2:
            new_mode = True
        else:
            print("\n  Invalid choice. Please select 1 or 2.")
            continue
        # Update config.py
        success = update_config_file("CSV_APPEND_MODE", new_mode)
        if success:
            print(f"\n  CSV_APPEND_MODE updated to: {'Append' if new_mode else 'Reset'}")
            print("  Changes saved to config.py")
            reload(config)
        else:
            print(f"\n  CSV_APPEND_MODE updated to: {'Append' if new_mode else 'Reset'} for this session only")
            print("  Warning: Failed to update config.py file")
        input("\n  Press Enter to continue...")
        return


def edit_plotting_settings():
    """Edit plotting-related settings"""
    import config
    from importlib import reload
    while True:
        print_header()
        print("  EDIT PLOTTING SETTINGS\n")
        print(f"  1. Show Metrics/Config Box on Main Plot: {'ON' if getattr(config, 'PLOT_SHOW_METRICS_BOX', True) else 'OFF'}")
        print(f"  2. Show Title on Profile Plots: {'ON' if getattr(config, 'PLOT_SHOW_TITLE', True) else 'OFF'}")
        print(f"  3. Show Cable Legend Box: {'ON' if getattr(config, 'PLOT_SHOW_CABLE_LEGEND', True) else 'OFF'}")
        print("\n  0. Back/Exit\n")
        choice = get_choice(3)
        if choice == 0:
            return
        elif choice == 1:
            new_value = not getattr(config, 'PLOT_SHOW_METRICS_BOX', True)
            success = update_config_file('PLOT_SHOW_METRICS_BOX', new_value)
            reload(config)
        elif choice == 2:
            new_value = not getattr(config, 'PLOT_SHOW_TITLE', True)
            success = update_config_file('PLOT_SHOW_TITLE', new_value)
            reload(config)
        elif choice == 3:
            new_value = not getattr(config, 'PLOT_SHOW_CABLE_LEGEND', True)
            success = update_config_file('PLOT_SHOW_CABLE_LEGEND', new_value)
            reload(config)


def edit_buoyancy_settings():
    """Edit buoyancy-related settings"""
    import config
    from importlib import reload
    while True:
        print_header()
        print("  EDIT BUOYANCY SETTINGS\n")
        print(f"  1. Seawater Density: {getattr(config, 'SEAWATER_DENSITY', 1025.0)} kg/m^3")
        print(f"  2. Buoyancy Module Volume: {getattr(config, 'BM_VOLUME', 1.0)} m^3")
        print(f"  3. Buoyancy Module Spacing: {getattr(config, 'BM_SPACING', 5.0)} m")
        print("\n  0. Back/Exit\n")
        choice = get_choice(3)
        if choice == 0:
            return
        elif choice == 1:
            try:
                new_value = float(input("\n  Enter new seawater density (kg/m^3): "))
                if new_value <= 0:
                    print("\n  Error: Value must be positive")
                    input("\n  Press Enter to continue...")
                    continue
                success = update_config_file('SEAWATER_DENSITY', new_value)
                reload(config)
            except ValueError:
                print("\n  Error: Please enter a valid number")
                input("\n  Press Enter to continue...")
        elif choice == 2:
            try:
                new_value = float(input("\n  Enter new buoyancy module volume (m^3): "))
                if new_value <= 0:
                    print("\n  Error: Value must be positive")
                    input("\n  Press Enter to continue...")
                    continue
                success = update_config_file('BM_VOLUME', new_value)
                reload(config)
            except ValueError:
                print("\n  Error: Please enter a valid number")
                input("\n  Press Enter to continue...")
        elif choice == 3:
            try:
                new_value = float(input("\n  Enter new buoyancy module spacing (m): "))
                if new_value <= 0:
                    print("\n  Error: Value must be positive")
                    input("\n  Press Enter to continue...")
                    continue
                success = update_config_file('BM_SPACING', new_value)
                reload(config)
            except ValueError:
                print("\n  Error: Please enter a valid number")
                input("\n  Press Enter to continue...")


def run_optimization():
    """Run optimization with current settings"""
    print_header()
    print("  RUN OPTIMIZATION\n")
    
    options = [
        "Run multi-process optimization",
        "Generate plots from current model"
    ]
    
    print_menu("Select optimization mode:", options)
    choice = get_choice(len(options))
    
    if choice == 0:
        return
    
    print_header()
    print("  STARTING OPTIMIZATION...\n")
    
    # Build the command based on user choice
    if choice == 1:
        cmd = ["python", "main.py", "--optimize", "--multi"]
    elif choice == 2:
        cmd = ["python", "main.py", "--plot-only"]
    
    print(f"  Running command: {' '.join(cmd)}\n")
    print("  This may take a while depending on your settings...\n")
    
    # Launch the process
    print("\n  Starting optimization process...\n")
    time.sleep(1)  # Short delay for user to read
    
    try:
        # Create an environment with updated config variables
        env = os.environ.copy()
        # For newer Python versions, use this to update the configuration before running
        env["SOL_PER_POP"] = str(SOL_PER_POP)
        env["NUM_GENERATIONS"] = str(NUM_GENERATIONS)
        env["MUTATION_PROBABILITY"] = str(MUTATION_PROBABILITY)
        env["CROSSOVER_PROBABILITY"] = str(CROSSOVER_PROBABILITY)
        env["KEEP_PARENTS"] = str(KEEP_PARENTS)
        env["NUM_PROCESSES"] = str(NUM_PROCESSES)
        
        # Run the command with the updated environment
        import sys
        subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    except Exception as e:
        print(f"\n  Error running optimization: {e}")
    
    print("\n  Optimization process completed.")
    input("\n  Press Enter to continue...")


def settings_menu():
    """Settings menu"""
    while True:
        print_header()
        print("  SETTINGS MENU\n")
        options = [
            "Edit Parameter Bounds",
            "Edit Genetic Algorithm Parameters",
            "Edit Execution Parameters",
            "Edit Safety Parameters",
            "Edit Fitness Weightings",
            "Edit CSV Logging Mode",
            "Plotting Settings",
            "Buoyancy Settings"
        ]
        print_menu("Select a setting to edit:", options)
        choice = get_choice(len(options))
        if choice == 0:
            return
        elif choice == 1:
            edit_param_bounds()
        elif choice == 2:
            edit_ga_params()
        elif choice == 3:
            edit_execution_params()
        elif choice == 4:
            edit_safety_params()
        elif choice == 5:
            edit_fitness_weightings()
        elif choice == 6:
            edit_csv_logging_mode()
        elif choice == 7:
            edit_plotting_settings()
        elif choice == 8:
            edit_buoyancy_settings()


def main_menu():
    """Main menu"""
    while True:
        print_header()
        
        options = [
            "Run optimization",
            "Run single static analysis",
            "Settings",
            "Help",
            "Exit"
        ]
        
        print_menu("Main Menu", options, show_back=False)
        choice = get_choice(len(options))
        
        if choice == 1:
            run_optimization()
        elif choice == 2:
            # Run single static analysis
            print_header()
            print("  RUN SINGLE STATIC ANALYSIS\n")
            mass = input("  Enter BM mass (kg) or leave blank for default: ")
            start = input("  Enter start arc length (m) or leave blank for default: ")
            num = input("  Enter number of modules or leave blank for default: ")
            cmd = ["python", "main.py", "--single-analysis"]
            if mass.strip():
                cmd.extend(["--mass", mass])
            if start.strip():
                cmd.extend(["--start", start])
            if num.strip():
                cmd.extend(["--num", num])
            print(f"\n  Running command: {' '.join(cmd)}\n")
            subprocess.run(cmd)
            input("\n  Press Enter to continue...")
        elif choice == 3:
            settings_menu()
        elif choice == 4:
            show_help()
        elif choice == 5:
            sys.exit(0)


def show_help():
    """Show help information"""
    print_header()
    print("  HELP\n")
    
    help_text = [
        "This tool uses genetic algorithms to optimize buoyancy module",
        "configurations for OrcaFlex cable systems.",
        "",
        "Main Features:",
        "- Set optimization parameters and bounds",
        "- Run single or multi-process optimization",
        "- Generate plots of optimization results",
        "",
        "Tips:",
        "- For faster results, use multi-process optimization",
        "- Check output/ directory for generated plots",
        "- Changes made through this interface are saved to config.py",
        "- All configuration changes take effect immediately",
        "",
        "Requirements:",
        "- OrcaFlex with OrcFxAPI Python module",
        "- Python packages: numpy, matplotlib, pygad, tqdm"
    ]
    
    for line in help_text:
        print(f"  {line}")
    
    input("\n  Press Enter to continue...")


if __name__ == "__main__":
    try:
        # Check if OrcFxAPI is available
        try:
            import OrcFxAPI
            print("OrcFxAPI successfully imported")
        except ImportError:
            print("WARNING: OrcFxAPI not found. This tool requires OrcaFlex to be installed.")
            print("You can continue, but optimization functions will fail.")
            time.sleep(2)
        
        # Ensure we have the latest configuration
        reload_config()
        
        # Start the main menu
        main_menu()
    except KeyboardInterrupt:
        print("\n\nExiting application...")
        sys.exit(0) 