"""
Shared utility functions for optimization modules
"""

import time
import logging
import numpy as np
import os
from config import (
    MODEL_PATH, CABLE_NAMES, BM_NAMES,
    PARAM_BOUNDS, BM_EoL_MASS_FACTOR,
    BM_SPACING
)
from orcaflex_utils import (
    load_model, update_bm_mass, configure_buoyancy_modules, 
    run_static_analysis, save_model
)

# Setup logger
logger = logging.getLogger(__name__)

def normalize_solution(solution):
    """
    Normalize solution values to the range [0, 1] based on parameter bounds.
    
    Args:
        solution (dict): Dictionary containing parameter values
        
    Returns:
        list: Normalized parameter values in the range [0, 1]
    """
    normalized = []
    
    # Normalize BM mass
    normalized.append((solution["bm_mass"] - PARAM_BOUNDS["bm_mass"][0]) / 
                    (PARAM_BOUNDS["bm_mass"][1] - PARAM_BOUNDS["bm_mass"][0]))
    
    # Normalize start arc length
    normalized.append((solution["start_arc_length"] - PARAM_BOUNDS["start_arc_length"][0]) / 
                    (PARAM_BOUNDS["start_arc_length"][1] - PARAM_BOUNDS["start_arc_length"][0]))
    
    # Normalize number of modules
    normalized.append((solution["num_modules"] - PARAM_BOUNDS["num_modules"][0]) / 
                    (PARAM_BOUNDS["num_modules"][1] - PARAM_BOUNDS["num_modules"][0]))
    
    return normalized

def denormalize_solution(normalized_solution):
    """
    Convert normalized solution back to actual parameter values.
    
    Args:
        normalized_solution: List of normalized values between 0 and 1
        
    Returns:
        tuple: (bm_mass, start_arc_length, num_modules) with proper numeric types
    """
    try:
        # Get parameter bounds
        bm_mass_bounds = PARAM_BOUNDS["bm_mass"]
        start_arc_length_bounds = PARAM_BOUNDS["start_arc_length"]
        num_modules_bounds = PARAM_BOUNDS["num_modules"]
        
        # Clamp normalized values to [0, 1]
        norm = [min(1, max(0, float(x))) for x in normalized_solution]
        # Denormalize each parameter - ensure bm_mass and start_arc_length are rounded integers
        bm_mass = int(round(norm[0] * (bm_mass_bounds[1] - bm_mass_bounds[0]) + bm_mass_bounds[0]))
        start_arc_length = int(round(norm[1] * (start_arc_length_bounds[1] - start_arc_length_bounds[0]) + start_arc_length_bounds[0]))
        num_modules = int(round(norm[2] * (num_modules_bounds[1] - num_modules_bounds[0]) + num_modules_bounds[0]))
        
        return bm_mass, start_arc_length, num_modules
        
    except (IndexError, TypeError, ValueError) as e:
        logger.error(f"Error denormalizing solution: {e}")
        # Return mid-point values as fallback
        bm_mass_bounds = PARAM_BOUNDS["bm_mass"]
        start_arc_length_bounds = PARAM_BOUNDS["start_arc_length"]
        num_modules_bounds = PARAM_BOUNDS["num_modules"]
        
        return (
            int(round((bm_mass_bounds[1] + bm_mass_bounds[0]) / 2)),
            int(round((start_arc_length_bounds[1] + start_arc_length_bounds[0]) / 2)),
            int(round((num_modules_bounds[1] + num_modules_bounds[0]) / 2))
        )

def save_best_solution(solution):
    """
    Save the best solution to a file for later analysis.
    
    Args:
        solution (dict): The best solution parameters
    """
    try:
        # Load model
        model = load_model(MODEL_PATH)
        
        # Extract parameters - ensure bm_mass and start_arc_length are integers
        bm_mass = int(round(solution["bm_mass"]))
        start_arc_length = int(round(solution["start_arc_length"]))
        num_modules = solution["num_modules"]
        
        # Calculate EoL mass (rounded to 2 decimal places)
        eol_bm_mass = round(bm_mass * BM_EoL_MASS_FACTOR, 2)
        
        # Separate SoL and EoL cables
        sol_cables = [cable for cable in CABLE_NAMES if "SoL" in cable]
        eol_cables = [cable for cable in CABLE_NAMES if "EoL" in cable]
        
        # Update buoyancy module masses
        update_bm_mass(model, BM_NAMES["SoL"], bm_mass)
        update_bm_mass(model, BM_NAMES["EoL"], eol_bm_mass)
        
        # Configure buoyancy modules on all cables
        for cable in sol_cables:
            configure_buoyancy_modules(
                model, cable, BM_NAMES["SoL"], 
                start_arc_length, num_modules, BM_SPACING
            )
        
        for cable in eol_cables:
            configure_buoyancy_modules(
                model, cable, BM_NAMES["EoL"], 
                start_arc_length, num_modules, BM_SPACING
            )
        
        # Run static analysis
        start_time = time.time()
        success = run_static_analysis(model)
        
        if success:
            elapsed_time = time.time() - start_time
            logger.info(f"Best solution static analysis completed in {elapsed_time:.2f} seconds")
            # Save the model
            model.SaveData("BestSolution.dat")
            logger.info(f"Best solution saved to BestSolution.dat")
        else:
            elapsed_time = time.time() - start_time
            logger.error(f"Best solution static analysis failed after {elapsed_time:.2f} seconds - could not save model")
    except Exception as e:
        logger.error(f"Error saving best solution: {e}") 