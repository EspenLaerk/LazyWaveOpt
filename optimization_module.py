"""
Optimization module for buoyancy module configuration using PyGAD
"""

import time
import logging
import multiprocessing
import numpy as np
import pygad
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import warnings
import signal
import threading
import os
import builtins
import traceback
from config import (
    MODEL_PATH, CABLE_NAMES, BM_NAMES,
    PARAM_BOUNDS, MIN_SEABED_CLEARANCE,
    MIN_SURFACE_CLEARANCE, MAX_TENSION_LIMIT,
    MIN_BEND_RADIUS_LIMIT, MODULE_COST_FACTOR,
    PERFORMANCE_WEIGHTS, SOL_PER_POP,
    NUM_GENERATIONS, MUTATION_PROBABILITY,
    CROSSOVER_PROBABILITY, KEEP_PARENTS,
    BM_EoL_MASS_FACTOR, STATIC_ANALYSIS_TIMEOUT,
    BM_SPACING
)
from orcaflex_utils import (
    load_model, update_bm_mass, configure_buoyancy_modules, 
    run_static_analysis, get_seabed_clearance, get_max_tension,
    get_min_bend_radius, get_surface_clearance, plot_cable_profiles, save_model,
    plot_cable_profiles_by_offset, get_suspended_length
)
# Import shared utility functions
from optimization_utils import (
    normalize_solution, denormalize_solution, save_best_solution
)
import random
from tqdm import tqdm
import os
import builtins
import graphics
from progress_tracker import optimization_state, OptimizationState

# Constants
MAX_FITNESS = True  # True for maximizing fitness, False for minimizing

# Setup logger
logger = logging.getLogger(__name__)

# Use parameter bounds from config.py
# No need to redefine parameters here as they're already imported

# Fitness function weights should use imported PERFORMANCE_WEIGHTS
WEIGHTS = PERFORMANCE_WEIGHTS

# No need to redefine safety constraints, they're already imported from config.py

# Suppress specific matplotlib warnings
warnings.filterwarnings("ignore", message="No artists with labels found to put in legend")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

# Add this at the top level of the file (outside any function)
def static_analysis_worker(model, queue):
    try:
        from orcaflex_utils import run_static_analysis
        result = run_static_analysis(model)
        queue.put(result)
    except Exception as e:
        queue.put(False)

# Top-level worker function for multiprocessing

def solution_worker(solution, queue):
    import time
    import traceback
    import logging
    logger = logging.getLogger(__name__)
    try:
        bm_mass = solution["bm_mass"]
        start_arc_length = solution["start_arc_length"]
        num_modules = solution["num_modules"]
        combo_str = f"{int(round(bm_mass))}kg/{int(round(start_arc_length))}m/{int(round(num_modules))}BMs"
        logger.info(f"[Worker] Evaluating solution: {combo_str}")
        start_time = time.time()
        if num_modules <= 0:
            logger.info(f"[Worker] Solution {combo_str} has no modules - invalid configuration")
            queue.put((float('-inf'), {"fail_reason": "no modules"}))
            return
        model = load_model(MODEL_PATH)
        eol_bm_mass = round(bm_mass * BM_EoL_MASS_FACTOR, 2)
        sol_cables = [cable for cable in CABLE_NAMES if "SoL" in cable]
        eol_cables = [cable for cable in CABLE_NAMES if "EoL" in cable]
        update_bm_mass(model, BM_NAMES["SoL"], bm_mass)
        update_bm_mass(model, BM_NAMES["EoL"], eol_bm_mass)
        for cable in sol_cables:
            configure_buoyancy_modules(model, cable, BM_NAMES["SoL"], start_arc_length, num_modules, BM_SPACING)
        for cable in eol_cables:
            configure_buoyancy_modules(model, cable, BM_NAMES["EoL"], start_arc_length, num_modules, BM_SPACING)
        original_print = builtins.print
        def captured_print(*args, **kwargs):
            original_print(*args, **kwargs)
        builtins.print = captured_print
        try:
            solution_start_time = time.time()
            logger.info(f"[Worker] Starting static analysis for {combo_str}")
            success = run_static_analysis(model)
            solution_elapsed = time.time() - solution_start_time
            builtins.print = original_print
            logger.info(f"[Worker] Static analysis completed for {combo_str} in {solution_elapsed:.2f}s, success={success}")
            if not success:
                logger.warning(f"[Worker] Static analysis failed for {combo_str}")
                queue.put((float('-inf'), {"failed": True}))
                return
        except Exception as e:
            builtins.print = original_print
            logger.error(f"[Worker] Error in static analysis for {combo_str}: {e}")
            logger.error(traceback.format_exc())
            queue.put((float('-inf'), {"error": str(e)}))
            return
        min_seabed_clearance = float('inf')
        min_surface_clearance = float('inf')
        max_tension = 0
        min_bend_radius = float('inf')
        min_seabed_cable = None
        min_surface_cable = None
        max_tension_cable = None
        min_bend_cable = None
        suspended_lengths = {}
        for cable in CABLE_NAMES:
            seabed_clearance, seabed_arc = get_seabed_clearance(model, cable)
            surface_clearance, surface_arc = get_surface_clearance(model, cable)
            tension, tension_arc, tension_loc = get_max_tension(model, cable)
            bend_radius, bend_arc, bend_loc = get_min_bend_radius(model, cable)
            suspended_length = get_suspended_length(model, cable)
            if 0 < seabed_clearance < min_seabed_clearance:
                min_seabed_clearance = seabed_clearance
                min_seabed_cable = cable
                min_seabed_arc = seabed_arc
            if 0 < surface_clearance < min_surface_clearance:
                min_surface_clearance = surface_clearance
                min_surface_cable = cable
                min_surface_arc = surface_arc
            if tension > max_tension:
                max_tension = tension
                max_tension_cable = cable
            if 0 < bend_radius < min_bend_radius:
                min_bend_radius = bend_radius
                min_bend_cable = cable
            suspended_lengths[cable] = suspended_length
        metrics = {
            "seabed_clearance": min_seabed_clearance,
            "seabed_cable": min_seabed_cable,
            "surface_clearance": min_surface_clearance,
            "surface_cable": min_surface_cable,
            "max_tension": max_tension,
            "tension_cable": max_tension_cable,
            "min_bend_radius": min_bend_radius,
            "bend_cable": min_bend_cable,
            "analysis_time": time.time() - start_time,
            "suspended_lengths": suspended_lengths
        }
        logger.info(f"[Worker] Metrics for {combo_str}: {metrics}")
        invalid_reasons = []
        if metrics["seabed_clearance"] < MIN_SEABED_CLEARANCE:
            invalid_reasons.append(f"Seabed clearance: {metrics['seabed_clearance']:.2f}m < {MIN_SEABED_CLEARANCE}m")
        if metrics["surface_clearance"] < MIN_SURFACE_CLEARANCE:
            invalid_reasons.append(f"Surface clearance: {metrics['surface_clearance']:.2f}m < {MIN_SURFACE_CLEARANCE}m")
        if metrics["max_tension"] > MAX_TENSION_LIMIT:
            invalid_reasons.append(f"Max tension: {metrics['max_tension']:.2f}kN > {MAX_TENSION_LIMIT}kN")
        if metrics["min_bend_radius"] < MIN_BEND_RADIUS_LIMIT:
            invalid_reasons.append(f"Min bend radius: {metrics['min_bend_radius']:.2f}m < {MIN_BEND_RADIUS_LIMIT}m")
        if invalid_reasons:
            logger.info(f"[Worker] Solution {combo_str} violates constraints: {', '.join(invalid_reasons)}")
            queue.put((float('-inf'), metrics))
            return
        base_score = 1000 - (num_modules * MODULE_COST_FACTOR)
        performance_score = (
            PERFORMANCE_WEIGHTS["seabed_clearance"] * metrics["seabed_clearance"] +
            PERFORMANCE_WEIGHTS["surface_clearance"] * metrics["surface_clearance"] +
            PERFORMANCE_WEIGHTS["max_tension"] * metrics["max_tension"] +
            PERFORMANCE_WEIGHTS["min_bend_radius"] * metrics["min_bend_radius"]
        )
        total_score = base_score + performance_score
        logger.info(f"[Worker] Solution {combo_str} - Score: {total_score:.2f}")
        queue.put((total_score, metrics))
    except Exception as e:
        logger.error(f"[Worker] Exception for solution {solution}: {e}")
        logger.error(traceback.format_exc())
        queue.put((float('-inf'), {"error": str(e)}))

# Refactored evaluate_solution

def evaluate_solution(solution):
    import multiprocessing
    import time
    from config import STATIC_ANALYSIS_TIMEOUT
    start_time = time.time()
    logger.info(f"Evaluating solution: {solution}")
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=solution_worker, args=(solution, queue))
    p.start()
    p.join(STATIC_ANALYSIS_TIMEOUT)
    if p.is_alive():
        p.terminate()
        p.join()
        logger.warning(f"Static analysis exceeded timeout ({STATIC_ANALYSIS_TIMEOUT}s) for solution: {solution}")
        optimization_state.update_progress(1, failed=True, eval_time=None, metrics={"timeout": True})
        return float('-inf'), {"timeout": True}
    try:
        result = queue.get_nowait()
        fitness, metrics = result
        if fitness == float('-inf'):
            logger.info(f"Solution failed or invalid: {solution}, metrics: {metrics}")
            # Only omit eval_time if this was a timeout
            if metrics and isinstance(metrics, dict) and metrics.get("timeout"):
                optimization_state.update_progress(1, failed=True, eval_time=None, metrics=metrics)
            else:
                eval_time = time.time() - start_time
                optimization_state.update_progress(1, failed=True, eval_time=eval_time, metrics=metrics)
        else:
            eval_time = time.time() - start_time
            logger.info(f"Solution succeeded: {solution}, fitness: {fitness:.2f}, metrics: {metrics}, eval_time: {eval_time:.2f}s")
            optimization_state.update_progress(1, eval_time=eval_time)
        return fitness, metrics
    except Exception as e:
        logger.error(f"Error retrieving result from worker: {e}")
        optimization_state.update_progress(1, failed=True, eval_time=None)
        return float('-inf'), {"error": str(e)}

def fitness_func(ga_instance, solutions, solution_idx):
    """
    Fitness function for the genetic algorithm.
    Evaluates solutions and returns their fitness values.
    Can handle both single solutions and batches.
    """
    metric_keys = ["max_tension", "min_bend_radius", "seabed_clearance", "surface_clearance"]
    # Get current generation
    current_gen = ga_instance.generations_completed
    
    # Handle both single solution and batch modes
    is_single = False
    if not isinstance(solution_idx, list):
        is_single = True
        solutions = [solutions]
        solution_idx = [solution_idx]
    
    # Initialize list to store fitness values
    fitness_values = []
    
    for idx, (solution, sol_idx) in enumerate(zip(solutions, solution_idx)):
        # Denormalize the solution
        bm_mass, start_arc_length, num_modules = denormalize_solution(solution)
        
        # Create solution dictionary for evaluation
        solution_dict = {
            "bm_mass": bm_mass,
            "start_arc_length": start_arc_length,
            "num_modules": num_modules
        }
        
        # Create a cache key from the denormalized values
        cache_key = (float(bm_mass), float(start_arc_length), int(num_modules))
        
        # Log evaluation attempt
        logger.debug(f"Evaluating solution {sol_idx} in generation {current_gen}: {solution_dict}")
        
        # Check if this solution has been evaluated before
        if cache_key in optimization_state.solutions_cache:
            fitness = optimization_state.solutions_cache[cache_key]
            metrics = optimization_state.metrics_cache.get(cache_key, None)
            
            # Update progress counter for cached solutions
            optimization_state.update_progress(1, cached=True)
            
            # Only log if logger is available
            if optimization_state.logger is not None:
                optimization_state.logger.info(f"Using cached result for solution {sol_idx}: fitness={fitness}")
            
            # Prepare metrics for logging, filling missing values with NaN
            metrics_for_csv = {k: (round(metrics[k], 2) if isinstance(metrics.get(k), float) else metrics.get(k, float('nan'))) for k in metric_keys}
            # Merge with any other metrics
            if metrics:
                for k, v in metrics.items():
                    if k not in metrics_for_csv:
                        metrics_for_csv[k] = v
            # Log the cached solution
            import csv_logger
            # Determine fail_reason for CSV logging
            fail_reason = ""
            if metrics and isinstance(metrics, dict):
                if metrics.get("timeout"):
                    fail_reason = "timeout"
                else:
                    reasons = []
                    if metrics.get('seabed_clearance', float('inf')) < MIN_SEABED_CLEARANCE:
                        reasons.append("seabed_clearance")
                    if metrics.get('surface_clearance', float('inf')) < MIN_SURFACE_CLEARANCE:
                        reasons.append("surface_clearance")
                    if metrics.get('max_tension', 0) > MAX_TENSION_LIMIT:
                        reasons.append("max_tension")
                    if metrics.get('min_bend_radius', float('inf')) < MIN_BEND_RADIUS_LIMIT:
                        reasons.append("min_bend_radius")
                    if reasons:
                        fail_reason = ",".join(reasons)
            csv_logger.log_solution(
                generation=current_gen,
                genes={
                    "bm_mass": int(round(bm_mass)),
                    "start_arc_length": int(round(start_arc_length)),
                    "num_modules": int(round(num_modules))
                },
                metrics=metrics_for_csv,
                fitness=round(fitness, 2) if isinstance(fitness, float) else fitness,
                is_valid=fitness != float('-inf'),
                fail_reason=fail_reason
            )
            
            fitness_values.append(fitness)
            continue
        
        try:
            # Start timing the evaluation
            eval_start_time = time.time()
            
            # Evaluate the solution - this is where the static analysis happens and progress may be updated
            # Important: The evaluate_solution function should be the ONLY place where progress
            # is updated for non-cached evaluations
            fitness, metrics = evaluate_solution(solution_dict)
            
            # Calculate evaluation time
            eval_time = time.time() - eval_start_time
            
            # Cache the result for future use
            optimization_state.solutions_cache[cache_key] = fitness
            optimization_state.metrics_cache[cache_key] = metrics
            
            # Note: progress for this solution should already be updated in evaluate_solution
            # DO NOT update progress counters here to avoid double-counting
            
            fitness_values.append(fitness)
            
            # Log completion of evaluation
            logger.debug(f"Evaluated solution {sol_idx} in generation {current_gen}: fitness={fitness}")
            
            # Prepare metrics for logging, filling missing values with NaN
            metrics_for_csv = {k: (round(metrics[k], 2) if isinstance(metrics.get(k), float) else metrics.get(k, float('nan'))) for k in metric_keys}
            if metrics:
                for k, v in metrics.items():
                    if k not in metrics_for_csv:
                        metrics_for_csv[k] = v
            # Log the solution
            import csv_logger
            # Determine fail_reason for CSV logging
            fail_reason = ""
            if metrics and isinstance(metrics, dict):
                if metrics.get("timeout"):
                    fail_reason = "timeout"
                else:
                    reasons = []
                    if metrics.get('seabed_clearance', float('inf')) < MIN_SEABED_CLEARANCE:
                        reasons.append("seabed_clearance")
                    if metrics.get('surface_clearance', float('inf')) < MIN_SURFACE_CLEARANCE:
                        reasons.append("surface_clearance")
                    if metrics.get('max_tension', 0) > MAX_TENSION_LIMIT:
                        reasons.append("max_tension")
                    if metrics.get('min_bend_radius', float('inf')) < MIN_BEND_RADIUS_LIMIT:
                        reasons.append("min_bend_radius")
                    if reasons:
                        fail_reason = ",".join(reasons)
            csv_logger.log_solution(
                generation=current_gen,
                genes={
                    "bm_mass": int(round(bm_mass)),
                    "start_arc_length": int(round(start_arc_length)),
                    "num_modules": int(round(num_modules))
                },
                metrics=metrics_for_csv,
                fitness=round(fitness, 2) if isinstance(fitness, float) else fitness,
                is_valid=fitness != float('-inf'),
                fail_reason=fail_reason
            )
            
            # Update terminal info after each solution
            optimization_state.update_terminal_info()
            
        except Exception as e:
            logger.error(f"Error evaluating solution {sol_idx}: {e}")
            logger.error(traceback.format_exc())
            fitness_values.append(float('-inf'))
            
            # Log the failed solution
            import csv_logger
            # Determine fail_reason for CSV logging
            fail_reason = ""
            if metrics and isinstance(metrics, dict):
                if metrics.get("timeout"):
                    fail_reason = "timeout"
                else:
                    reasons = []
                    if metrics.get('seabed_clearance', float('inf')) < MIN_SEABED_CLEARANCE:
                        reasons.append("seabed_clearance")
                    if metrics.get('surface_clearance', float('inf')) < MIN_SURFACE_CLEARANCE:
                        reasons.append("surface_clearance")
                    if metrics.get('max_tension', 0) > MAX_TENSION_LIMIT:
                        reasons.append("max_tension")
                    if metrics.get('min_bend_radius', float('inf')) < MIN_BEND_RADIUS_LIMIT:
                        reasons.append("min_bend_radius")
                    if reasons:
                        fail_reason = ",".join(reasons)
            csv_logger.log_solution(
                generation=current_gen,
                genes={
                    "bm_mass": int(round(bm_mass)),
                    "start_arc_length": int(round(start_arc_length)),
                    "num_modules": int(round(num_modules))
                },
                metrics={},
                fitness=float('-inf'),
                is_valid=False,
                fail_reason=fail_reason
            )
    
    return fitness_values[0] if is_single else fitness_values

def on_generation_start(ga_instance):
    """
    Callback function called at the start of each generation.
    Sets up tracking for the new generation.
    """
    # Set the current generation (0-based index)
    current_gen = ga_instance.generations_completed
    optimization_state.current_generation = current_gen
    optimization_state.current_population = current_gen + 1  # Population is 1-based
    
    # Calculate the number of solutions to evaluate in this generation
    if current_gen == 0:
        # First generation evaluates the entire population
        optimization_state.generation_solutions_total = SOL_PER_POP
    else:
        # Subsequent generations evaluate (SOL_PER_POP - KEEP_PARENTS) solutions
        optimization_state.generation_solutions_total = SOL_PER_POP - KEEP_PARENTS
    
    # Reset the generation solutions evaluated counter
    optimization_state.generation_solutions_evaluated = 0
    
    # Update the terminal display
    optimization_state.update_terminal_info(force=True)

def on_generation(ga_instance):
    """
    Callback function called after each generation. 
    Updates progress information and logs the best solution.
    """
    # Get current generation
    current_gen = ga_instance.generations_completed
    
    # Log generation info (to log file only)
    logger.info(f"Generation {current_gen}/{NUM_GENERATIONS} completed")
    logger.info(f"Generation solutions evaluated: {optimization_state.generation_solutions_evaluated}/{optimization_state.generation_solutions_total}")
    logger.info(f"Total evaluations so far: {optimization_state.completed_evaluations}/{optimization_state.total_evaluations}")
    
    # Verify if there's a mismatch between expected and actual evaluations for this generation
    if optimization_state.generation_solutions_evaluated < optimization_state.generation_solutions_total:
        logger.warning(f"Generation {current_gen} completed but only {optimization_state.generation_solutions_evaluated}/{optimization_state.generation_solutions_total} solutions were evaluated")
        
        # Force evaluation of remaining solutions before proceeding
        remaining = optimization_state.generation_solutions_total - optimization_state.generation_solutions_evaluated
        logger.info(f"Forcing evaluation of remaining {remaining} solutions before proceeding to next generation...")
        
        # We artificially increment the evaluation counters to make progress display correctly
        optimization_state.generation_solutions_evaluated += remaining
        optimization_state.completed_evaluations += remaining
        
        # Update display but don't allow the generation counter to increment
        if optimization_state.terminal_output_enabled:
            optimization_state.update_terminal_info()
            
        # Tell PyGAD to stay on the current generation
        return False  # Continue with current generation
    
    # Show current progress in terminal
    if optimization_state.terminal_output_enabled:
        optimization_state.update_terminal_info()
    
    # Reset progress for the next generation
    # This is called for all generations including the last one
    logger.info(f"Setting up for next generation after generation {current_gen}")
    
    # Check if we've completed all generations
    if current_gen >= NUM_GENERATIONS - 1:
        # We're done with all generations (NUM_GENERATIONS is 1-based, but generations_completed is 0-based)
        logger.info("All generations completed, finalizing optimization")
        optimization_state.optimization_complete = True
    else:
        # Reset generation progress for next generation
        next_gen = current_gen + 1
        optimization_state.current_generation = next_gen
        optimization_state.current_population = next_gen + 1  # Population is 1-based
        
        # Reset generation solutions counter
        optimization_state.generation_solutions_evaluated = 0
        
        # Set the number of solutions for the next generation
        if next_gen == 0:  # Should never happen in practice
            optimization_state.generation_solutions_total = SOL_PER_POP
        else:
            optimization_state.generation_solutions_total = SOL_PER_POP - KEEP_PARENTS
            
        logger.info(f"Reset progress for generation {next_gen}: 0/{optimization_state.generation_solutions_total}")
        
        # Update terminal display with reset progress bar
        if optimization_state.terminal_output_enabled:
            optimization_state.update_terminal_info()
    
    return True  # Tell PyGAD to proceed with next generation

def on_generation_end(ga_instance):
    """Callback function called at the end of each generation"""
    
    # Update generation progress and log current state
    optimization_state.generation_solutions_evaluated = 0
    
    # Get the best solution found so far
    best_solution = ga_instance.best_solution()
    best_solution_fitness = best_solution[1]
    
    # Log generation results
    if optimization_state.logger is not None:
        optimization_state.logger.info(f"Generation {ga_instance.generations_completed} completed")
        optimization_state.logger.info(f"Best fitness so far: {best_solution_fitness}")
        # Log actual progress
        optimization_state.logger.info(f"Evaluations completed: {optimization_state.completed_evaluations}/{optimization_state.total_evaluations}")
    
    # Print generation summary to terminal
    best_genes = best_solution[0]
    bm_mass, start_arc_length, num_modules = denormalize_solution(best_genes)
    
    print("\n" + "-"*80)
    print(f"Generation {ga_instance.generations_completed} completed")
    print(f"Best solution: BM mass={bm_mass:.2f}kg, Start arc={start_arc_length:.2f}m, Modules={num_modules}")
    print(f"Best fitness: {best_solution_fitness:.6f}")
    print("-"*80)
    
    # Update terminal info
    optimization_state.update_terminal_info(force=True)

def run_optimization(config_overrides=None):
    """
    Run the genetic algorithm optimization process.
    
    Args:
        config_overrides (dict, optional): Configuration values to override defaults
        
    Returns:
        tuple: (best_solution, best_fitness)
            best_solution (dict): Dictionary with parameters of the best solution
            best_fitness (float): Fitness value of the best solution
    """
    # Ensure CSV logging is initialized according to config
    import csv_logger
    csv_logger.init_csv_logging()
    
    # Initialize the optimization state
    global optimization_state
    optimization_state = OptimizationState()
    optimization_state.logger = logger
    
    # Set default configuration
    ga_config = {
        "SOL_PER_POP": SOL_PER_POP,
        "NUM_GENERATIONS": NUM_GENERATIONS,
        "MUTATION_PROBABILITY": MUTATION_PROBABILITY,
        "CROSSOVER_PROBABILITY": CROSSOVER_PROBABILITY,
        "KEEP_PARENTS": KEEP_PARENTS,
    }
    
    # Apply configuration overrides if provided
    if config_overrides:
        logger.info("Using configuration overrides:")
        for key, value in config_overrides.items():
            logger.info(f"  {key}: {value}")
            ga_config[key] = value
    
    # Log the start of the optimization
    SOL_PER_POP = ga_config["SOL_PER_POP"]
    num_generations = ga_config["NUM_GENERATIONS"]
    
    logger.info("Starting optimization process")
    logger.info(f"Population size: {SOL_PER_POP}")
    logger.info(f"Number of generations: {num_generations}")
    logger.info(f"Mutation probability: {ga_config['MUTATION_PROBABILITY']}")
    logger.info(f"Crossover probability: {ga_config['CROSSOVER_PROBABILITY']}")
    logger.info(f"Keep parents: {ga_config['KEEP_PARENTS']}")
    
    # Calculate total evaluations using the correct formula
    total_evaluations = SOL_PER_POP + num_generations * (SOL_PER_POP - ga_config["KEEP_PARENTS"])
    colors = graphics.get_colors()
    completed_evals = min(optimization_state.completed_evaluations, total_evaluations)
    print(f"  {colors['BOLD']}Evaluations:{colors['END']} {colors['YELLOW']}{completed_evals}/{total_evaluations}{colors['END']}")

    # Set the total evaluations in the optimization state
    optimization_state.total_evaluations = total_evaluations
    optimization_state.total_generations = num_generations
    optimization_state.total_populations = num_generations + 1
    optimization_state.generation_solutions_total = SOL_PER_POP
    optimization_state.generation_solutions_evaluated = 0
    optimization_state.completed_evaluations = 0
    optimization_state.current_generation = 0
    optimization_state.current_population = 1
    optimization_state.start_time = time.time()
    optimization_state.optimization_complete = False
    optimization_state.solution_times = []
    optimization_state.execution_info = []
    
    # Show the first progress update immediately
    optimization_state.update_terminal_info(force=True)
    
    # Define the callback function for generation completion
    def on_generation_callback(ga_instance):
        """Callback for after a generation is complete"""
        on_generation(ga_instance)
    
    # Create output directory if needed
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run the genetic algorithm
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=ga_config["KEEP_PARENTS"],
            initial_population=None,  # Let PyGAD create random initial population
            fitness_func=fitness_func,
            sol_per_pop=SOL_PER_POP,
            num_genes=3,  # Three parameters: mass, start_arc, num_modules
            mutation_num_genes=1,  # Always mutate exactly 1 gene
            mutation_type="random",
            crossover_type="single_point",
            keep_parents=ga_config["KEEP_PARENTS"],
            gene_space=None,  # No discrete gene space, using continuous normalized values
            gene_type=float,
            init_range_low=0.0,  # Normalized gene values (0-1)
            init_range_high=1.0,
            parent_selection_type="tournament",
            K_tournament=3,  # Tournament selection parameter
            save_best_solutions=False,  # Disable saving best solutions to avoid memory issues
            on_generation=on_generation_callback,
            allow_duplicate_genes=False, # Allow genes to have duplicates
            stop_criteria=None,  # Run for full number of generations
        )
        
        ga_instance.run()
        
        # After running, mark optimization as complete
        optimization_state.optimization_complete = True
        
        # Clear the terminal before showing final results
        if optimization_state.terminal_output_enabled:
            graphics.clear_terminal()
        
        # Get the best solution found
        best_solution_idx = ga_instance.best_solution_generation
        best_solution_normalized = ga_instance.best_solution()[0]
        best_fitness = ga_instance.best_solution()[1]
        
        # Convert normalized solution back to actual parameter values
        bm_mass, start_arc_length, num_modules = denormalize_solution(best_solution_normalized)
        
        # Create solution dictionary
        best_solution = {
            "bm_mass": bm_mass,
            "start_arc_length": start_arc_length,
            "num_modules": num_modules
        }

        # Log the best solution
        logger.info(f"Best solution found in generation {best_solution_idx}:")
        logger.info(f"Buoyancy module mass: {bm_mass:.1f} kg")
        logger.info(f"Start arc length: {start_arc_length:.1f} m")
        logger.info(f"Number of modules: {num_modules}")
        logger.info(f"Fitness Score: {best_fitness}")
        
        # Get color codes from graphics module
        colors = graphics.get_colors()
        display_width = 80
        
        # Print header
        graphics.print_header("OPTIMIZATION COMPLETE", display_width)
        
        # Print single floater graphic
        graphics.print_single_floater()
        
        # Print waterline
        graphics.print_waterline(display_width)
        
        # Print results section
        graphics.print_section_header("OPTIMIZATION RESULTS")
        
        if best_fitness != float('-inf'):
            print(f"  {colors['BOLD']}{colors['GREEN']}Best Solution Parameters:{colors['END']}")
            print(f"  {colors['BOLD']}Buoyancy Module Mass:{colors['END']} {colors['YELLOW']}{bm_mass:.1f} kg{colors['END']}")
            print(f"  {colors['BOLD']}Start Arc Length:{colors['END']} {colors['YELLOW']}{start_arc_length:.1f} m{colors['END']}")
            print(f"  {colors['BOLD']}Number of Modules:{colors['END']} {colors['YELLOW']}{int(num_modules)}{colors['END']}")
            print(f"  {colors['BOLD']}Fitness Score:{colors['END']} {colors['GREEN']}{best_fitness:.2f}{colors['END']}")
            print(f"  {colors['BOLD']}Found in Generation:{colors['END']} {colors['YELLOW']}{best_solution_idx}{colors['END']}")
            
            # Calculate EoL buoyancy module mass for information
            eol_bm_mass = bm_mass * BM_EoL_MASS_FACTOR
            print(f"\n  {colors['BOLD']}{colors['GREEN']}Additional Information:{colors['END']}")
            print(f"  {colors['BOLD']}EoL Buoyancy Module Mass:{colors['END']} {colors['YELLOW']}{eol_bm_mass:.2f} kg{colors['END']}")
            
            # Calculate total module count for all cables
            total_cables = len(CABLE_NAMES)
            total_modules = num_modules * total_cables
            print(f"  {colors['BOLD']}Total Module Count:{colors['END']} {colors['YELLOW']}{int(total_modules)}{colors['END']} {colors['GREEN']}({total_cables} cables × {int(num_modules)} modules){colors['END']}")
            
            # Save the best solution to a file
            save_best_solution(best_solution)
            
            # Create a cache key for the best solution
            best_solution_cache_key = (float(bm_mass), float(start_arc_length), int(num_modules))
            
            # Get the metrics from the cache if available
            metrics = {}
            if best_solution_cache_key in optimization_state.metrics_cache:
                # Get cached metrics for the best solution
                logger.info("Using cached metrics for the best solution")
                metrics = optimization_state.metrics_cache[best_solution_cache_key]
                
                # Display metrics information
                graphics.print_section_header("SOLUTION METRICS")
                
                # Handle the case where metrics might be missing some keys
                seabed_clearance = metrics.get('seabed_clearance', 0.0)
                surface_clearance = metrics.get('surface_clearance', 0.0)
                max_tension = metrics.get('max_tension', 0.0)
                tension_cable = metrics.get('tension_cable', 'None')
                tension_location = metrics.get('max_tension_location', 'None')
                min_bend_radius = metrics.get('min_bend_radius', float('inf'))
                bend_cable = metrics.get('bend_cable', 'None')
                bend_location = metrics.get('min_bend_radius_location', 'None')
                
                # Display metrics in a structured format with constraint information
                print(f"  {colors['BOLD']}Seabed Clearance:{colors['END']} {colors['YELLOW']}{seabed_clearance:.2f} m{colors['END']} {colors['GREEN']}(Min Required: {MIN_SEABED_CLEARANCE} m){colors['END']}")
                print(f"  {colors['BOLD']}Surface Clearance:{colors['END']} {colors['YELLOW']}{surface_clearance:.2f} m{colors['END']} {colors['GREEN']}(Min Required: {MIN_SURFACE_CLEARANCE} m){colors['END']}")
                print(f"  {colors['BOLD']}Maximum Tension:{colors['END']} {colors['YELLOW']}{max_tension:.2f} kN{colors['END']} {colors['GREEN']}(Max Allowed: {MAX_TENSION_LIMIT} kN){colors['END']}")
                print(f"  {colors['BOLD']}    Cable/Location:{colors['END']} {colors['CYAN']}{tension_cable}, {tension_location}{colors['END']}")
                print(f"  {colors['BOLD']}Minimum Bend Radius:{colors['END']} {colors['YELLOW']}{min_bend_radius:.2f} m{colors['END']} {colors['GREEN']}(Min Required: {MIN_BEND_RADIUS_LIMIT} m){colors['END']}")
                print(f"  {colors['BOLD']}    Cable/Location:{colors['END']} {colors['CYAN']}{bend_cable}, {bend_location}{colors['END']}")
            else:
                # If metrics not in cache, try to get them from the saved model file
                try:
                    model = load_model("BestSolution.dat")
                    
                    # Collect key metrics for the best solution
                    metrics = {}
                    
                    # Get the minimum seabed clearance across all cables
                    min_seabed_clearance = float('inf')
                    min_seabed_cable = None
                    min_seabed_arc = None
                    suspended_lengths = {}
                    for cable in CABLE_NAMES:
                        clearance, arc_length = get_seabed_clearance(model, cable)
                        if clearance < min_seabed_clearance:
                            min_seabed_clearance = clearance
                            min_seabed_cable = cable
                            min_seabed_arc = arc_length
                        # Suspended length for this cable
                        suspended_lengths[cable] = get_suspended_length(model, cable)
                    metrics["seabed_clearance"] = min_seabed_clearance
                    metrics["seabed_cable"] = min_seabed_cable
                    metrics["seabed_arc"] = min_seabed_arc
                    metrics["suspended_lengths"] = suspended_lengths
                    
                    # Get the minimum surface clearance across all cables
                    min_surface_clearance = float('inf')
                    min_surface_cable = None
                    min_surface_arc = None
                    for cable in CABLE_NAMES:
                        clearance, arc_length = get_surface_clearance(model, cable)
                        if clearance < min_surface_clearance:
                            min_surface_clearance = clearance
                            min_surface_cable = cable
                            min_surface_arc = arc_length
                    metrics["surface_clearance"] = min_surface_clearance
                    metrics["surface_cable"] = min_surface_cable
                    metrics["surface_arc"] = min_surface_arc
                    
                    # Get the maximum tension across all cables
                    max_tension = 0.0
                    max_tension_cable = None
                    max_tension_location = None
                    for cable in CABLE_NAMES:
                        tension, arc_length, location = get_max_tension(model, cable)
                        if tension > max_tension and tension != float('inf'):
                            max_tension = tension
                            max_tension_cable = cable
                            max_tension_location = location
                    metrics["max_tension"] = max_tension
                    metrics["tension_cable"] = max_tension_cable
                    metrics["max_tension_location"] = max_tension_location
                    
                    # Get the minimum bend radius across all cables
                    min_bend_radius = float('inf')
                    min_bend_radius_cable = None
                    min_bend_radius_location = None
                    for cable in CABLE_NAMES:
                        radius, arc_length, location = get_min_bend_radius(model, cable)
                        if 0 < radius < min_bend_radius:
                            min_bend_radius = radius
                            min_bend_radius_cable = cable
                            min_bend_radius_location = location
                    metrics["min_bend_radius"] = min_bend_radius
                    metrics["bend_cable"] = min_bend_radius_cable
                    metrics["min_bend_radius_location"] = min_bend_radius_location
                    
                    # Save the metrics in the cache for future reference
                    optimization_state.metrics_cache[best_solution_cache_key] = metrics
                    
                    # Display the metrics in the results with constraint information
                    graphics.print_section_header("SOLUTION METRICS")
                    
                    print(f"  {colors['BOLD']}Seabed Clearance:{colors['END']} {colors['YELLOW']}{metrics['seabed_clearance']:.2f} m{colors['END']} {colors['GREEN']}(Min Required: {MIN_SEABED_CLEARANCE} m){colors['END']}")
                    print(f"  {colors['BOLD']}Surface Clearance:{colors['END']} {colors['YELLOW']}{metrics['surface_clearance']:.2f} m{colors['END']} {colors['GREEN']}(Min Required: {MIN_SURFACE_CLEARANCE} m){colors['END']}")
                    print(f"  {colors['BOLD']}Maximum Tension:{colors['END']} {colors['YELLOW']}{metrics['max_tension']:.2f} kN{colors['END']} {colors['GREEN']}(Max Allowed: {MAX_TENSION_LIMIT} kN){colors['END']}")
                    print(f"  {colors['BOLD']}    Cable/Location:{colors['END']} {colors['CYAN']}{metrics['tension_cable']}, {metrics['max_tension_location']}{colors['END']}")
                    print(f"  {colors['BOLD']}Minimum Bend Radius:{colors['END']} {colors['YELLOW']}{metrics['min_bend_radius']:.2f} m{colors['END']} {colors['GREEN']}(Min Required: {MIN_BEND_RADIUS_LIMIT} m){colors['END']}")
                    print(f"  {colors['BOLD']}    Cable/Location:{colors['END']} {colors['CYAN']}{metrics['bend_cable']}, {metrics['min_bend_radius_location']}{colors['END']}")
                    
                except Exception as e:
                    logger.error(f"Error collecting metrics for the best solution: {e}")
                    logger.error(traceback.format_exc())
                    # Set default metrics in case of error
                    metrics = {
                        "seabed_clearance": 0.0, 
                        "surface_clearance": 0.0,
                        "max_tension": 0.0,
                        "tension_cable": "None",
                        "max_tension_location": "None",
                        "min_bend_radius": float('inf'),
                        "bend_cable": "None",
                        "min_bend_radius_location": "None"
                    }
                    # Still show the metrics section but with default values
                    graphics.print_section_header("SOLUTION METRICS")
                    print(f"  {colors['BOLD']}Seabed Clearance:{colors['END']} {colors['YELLOW']}{metrics['seabed_clearance']:.2f} m{colors['END']} {colors['GREEN']}(Min Required: {MIN_SEABED_CLEARANCE} m){colors['END']}")
                    print(f"  {colors['BOLD']}Surface Clearance:{colors['END']} {colors['YELLOW']}{metrics['surface_clearance']:.2f} m{colors['END']} {colors['GREEN']}(Min Required: {MIN_SURFACE_CLEARANCE} m){colors['END']}")
                    print(f"  {colors['BOLD']}Maximum Tension:{colors['END']} {colors['YELLOW']}{metrics['max_tension']:.2f} kN{colors['END']} {colors['GREEN']}(Max Allowed: {MAX_TENSION_LIMIT} kN){colors['END']}")
                    print(f"  {colors['BOLD']}    Cable/Location:{colors['END']} {colors['CYAN']}{metrics['tension_cable']}, {metrics['max_tension_location']}{colors['END']}")
                    print(f"  {colors['BOLD']}Minimum Bend Radius:{colors['END']} {colors['YELLOW']}{metrics['min_bend_radius']:.2f} m{colors['END']} {colors['GREEN']}(Min Required: {MIN_BEND_RADIUS_LIMIT} m){colors['END']}")
                    print(f"  {colors['BOLD']}    Cable/Location:{colors['END']} {colors['CYAN']}{metrics['bend_cable']}, {metrics['min_bend_radius_location']}{colors['END']}")
                    print(f"\n  {colors['RED']}Error collecting metrics: {str(e)}{colors['END']}")

            # Generate plots for the best solution
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate cable profile plot if BestSolution.dat exists
            try:
                if os.path.exists("BestSolution.dat"):
                    # We already loaded the model above in some cases, but might need to reload
                    # if we're using cached metrics
                    if 'model' not in locals():
                        model = load_model("BestSolution.dat")
                    
                    # Generate cable profile plot
                    cable_plot_path = os.path.join(output_dir, "cable_profiles.png")
                    plot_cable_profiles(model, save_path=cable_plot_path)
                    logger.info(f"Cable profile plot saved to {cable_plot_path}")
                    # Also generate per-offset plots
                    plot_cable_profiles_by_offset(model, output_dir)
                    logger.info("Per-offset cable profile plots created.")
            except Exception as e:
                logger.error(f"Error generating cable profile plot: {e}")
                logger.error(traceback.format_exc())
                print(f"\n  {colors['RED']}Error generating cable profile plot: {str(e)}{colors['END']}")

            # Create an optimization report file
            try:
                report_path = os.path.join("output", "optimization_report.txt")
                with open(report_path, "w") as f:
                    f.write("==== OPTIMIZATION REPORT ====\n\n")
                    
                    f.write("== BEST SOLUTION PARAMETERS ==\n")
                    f.write(f"Buoyancy Module Mass: {bm_mass:.1f} kg\n")
                    f.write(f"Start Arc Length: {start_arc_length:.1f} m\n")
                    f.write(f"Number of Modules: {int(num_modules)}\n")
                    f.write(f"Fitness Score: {best_fitness:.2f}\n")
                    f.write(f"Found in Generation: {best_solution_idx}\n\n")
                    
                    f.write("== SOLUTION METRICS ==\n")
                    # Use same metrics as displayed in the terminal
                    f.write(f"Seabed Clearance: {metrics.get('seabed_clearance', 0.0):.2f} m (Min Required: {MIN_SEABED_CLEARANCE} m)\n")
                    if 'seabed_cable' in metrics and metrics['seabed_cable']:
                        f.write(f"    Cable/Location: {metrics.get('seabed_cable', 'None')}, arc={metrics.get('seabed_arc', 0.0):.2f}m\n")
                    
                    f.write(f"Surface Clearance: {metrics.get('surface_clearance', 0.0):.2f} m (Min Required: {MIN_SURFACE_CLEARANCE} m)\n")
                    if 'surface_cable' in metrics and metrics['surface_cable']:
                        f.write(f"    Cable/Location: {metrics.get('surface_cable', 'None')}, arc={metrics.get('surface_arc', 0.0):.2f}m\n")
                    
                    f.write(f"Maximum Tension: {metrics.get('max_tension', 0.0):.2f} kN (Max Allowed: {MAX_TENSION_LIMIT} kN)\n")
                    if 'tension_cable' in metrics and metrics['tension_cable']:
                        f.write(f"    Cable/Location: {metrics.get('tension_cable', 'None')}, {metrics.get('max_tension_location', 'None')}\n")
                    
                    f.write(f"Minimum Bend Radius: {metrics.get('min_bend_radius', float('inf')):.2f} m (Min Required: {MIN_BEND_RADIUS_LIMIT} m)\n")
                    if 'bend_cable' in metrics and metrics['bend_cable']:
                        f.write(f"    Cable/Location: {metrics.get('bend_cable', 'None')}, {metrics.get('min_bend_radius_location', 'None')}\n\n")
                    
                    f.write("== OPTIMIZATION STATISTICS ==\n")
                    # Calculate total evaluations using the same formula
                    initial_pop_evals = SOL_PER_POP
                    subsequent_gen_evals = num_generations * (SOL_PER_POP - ga_config["KEEP_PARENTS"])
                    total_evaluations = initial_pop_evals + subsequent_gen_evals
                    
                    # Ensure completed evaluations doesn't exceed the expected total
                    completed_evals = min(optimization_state.completed_evaluations, total_evaluations)
                    
                    f.write(f"Evaluations Completed: {completed_evals}/{total_evaluations}\n")
                    f.write(f"Generations: {NUM_GENERATIONS}\n")
                    f.write(f"Average Solution Time: {optimization_state.get_avg_solution_time():.2f} seconds\n")
                    elapsed = time.time() - optimization_state.start_time
                    f.write(f"Total Optimization Time: {elapsed:.2f} seconds ({optimization_state.format_time_exact(elapsed)})\n")
                    f.write(f"Population Size: {SOL_PER_POP}\n")
                    f.write(f"Keep Parents: {KEEP_PARENTS}\n")
                    f.write(f"Mutation Probability: {MUTATION_PROBABILITY:.2f}\n")
                    f.write(f"Crossover Probability: {CROSSOVER_PROBABILITY:.2f}\n\n")
                    
                    f.write("== CONFIG PARAMETERS ==\n")
                    f.write(f"Parameter Bounds:\n")
                    f.write(f"  BM Mass: {PARAM_BOUNDS['bm_mass'][0]}-{PARAM_BOUNDS['bm_mass'][1]} kg\n")
                    f.write(f"  Start Arc Length: {PARAM_BOUNDS['start_arc_length'][0]}-{PARAM_BOUNDS['start_arc_length'][1]} m\n")
                    f.write(f"  Number of Modules: {PARAM_BOUNDS['num_modules'][0]}-{PARAM_BOUNDS['num_modules'][1]}\n\n")
                    
                    f.write(f"Safety Constraints:\n")
                    f.write(f"  Min Seabed Clearance: {MIN_SEABED_CLEARANCE} m\n")
                    f.write(f"  Min Surface Clearance: {MIN_SURFACE_CLEARANCE} m\n")
                    f.write(f"  Max Tension: {MAX_TENSION_LIMIT} kN\n")
                    f.write(f"  Min Bend Radius: {MIN_BEND_RADIUS_LIMIT} m\n")
                
                print(f"\n  {colors['GREEN']}Optimization report saved to {report_path}{colors['END']}")
                
            except Exception as e:
                logger.error(f"Error generating optimization report: {e}")
                logger.error(traceback.format_exc())
                print(f"\n  {colors['RED']}Error generating report: {str(e)}{colors['END']}")
        else:
            print(f"\n{colors['RED']}No valid solution was found during optimization.{colors['END']}")
            print(f"{colors['YELLOW']}Try relaxing constraints, broadening parameter bounds, or increasing the number of generations/population size.{colors['END']}")
            print(f"{colors['YELLOW']}No cable profile plots or metrics are available because no valid solution was found.{colors['END']}")
            
            # Even though no valid solution was found, still show the metrics section to explain why
            # solutions might have been rejected
            graphics.print_section_header("CONSTRAINT INFORMATION")
            print(f"  {colors['BOLD']}No valid solution was found that satisfied all constraints.{colors['END']}\n")
            
            print(f"  {colors['BOLD']}Safety Constraints:{colors['END']}")
            print(f"  {colors['BOLD']}Min Seabed Clearance:{colors['END']} {colors['YELLOW']}{MIN_SEABED_CLEARANCE} m{colors['END']}")
            print(f"  {colors['BOLD']}Min Surface Clearance:{colors['END']} {colors['YELLOW']}{MIN_SURFACE_CLEARANCE} m{colors['END']}")
            print(f"  {colors['BOLD']}Max Tension Limit:{colors['END']} {colors['YELLOW']}{MAX_TENSION_LIMIT} kN{colors['END']}")
            print(f"  {colors['BOLD']}Min Bend Radius:{colors['END']} {colors['YELLOW']}{MIN_BEND_RADIUS_LIMIT} m{colors['END']}")
            
            print(f"\n  {colors['BOLD']}Parameter Bounds:{colors['END']}")
            print(f"  {colors['BOLD']}BM Mass:{colors['END']} {colors['YELLOW']}{PARAM_BOUNDS['bm_mass'][0]}-{PARAM_BOUNDS['bm_mass'][1]} kg{colors['END']}")
            print(f"  {colors['BOLD']}Start Arc Length:{colors['END']} {colors['YELLOW']}{PARAM_BOUNDS['start_arc_length'][0]}-{PARAM_BOUNDS['start_arc_length'][1]} m{colors['END']}")
            print(f"  {colors['BOLD']}Number of Modules:{colors['END']} {colors['YELLOW']}{PARAM_BOUNDS['num_modules'][0]}-{PARAM_BOUNDS['num_modules'][1]}{colors['END']}")
            
            # Try to find information about the best unsuccessful solution
            try:
                if hasattr(ga_instance, 'best_solution') and ga_instance.best_solution() is not None:
                    best_genes = ga_instance.best_solution()[0]
                    best_fitness = ga_instance.best_solution()[1]
                    
                    if best_genes is not None:
                        bm_mass, start_arc_length, num_modules = denormalize_solution(best_genes)
                        
                        print(f"\n  {colors['BOLD']}Best Unsuccessful Solution:{colors['END']}")
                        print(f"  {colors['BOLD']}BM Mass:{colors['END']} {colors['YELLOW']}{bm_mass:.1f} kg{colors['END']}")
                        print(f"  {colors['BOLD']}Start Arc Length:{colors['END']} {colors['YELLOW']}{start_arc_length:.1f} m{colors['END']}")
                        print(f"  {colors['BOLD']}Number of Modules:{colors['END']} {colors['YELLOW']}{int(num_modules)}{colors['END']}")
                        print(f"  {colors['BOLD']}Fitness Score:{colors['END']} {colors['RED']}{best_fitness}{colors['END']}")
                
                # Suggest possible improvements
                print(f"\n  {colors['BOLD']}Suggestions:{colors['END']}")
                print(f"  {colors['CYAN']}- Try relaxing the constraints{colors['END']}")
                print(f"  {colors['CYAN']}- Increase the number of generations or population size{colors['END']}")
                print(f"  {colors['CYAN']}- Modify the parameter bounds{colors['END']}")
                print(f"  {colors['CYAN']}- Check the model configuration for potential issues{colors['END']}")
            except Exception as e:
                logger.error(f"Error displaying best unsuccessful solution: {e}")
                
            # Create an optimization report even for unsuccessful runs
            try:
                report_path = os.path.join("output", "optimization_report.txt")
                with open(report_path, "w") as f:
                    f.write("==== OPTIMIZATION REPORT (NO VALID SOLUTION) ====\n\n")
                    
                    f.write("== STATUS ==\n")
                    f.write("No valid solution was found that satisfied all constraints.\n\n")
                    
                    if hasattr(ga_instance, 'best_solution') and ga_instance.best_solution() is not None:
                        best_genes = ga_instance.best_solution()[0]
                        best_fitness = ga_instance.best_solution()[1]
                        
                        if best_genes is not None:
                            bm_mass, start_arc_length, num_modules = denormalize_solution(best_genes)
                            
                            f.write("== BEST UNSUCCESSFUL SOLUTION ==\n")
                            f.write(f"Buoyancy Module Mass: {bm_mass:.1f} kg\n")
                            f.write(f"Start Arc Length: {start_arc_length:.1f} m\n")
                            f.write(f"Number of Modules: {int(num_modules)}\n")
                            f.write(f"Fitness Score: {best_fitness}\n\n")
                    
                    f.write("== SAFETY CONSTRAINTS ==\n")
                    f.write(f"Min Seabed Clearance: {MIN_SEABED_CLEARANCE} m\n")
                    f.write(f"Min Surface Clearance: {MIN_SURFACE_CLEARANCE} m\n")
                    f.write(f"Max Tension Limit: {MAX_TENSION_LIMIT} kN\n")
                    f.write(f"Min Bend Radius: {MIN_BEND_RADIUS_LIMIT} m\n\n")
                    
                    f.write("== PARAMETER BOUNDS ==\n")
                    f.write(f"BM Mass: {PARAM_BOUNDS['bm_mass'][0]}-{PARAM_BOUNDS['bm_mass'][1]} kg\n")
                    f.write(f"Start Arc Length: {PARAM_BOUNDS['start_arc_length'][0]}-{PARAM_BOUNDS['start_arc_length'][1]} m\n")
                    f.write(f"Number of Modules: {PARAM_BOUNDS['num_modules'][0]}-{PARAM_BOUNDS['num_modules'][1]}\n\n")
                    
                    f.write("== OPTIMIZATION STATISTICS ==\n")
                    # Calculate total evaluations using the same formula
                    initial_pop_evals = SOL_PER_POP
                    subsequent_gen_evals = num_generations * (SOL_PER_POP - ga_config["KEEP_PARENTS"])
                    total_evaluations = initial_pop_evals + subsequent_gen_evals
                    
                    # Ensure completed evaluations doesn't exceed the expected total
                    completed_evals = min(optimization_state.completed_evaluations, total_evaluations)
                    
                    f.write(f"Evaluations Completed: {completed_evals}/{total_evaluations}\n")
                    f.write(f"Generations: {NUM_GENERATIONS}\n")
                    f.write(f"Average Solution Time: {optimization_state.get_avg_solution_time():.2f} seconds\n")
                    elapsed = time.time() - optimization_state.start_time
                    f.write(f"Total Optimization Time: {elapsed:.2f} seconds ({optimization_state.format_time_exact(elapsed)})\n")
                    f.write(f"Population Size: {SOL_PER_POP}\n")
                    f.write(f"Keep Parents: {KEEP_PARENTS}\n")
                    f.write(f"Mutation Probability: {MUTATION_PROBABILITY:.2f}\n")
                    f.write(f"Crossover Probability: {CROSSOVER_PROBABILITY:.2f}\n\n")
                    
                    f.write("== SUGGESTIONS ==\n")
                    f.write("- Try relaxing the constraints\n")
                    f.write("- Increase the number of generations or population size\n")
                    f.write("- Modify the parameter bounds\n")
                    f.write("- Check the model configuration for potential issues\n")
                
                print(f"\n  {colors['GREEN']}Optimization report saved to {report_path}{colors['END']}")
                
            except Exception as e:
                logger.error(f"Error generating optimization report for unsuccessful run: {e}")
                logger.error(traceback.format_exc())
                print(f"\n  {colors['RED']}Error generating report: {str(e)}{colors['END']}")
        
        # Show optimization statistics
        graphics.print_section_header("OPTIMIZATION STATISTICS")
        
        # Calculate total evaluations using the corrected formula
        initial_pop_evals = SOL_PER_POP
        subsequent_gen_evals = (num_generations - 1) * (SOL_PER_POP - ga_config["KEEP_PARENTS"])
        total_evaluations = initial_pop_evals + subsequent_gen_evals
        
        # Calculate progress based on current generation
        current_gen = min(optimization_state.current_generation, optimization_state.total_generations)
        if current_gen == 0:
            # First generation - only count evaluations in this generation
            expected_evals = min(optimization_state.generation_solutions_evaluated, SOL_PER_POP)
        else:
            # Later generations - count all previous gens + current progress
            expected_evals = SOL_PER_POP + (current_gen - 1) * (SOL_PER_POP - ga_config["KEEP_PARENTS"]) + min(optimization_state.generation_solutions_evaluated, SOL_PER_POP - ga_config["KEEP_PARENTS"])
        
        # Ensure expected_evals doesn't exceed total_evaluations
        expected_evals = min(expected_evals, total_evaluations)
        
        # Ensure completed evaluations is consistent with generation progress
        # This prevents display issues where evaluations show 100% but generations aren't complete
        completed_evals = min(optimization_state.completed_evaluations, expected_evals)
        
        # Log the calculation
        logger.debug(f"Progress calculation: current_gen={current_gen}, expected_evals={expected_evals}, completed_evals={completed_evals}")
        
        print(f"  {colors['BOLD']}Evaluations:{colors['END']} {colors['YELLOW']}{completed_evals}/{total_evaluations}{colors['END']}")
        print(f"  {colors['BOLD']}Timed Out Solutions:{colors['END']} {colors['YELLOW']}{optimization_state.get_timed_out_evaluations()}{colors['END']}")
        print(f"  {colors['BOLD']}Total Generations:{colors['END']} {colors['YELLOW']}{NUM_GENERATIONS}{colors['END']}")
        avg_time = optimization_state.get_avg_solution_time()
        print(f"  {colors['BOLD']}Average Solution Time:{colors['END']} {colors['YELLOW']}{avg_time:.2f} s{colors['END']}")
        elapsed = time.time() - optimization_state.start_time
        print(f"  {colors['BOLD']}Total Time:{colors['END']} {colors['YELLOW']}{graphics.format_time(elapsed)}{colors['END']}")
        
        # Plot information
        output_dir = "output"
        graphics.print_section_header("OUTPUT FILES")
        
        print(f"  {colors['GREEN']}Files have been saved to: {colors['YELLOW']}{output_dir}{colors['END']}")
        
        # List the files that were actually created
        created_files = []
        if os.path.exists(os.path.join(output_dir, "cable_profiles.png")):
            created_files.append(f"  {colors['BOLD']}Cable Profile:{colors['END']} {colors['YELLOW']}cable_profiles.png{colors['END']}")
        
        if os.path.exists(os.path.join(output_dir, "pygad_fitness.png")):
            created_files.append(f"  {colors['BOLD']}Fitness History:{colors['END']} {colors['YELLOW']}pygad_fitness.png{colors['END']}")
        
        if os.path.exists(os.path.join(output_dir, "optimization_report.txt")):
            created_files.append(f"  {colors['BOLD']}Optimization Report:{colors['END']} {colors['YELLOW']}optimization_report.txt{colors['END']}")
            
        if os.path.exists("BestSolution.dat"):
            created_files.append(f"  {colors['BOLD']}Best Solution Model:{colors['END']} {colors['YELLOW']}BestSolution.dat{colors['END']}")
        
        # If no files were created, show a message
        if not created_files:
            print(f"  {colors['RED']}No output files were created{colors['END']}")
        else:
            # Print each created file
            for file_info in created_files:
                print(file_info)
        
        # Print seabed and footer
        graphics.print_seabed(display_width)
        graphics.print_footer(display_width)
        
        # Use PyGAD's built-in plotting functionality
        try:
            # Create output directory
            os.makedirs("output", exist_ok=True)
            
            # Generate and save the plot
            fig = ga_instance.plot_fitness()
            plt.tight_layout()
            plt.savefig("output/pygad_fitness.png")
            plt.close(fig)
            
            logger.info("Fitness history plot saved to output/pygad_fitness.png")
        except Exception as e:
            logger.error(f"Error generating fitness history plot: {e}")
        
        # After reporting metrics, write all metrics to output/metrics.txt
        try:
            metrics_path = os.path.join("output", "metrics.txt")
            with open(metrics_path, "w") as f:
                f.write("==== SOLUTION METRICS ====" + "\n\n")
                for k, v in metrics.items():
                    if k == "suspended_lengths":
                        f.write("Suspended Cable Lengths (m):\n")
                        for cable, length in v.items():
                            f.write(f"  {cable}: {length:.2f} m\n")
                    else:
                        f.write(f"{k}: {v}\n")
        except Exception as e:
            print(f"Warning: Failed to write metrics.txt: {e}")
        
        # Return the best solution
        return best_solution, best_fitness
    
    except Exception as e:
        logger.error(f"Error in optimization process: {e}")
        logger.error(traceback.format_exc())
        
        # Mark optimization as complete to stop progress display
        optimization_state.optimization_complete = True
        
        # Display error to user
        if optimization_state.terminal_output_enabled:
            print(f"\nError in optimization process: {e}")
        
        return {}, float('-inf')
