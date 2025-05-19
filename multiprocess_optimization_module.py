"""
Multiprocessing version of optimization module for buoyancy module configuration using PyGAD
"""

import os
import sys
import time
import math
import logging
import multiprocessing
import threading
import traceback
import builtins
import warnings
import random
import numpy as np
import pygad
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from config import (
    MODEL_PATH, CABLE_NAMES, BM_NAMES, BM_EoL_MASS_FACTOR,
    PARAM_BOUNDS, MIN_SEABED_CLEARANCE, MIN_SURFACE_CLEARANCE,
    MAX_TENSION_LIMIT, MIN_BEND_RADIUS_LIMIT, MODULE_COST_FACTOR,
    PERFORMANCE_WEIGHTS, NUM_GENERATIONS,
    MUTATION_PROBABILITY, CROSSOVER_PROBABILITY, KEEP_PARENTS,
    USE_MULTIPROCESSING, NUM_PROCESSES, 
    ENABLE_SOLUTION_CACHE,
    PROGRESS_UPDATE_INTERVAL,
    STATIC_ANALYSIS_TIMEOUT, MULTIPROCESS_TIMEOUT_MULTIPLIER,
    BM_SPACING,
    NUM_PARENTS_MATING, SOL_PER_POP, PARENT_SELECTION_TYPE,
    CROSSOVER_TYPE, MUTATION_TYPE, MUTATION_PERCENT_GENES, MUTATION_BY_REPLACEMENT,
    GENE_TYPE, INIT_RANGE_LOW, INIT_RANGE_HIGH, ALLOW_DUPLICATE_GENES, STOP_CRITERIA
)

from orcaflex_utils import (
    load_model, update_bm_mass, configure_buoyancy_modules, run_static_analysis,
    get_seabed_clearance, get_surface_clearance, get_max_tension, get_min_bend_radius,
    save_model, plot_cable_profiles, plot_cable_profiles_by_offset
)

from optimization_utils import (
    normalize_solution, denormalize_solution, save_best_solution
)
from progress_tracker import optimization_state, OptimizationState
import graphics
import csv_logger
import copy
import pandas as pd

# A fixed-size dictionary to limit memory usage from cached solutions.
# Automatically discards the oldest entries once the maximum size is exceeded.
from collections import OrderedDict

class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, maxlen=2000, **kwargs):
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        # Move existing key to end (most recently used)
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        # If we exceed the max length, remove the oldest entry
        if len(self) > self.maxlen:
            self.popitem(last=False)


def clear_terminal():
    """Clear the terminal screen"""
    graphics.clear_terminal()

# Constants
MAX_FITNESS = True  # True for maximizing fitness, False for minimizing

# Setup logger
logger = logging.getLogger(__name__)

# Suppress specific matplotlib warnings
warnings.filterwarnings("ignore", message="No artists with labels found to put in legend")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

def init_logging(console_output=False):
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
        filemode="a",  # Use 'a' to append instead of overwriting
        encoding="utf-8"
    )
    
    # Add console handler only if requested
    if console_output:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

# Add multiprocessing-specific attributes to the shared optimization_state
optimization_state.pool = None
optimization_state.solution_results = {}
optimization_state.active_solutions = 0
optimization_state.current_batch_completed = 0
optimization_state.current_batch_total = 0
optimization_state.current_solution = [0.0] * 3
optimization_state.current_solution_idx = None
# Use limited-size caches to avoid excessive memory use during long optimisations.
optimization_state.solutions_cache = LimitedSizeDict(maxlen=2000)
optimization_state.metrics_cache = LimitedSizeDict(maxlen=2000)
optimization_state.pending_solutions = []
optimization_state.pending_indices = []
optimization_state.pending_cache_keys = []
optimization_state.display_thread = None
optimization_state.stop_display_thread = False
optimization_state.display_needs_update = False

def initialize_pool():
    """Initialize the process pool"""
    if optimization_state.pool is None:
        optimization_state.pool = Pool(processes=NUM_PROCESSES)
        logger.info(f"Initialized process pool with {NUM_PROCESSES} parallel processes")

def cleanup():
    """Clean up multiprocessing resources"""
    # Stop the display update thread if it's running
    if optimization_state.display_thread is not None and optimization_state.display_thread.is_alive():
        optimization_state.stop_display_thread = True
        optimization_state.display_thread.join(timeout=2.0)
    
    # Clean up the process pool
    if optimization_state.pool:
        optimization_state.pool.close()
        optimization_state.pool.join()

def start_display_update_thread():
    """Start a thread that updates the display regularly"""
    if optimization_state.display_thread is None or not optimization_state.display_thread.is_alive():
        optimization_state.stop_display_thread = False
        optimization_state.display_thread = threading.Thread(target=_display_updater)
        optimization_state.display_thread.daemon = True
        optimization_state.display_thread.start()
        logger.info("Started display update thread")

# --- Progress Bar Diagnostics ---
_last_progress_state = None
_last_progress_time = 0

def update_progress_bars(force_full_update=False):
    """Update terminal with current optimization status"""
    global _last_progress_state, _last_progress_time
    if not optimization_state.terminal_output_enabled or optimization_state.total_evaluations is None:
        return
    if not force_full_update and optimization_state.active_solutions == 0 and optimization_state.current_batch_total == 0:
        return
    # Only print/log progress if state has changed or at most once per 10 seconds
    import time as _time
    state = (
        optimization_state.completed_evaluations,
        optimization_state.generation_solutions_evaluated,
        optimization_state.current_generation,
        optimization_state.current_batch_completed,
        optimization_state.current_batch_total
    )
    now = _time.time()
    should_update = force_full_update or state != _last_progress_state or now - _last_progress_time > 10.0
    if should_update:
        logger.info(f"[PROGRESS-DEBUG] Progress update: completed={optimization_state.completed_evaluations}, gen_eval={optimization_state.generation_solutions_evaluated}, gen={optimization_state.current_generation}, batch={optimization_state.current_batch_completed}/{optimization_state.current_batch_total}")
        _last_progress_state = state
        _last_progress_time = now
    if not force_full_update and optimization_state.active_solutions == 0 and optimization_state.current_batch_total == 0:
        return
    if optimization_state.current_batch_total > 0 and optimization_state.current_batch_completed >= optimization_state.current_batch_total:
        optimization_state.active_solutions = 0
    if optimization_state.active_solutions == 0 and optimization_state.current_batch_total > 0:
        optimization_state.current_batch_total = 0
        optimization_state.current_batch_completed = 0
    # Clear the terminal before printing progress
    graphics.clear_terminal()
    # Calculate progress percentages
    gen_percent = min((optimization_state.generation_solutions_evaluated / optimization_state.generation_solutions_total) * 100 
                     if optimization_state.generation_solutions_total > 0 else 0, 100)
    total_percent = min((optimization_state.completed_evaluations / optimization_state.total_evaluations) * 100 
                      if optimization_state.total_evaluations > 0 else 0, 100)
    # Calculate times
    elapsed = time.time() - optimization_state.start_time
    elapsed_rounded = int(elapsed // 10) * 10  # Round down to nearest 10 seconds
    avg_time = optimization_state.get_avg_solution_time()
    remaining_evals = max(0, optimization_state.total_evaluations - optimization_state.completed_evaluations)
    est_remaining = remaining_evals * avg_time if avg_time > 0 else 0
    # Format strings
    elapsed_str = optimization_state.format_time_exact(elapsed_rounded)
    avg_str = f"{avg_time:.1f}s"
    remain_str = optimization_state.format_time_exact(est_remaining)
    current_gen = optimization_state.current_generation
    current_pop = current_gen + 1
    total_pops = optimization_state.total_generations + 1
    gen_str = f"{current_gen}/{optimization_state.total_generations - 1}"  # Shows gen 0 to 49
    pop_str = f"{current_pop}/{total_pops}"                                # Shows pop 1 to 51
    eval_str = f"{min(optimization_state.completed_evaluations, optimization_state.total_evaluations)}/{optimization_state.total_evaluations}"

    # Calculate worker and batch info
    optimal_workers = min(SOL_PER_POP, NUM_PROCESSES)
    workers_str = f"{optimization_state.active_solutions}/{optimal_workers}"
    batch_str = f"{optimization_state.current_batch_completed}/{optimization_state.current_batch_total}" if optimization_state.current_batch_total > 0 else "0/0"
    # Print progress
    logger.info(f"[DISPLAY-DEBUG] Calling print_multiprocess_optimization_progress at {time.time()} (elapsed={elapsed_rounded}s, batch={optimization_state.current_batch_completed}/{optimization_state.current_batch_total})")
    graphics.print_multiprocess_optimization_progress(
        gen_percent=gen_percent,
        total_percent=total_percent,
        elapsed=elapsed_rounded,
        avg_time=avg_time,
        est_remaining=est_remaining,
        current_gen=current_gen,
        total_generations=optimization_state.total_generations,
        current_pop=current_pop,
        total_populations=total_pops,
        completed_evaluations=optimization_state.completed_evaluations,
        total_evaluations=optimization_state.total_evaluations,
        workers_str=workers_str,
        batch_str=batch_str
    )
    import sys
    sys.stdout.flush()
    optimization_state.display_needs_update = False

def init_worker():
    """Initialize worker process with global state"""
    global optimization_state, solution_queue, lock
    # These are initialized in the parent process and passed to workers

def evaluate_solution_wrapper(solution_dict, generation=None, queue=None):
    """
    Wrapper function for solution evaluation in parallel processes.
    Handles both normalized and denormalized solution formats.
    Now returns results to the main process for CSV logging.
    Only the progress handler method is used for timeouts in multiprocess mode.
    """
    import logging
    import os
    import time as _time
    start_time = _time.time()  # Start timing at the very top
    # Explicitly configure logging in the worker process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="log.txt",
        filemode="a"
    )
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"[WORKER-DEBUG] PID={os.getpid()} input solution_dict: {solution_dict}")
        solution = solution_dict.get("solution", solution_dict)
        # Denormalize the solution (genes are normalized floats in [0, 1])
        bm_mass, start_arc_length, num_modules = denormalize_solution(solution)
        bm_mass = int(round(bm_mass))
        start_arc_length = int(round(start_arc_length))
        num_modules = int(round(num_modules))
        logger.info(f"[WORKER-DEBUG] PID={os.getpid()} denormalized: bm_mass={bm_mass}, start_arc_length={start_arc_length}, num_modules={num_modules}")
        solution_dict = {
            "bm_mass": bm_mass,
            "start_arc_length": start_arc_length,
            "num_modules": num_modules
        }
        # Log after denormalization
        logger.info(f"[WORKER-DEBUG] PID={os.getpid()} using solution_dict: {solution_dict}")
        metrics = {}
        try:
            logger.info(f"[WORKER-DEBUG] PID={os.getpid()} loading model and configuring parameters: {solution_dict}")
            model = load_model(MODEL_PATH)
            logger.info(f"[WORKER-DEBUG] PID={os.getpid()} loaded model instance id={id(model)} from {MODEL_PATH}")
            eol_bm_mass = round(solution_dict["bm_mass"] * BM_EoL_MASS_FACTOR, 2)
            # Log BM mass before and after update
            logger.info(f"[WORKER-DEBUG] PID={os.getpid()} BM mass before update: SoL={model[BM_NAMES['SoL']].Mass}, EoL={model[BM_NAMES['EoL']].Mass}")
            update_bm_mass(model, BM_NAMES["SoL"], solution_dict["bm_mass"])
            update_bm_mass(model, BM_NAMES["EoL"], eol_bm_mass)
            logger.info(f"[WORKER-DEBUG] PID={os.getpid()} BM mass after update: SoL={model[BM_NAMES['SoL']].Mass}, EoL={model[BM_NAMES['EoL']].Mass}")
            sol_cables = [cable for cable in CABLE_NAMES if "SoL" in cable]
            eol_cables = [cable for cable in CABLE_NAMES if "EoL" in cable]
            for cable in sol_cables:
                configure_buoyancy_modules(
                    model, cable, BM_NAMES["SoL"], 
                    solution_dict["start_arc_length"], 
                    solution_dict["num_modules"], 
                    BM_SPACING
                )
                # Log cable attachment positions
                try:
                    att = model[cable].AttachmentZ
                    logger.info(f"[WORKER-DEBUG] PID={os.getpid()} {cable} attachments: {[att[i] for i in range(model[cable].NumberOfAttachments)]}")
                except Exception as e:
                    logger.warning(f"[WORKER-DEBUG] PID={os.getpid()} Could not log {cable} attachments: {e}")
            for cable in eol_cables:
                configure_buoyancy_modules(
                    model, cable, BM_NAMES["EoL"], 
                    solution_dict["start_arc_length"], 
                    solution_dict["num_modules"], 
                    BM_SPACING
                )
                try:
                    att = model[cable].AttachmentZ
                    logger.info(f"[WORKER-DEBUG] PID={os.getpid()} {cable} attachments: {[att[i] for i in range(model[cable].NumberOfAttachments)]}")
                except Exception as e:
                    logger.warning(f"[WORKER-DEBUG] PID={os.getpid()} Could not log {cable} attachments: {e}")
            logger.info(f"[WORKER-DEBUG] PID={os.getpid()} before model.CalculateStatics() for: {solution_dict}")
            # print(f"[WORKER-PRINT] PID={os.getpid()} before model.CalculateStatics() for: {solution_dict}")
            timeout = STATIC_ANALYSIS_TIMEOUT * MULTIPROCESS_TIMEOUT_MULTIPLIER
            result = run_static_analysis(model, timeout=timeout)
            logger.info(f"[WORKER-DEBUG] PID={os.getpid()} run_static_analysis result: {result}")
            # print(f"[WORKER-PRINT] PID={os.getpid()} run_static_analysis result: {result}")
            elapsed = _time.time() - start_time
            solution_time = round(elapsed, 1)
            if result == 'timeout':
                logger.warning(f"[WORKER-DEBUG] PID={os.getpid()} statics timed out, skipping metrics extraction.")
                genes = {
                    "bm_mass": int(round(solution_dict["bm_mass"])),
                    "num_modules": int(round(solution_dict["num_modules"])),
                    "start_arc_length": int(round(solution_dict["start_arc_length"]))
                }
                metric_keys = ["max_tension", "min_bend_radius", "seabed_clearance", "surface_clearance"]
                metrics_formatted = {k: float('nan') for k in metric_keys}
                is_valid = False
                fail_reason = "timeout"
                if queue is not None:
                    queue.put((float('-inf'), metrics_formatted, genes, is_valid, fail_reason, solution_time))
                    logger.info(f"[WORKER-DEBUG] put result in queue: {{...}}, solution_time={solution_time}")
                    return
                return float('-inf'), metrics_formatted, genes, is_valid, fail_reason, solution_time
            if not result:
                logger.warning(f"[WORKER-DEBUG] PID={os.getpid()} statics failed (not timeout), skipping metrics extraction.")
                genes = {
                    "bm_mass": int(round(solution_dict["bm_mass"])),
                    "num_modules": int(round(solution_dict["num_modules"])),
                    "start_arc_length": int(round(solution_dict["start_arc_length"]))
                }
                metric_keys = ["max_tension", "min_bend_radius", "seabed_clearance", "surface_clearance"]
                metrics_formatted = {k: float('nan') for k in metric_keys}
                is_valid = False
                fail_reason = "static analysis failed"
                if queue is not None:
                    queue.put((float('-inf'), metrics_formatted, genes, is_valid, fail_reason, solution_time))
                    logger.info(f"[WORKER-DEBUG] put result in queue: {{...}}, solution_time={solution_time}")
                    return
                return float('-inf'), metrics_formatted, genes, is_valid, fail_reason, solution_time
            # If we reach here, statics succeeded, so extract metrics
            logger.info(f"[WORKER-DEBUG] PID={os.getpid()} statics succeeded, extracting metrics...")
            min_seabed_clearance = float('inf')
            min_surface_clearance = float('inf')
            max_tension = 0
            min_bend_radius = float('inf')
            min_seabed_cable = None
            min_surface_cable = None
            max_tension_cable = None
            min_bend_cable = None
            for cable in CABLE_NAMES:
                seabed_clearance, seabed_arc = get_seabed_clearance(model, cable)
                surface_clearance, surface_arc = get_surface_clearance(model, cable)
                tension, tension_arc, tension_loc = get_max_tension(model, cable)
                bend_radius, bend_arc, bend_loc = get_min_bend_radius(model, cable)
                if 0 < seabed_clearance < min_seabed_clearance:
                    min_seabed_clearance = seabed_clearance
                    min_seabed_cable = cable
                if 0 < surface_clearance < min_surface_clearance:
                    min_surface_clearance = surface_clearance
                    min_surface_cable = cable
                if tension > max_tension:
                    max_tension = tension
                    max_tension_cable = cable
                if 0 < bend_radius < min_bend_radius:
                    min_bend_radius = bend_radius
                    min_bend_cable = cable
            metrics.update({
                "seabed_clearance": min_seabed_clearance,
                "seabed_cable": min_seabed_cable,
                "surface_clearance": min_surface_clearance,
                "surface_cable": min_surface_cable,
                "max_tension": max_tension,
                "tension_cable": max_tension_cable,
                "min_bend_radius": min_bend_radius,
                "bend_cable": min_bend_cable
            })
            logger.info(f"[WORKER-DEBUG] PID={os.getpid()} extracted metrics: {metrics}")
            genes = {
                "bm_mass": int(round(solution_dict["bm_mass"])),
                "num_modules": int(round(solution_dict["num_modules"])),
                "start_arc_length": int(round(solution_dict["start_arc_length"]))
            }
            metric_keys = ["max_tension", "min_bend_radius", "seabed_clearance", "surface_clearance"]
            metrics_formatted = {k: (round(metrics[k], 2) if isinstance(metrics.get(k), float) else metrics.get(k, float('nan'))) for k in metric_keys}
            is_valid = not (
                min_seabed_clearance < MIN_SEABED_CLEARANCE or
                min_surface_clearance < MIN_SURFACE_CLEARANCE or
                max_tension > MAX_TENSION_LIMIT or
                min_bend_radius < MIN_BEND_RADIUS_LIMIT
            )
            fail_reason = ""
            if not is_valid:
                reasons = []
                if min_seabed_clearance < MIN_SEABED_CLEARANCE:
                    reasons.append("seabed clearance too low")
                if min_surface_clearance < MIN_SURFACE_CLEARANCE:
                    reasons.append("surface clearance too low")
                if max_tension > MAX_TENSION_LIMIT:
                    reasons.append("max tension too high")
                if min_bend_radius < MIN_BEND_RADIUS_LIMIT:
                    reasons.append("bend radius too small")
                fail_reason = ", ".join(reasons) if reasons else "constraint violation"
            fitness = float('-inf') if not is_valid else 1000 - (solution_dict["num_modules"] * MODULE_COST_FACTOR) + (
                PERFORMANCE_WEIGHTS["seabed_clearance"] * min_seabed_clearance +
                PERFORMANCE_WEIGHTS["surface_clearance"] * min_surface_clearance +
                PERFORMANCE_WEIGHTS["max_tension"] * max_tension +
                PERFORMANCE_WEIGHTS["min_bend_radius"] * min_bend_radius
            )
            if queue is not None:
                queue.put((fitness, metrics_formatted, genes, is_valid, fail_reason, solution_time))
                logger.info(f"[WORKER-DEBUG] put result in queue: {{...}}, solution_time={solution_time}")
                return
            return fitness, metrics_formatted, genes, is_valid, fail_reason, solution_time
        except Exception as e:
            elapsed = _time.time() - start_time
            solution_time = round(elapsed, 1)
            genes = {
                "bm_mass": int(round(solution_dict.get("bm_mass", 0))),
                "num_modules": int(round(solution_dict.get("num_modules", 0))),
                "start_arc_length": int(round(solution_dict.get("start_arc_length", 0)))
            }
            metric_keys = ["max_tension", "min_bend_radius", "seabed_clearance", "surface_clearance"]
            metrics_formatted = {k: float('nan') for k in metric_keys}
            is_valid = False
            fail_reason = str(e)
            if queue is not None:
                queue.put((float('-inf'), metrics_formatted, genes, is_valid, fail_reason, solution_time))
                logger.info(f"[WORKER-DEBUG] put result in queue: {{...}}, solution_time={solution_time}")
                return
            return float('-inf'), metrics_formatted, genes, is_valid, fail_reason, solution_time
    except Exception as e:
        elapsed = _time.time() - start_time
        solution_time = round(elapsed, 1)
        genes = {
            "bm_mass": int(round(solution_dict.get("bm_mass", 0))),
            "num_modules": int(round(solution_dict.get("num_modules", 0))),
            "start_arc_length": int(round(solution_dict.get("start_arc_length", 0)))
        }
        metric_keys = ["max_tension", "min_bend_radius", "seabed_clearance", "surface_clearance"]
        metrics_formatted = {k: float('nan') for k in metric_keys}
        is_valid = False
        fail_reason = f"wrapper_error: {str(e)}"
        if queue is not None:
            queue.put((float('-inf'), metrics_formatted, genes, is_valid, fail_reason, solution_time))
            logger.info(f"[WORKER-DEBUG] put result in queue: {{...}}, solution_time={solution_time}")
            return
        return float('-inf'), metrics_formatted, genes, is_valid, fail_reason, solution_time

def handle_solution_complete(result, idx, results):
    """Callback function to handle completion of individual solutions in a batch."""
    try:
        # Store the result
        results[idx] = result
        # Get the fitness value from the result tuple
        fitness = result[0] if isinstance(result, tuple) else float('-inf')
        # Update progress tracking - increment all relevant counters
        optimization_state.current_batch_completed += 1
        # DO NOT increment generation_solutions_evaluated or completed_evaluations here!
        # These are now only incremented after a batch is processed in fitness_func_parallel
        
        # Determine if solution is valid and get failure reason if not
        fail_reason = ""
        is_valid = not (
            fitness == float('-inf')
            or (isinstance(result, tuple) and isinstance(result[1], dict) and 'error' in result[1])
        )
        
        # Count failures as failed evaluations
        if not is_valid:
            optimization_state.failed_evaluations += 1
            # Get detailed failure reason if available
            if isinstance(result, tuple) and isinstance(result[1], dict):
                metrics = result[1]
                reasons = []
                if metrics.get('seabed_clearance', float('inf')) < MIN_SEABED_CLEARANCE:
                    reasons.append("seabed clearance too low")
                if metrics.get('surface_clearance', float('inf')) < MIN_SURFACE_CLEARANCE:
                    reasons.append("surface clearance too low")
                if metrics.get('max_tension', 0) > MAX_TENSION_LIMIT:
                    reasons.append("max tension too high")
                if metrics.get('min_bend_radius', float('inf')) < MIN_BEND_RADIUS_LIMIT:
                    reasons.append("bend radius too small")
                if 'error' in metrics:
                    reasons.append("error")
                fail_reason = ", ".join(reasons) if reasons else "constraint violation"
            else:
                fail_reason = "constraint violation"
        
        # Get the solution from pending_solutions
        solution = optimization_state.pending_solutions[idx] if idx < len(optimization_state.pending_solutions) else None
        
        # Log the solution
        # import csv_logger
        # ... (remove or comment out the csv_logger.log_solution call and related formatting) ...
        
        # Update best solution if this one is better
        update_best_solution(fitness, solution, log_prefix="[handle_solution_complete] ")
        
        # Decrement active solutions as each one completes
        optimization_state.active_solutions = max(0, optimization_state.active_solutions - 1)
        
        # Ensure batch counts are consistent
        if optimization_state.current_batch_completed > optimization_state.current_batch_total:
            optimization_state.current_batch_completed = optimization_state.current_batch_total
        
        # Log the completion
        if is_valid:
            logger.info(f"Solution {idx} completed with fitness: {fitness}")
        else:
            logger.info(f"Solution {idx} failed: {fail_reason}")
        
        # Update display
        optimization_state.display_needs_update = True
        
    except Exception as e:
        logger.error(f"Error in handle_solution_complete: {e}")
        logger.error(traceback.format_exc())
        # Ensure we still update counters even on error
        optimization_state.current_batch_completed += 1
        # DO NOT increment generation_solutions_evaluated or completed_evaluations here!
        optimization_state.failed_evaluations += 1
        optimization_state.active_solutions = max(0, optimization_state.active_solutions - 1)
    optimization_state.display_needs_update = True

def on_generation(ga_instance):
    """Callback function called at the end of each generation"""
    current_gen = ga_instance.generations_completed

    # Update generation counter in state
    optimization_state.current_generation = current_gen

    # If any leftover solutions are pending, process them
    if len(optimization_state.pending_solutions) > 0:
        if optimization_state.completed_evaluations >= optimization_state.total_evaluations:
            logger.warning(f"[GENERATION-END] Skipping leftover batch at gen {current_gen} — evaluations complete ({optimization_state.completed_evaluations}/{optimization_state.total_evaluations})")
            optimization_state.pending_solutions = []
            optimization_state.pending_cache_keys = []
        else:
            logger.warning(f"[GENERATION-END] Forcing evaluation of {len(optimization_state.pending_solutions)} leftover solutions at end of generation {current_gen}")

            try:
                # Set batch counters before processing
                optimization_state.current_batch_total = len(optimization_state.pending_solutions)
                optimization_state.current_batch_completed = 0
                optimal_workers = min(SOL_PER_POP, NUM_PROCESSES)
                optimization_state.active_solutions = min(optimal_workers, len(optimization_state.pending_solutions))
                optimization_state.display_needs_update = True

                start_time = time.time()
                results = []
                futures = []

                # Submit all solutions to the pool
                for solution in optimization_state.pending_solutions:
                    future = optimization_state.pool.apply_async(
                        evaluate_solution_wrapper,
                        ({"solution": solution}, None)
                    )
                    futures.append(future)

                # Collect results with regular display updates
                logger.info(f"Waiting for {len(futures)} solutions to complete")
                for i, future in enumerate(futures):
                    try:
                        optimization_state.display_needs_update = True
                        result = future.get()
                        results.append(result)
                        optimization_state.current_batch_completed += 1
                        optimization_state.generation_solutions_evaluated += 1
                        optimization_state.completed_evaluations += 1
                        optimization_state.active_solutions = max(0, len(futures) - (i + 1))
                        logger.info(f"Solution {i+1}/{len(futures)} completed, {optimization_state.active_solutions} workers still active")
                        optimization_state.display_needs_update = True
                    except Exception as e:
                        logger.error(f"Solution evaluation failed: {str(e)}")
                        results.append((float('-inf'), None))
                        optimization_state.current_batch_completed += 1
                        optimization_state.generation_solutions_evaluated += 1
                        optimization_state.completed_evaluations += 1
                        optimization_state.failed_evaluations += 1
                        optimization_state.active_solutions = max(0, len(futures) - (i + 1))
                        optimization_state.display_needs_update = True

                # Log results
                logger.info(f"[FINAL-BATCH-DEBUG] About to log final batch. pending_solutions={len(optimization_state.pending_solutions)}, results={len(results)}")
                for idx, result in enumerate(results):
                    logger.info(f"[FINAL-BATCH-DEBUG] Processing result idx={idx}: {result}")
                    float_genes = optimization_state.pending_solutions[idx]
                    cache_key = optimization_state.pending_cache_keys[idx]
                    logger.info(f"[FINAL-BATCH-DEBUG] float_genes={float_genes}, cache_key={cache_key}")

                    if result is None or len(result) < 5:
                        genes = {
                            "bm_mass": int(round(float_genes[0])) if idx < len(optimization_state.pending_solutions) else 0,
                            "num_modules": int(round(float_genes[2])) if idx < len(optimization_state.pending_solutions) else 0,
                            "start_arc_length": int(round(float_genes[1])) if idx < len(optimization_state.pending_solutions) else 0
                        }
                        metric_keys = ["max_tension", "min_bend_radius", "seabed_clearance", "surface_clearance"]
                        metrics_formatted = {k: float('nan') for k in metric_keys}
                        fitness = float('-inf')
                        is_valid = False
                        fail_reason = "no result"
                        solution_time = None
                    else:
                        if len(result) == 6:
                            fitness, metrics_formatted, genes, is_valid, fail_reason, solution_time = result
                        else:
                            fitness, metrics_formatted, genes, is_valid, fail_reason = result
                            solution_time = None
                        if metrics_formatted is not None:
                            if "seabed_clearance" in metrics_formatted:
                                metrics_formatted["min_seabed_clearance"] = metrics_formatted["seabed_clearance"]
                            if "surface_clearance" in metrics_formatted:
                                metrics_formatted["min_surface_clearance"] = metrics_formatted["surface_clearance"]

                    csv_logger.log_solution(
                        generation=current_gen,
                        genes=genes,
                        metrics=metrics_formatted,
                        fitness=round(fitness, 2) if isinstance(fitness, float) else fitness,
                        is_valid=is_valid,
                        fail_reason=fail_reason,
                        solution_time=solution_time
                    )

                    update_best_solution(fitness, float_genes, log_prefix=f"[Generation {current_gen}] ")
                    if not is_valid:
                        optimization_state.failed_evaluations += 1
                    if fail_reason == "timeout":
                        optimization_state.timed_out_evaluations += 1
                    optimization_state.generation_solutions_evaluated += 1

                logger.info(f"[FINAL-BATCH-DEBUG] Generation {current_gen}: Finished logging batch. Solutions logged: {len(results)}. Completed evals: {optimization_state.completed_evaluations}")

                optimization_state.pending_solutions = []
                optimization_state.pending_cache_keys = []

            except Exception as e:
                logger.error(f"Error processing remaining solutions: {str(e)}")
                optimization_state.active_solutions = 0
                optimization_state.current_batch_total = 0
                optimization_state.current_batch_completed = 0
                optimization_state.display_needs_update = True


def fitness_func_parallel(ga_instance, gene, solution_idx):
    """Fitness function that handles parallel evaluation of solutions."""
    try:
        # Log every call for debugging
        logger.info(f"[FITNESS-FUNC-CALL] Generation: {getattr(ga_instance, 'generations_completed', 'N/A')}, Solution Index: {solution_idx}, Gene: {gene}")
        # Start the display update thread if not already running
        if optimization_state.display_thread is None or not optimization_state.display_thread.is_alive():
            start_display_update_thread()
            logger.info("Started display update thread from fitness function")
        
        current_gen = ga_instance.generations_completed
        batch_size = optimization_state.generation_solutions_total
        optimal_workers = min(SOL_PER_POP, NUM_PROCESSES)
        
        # Handle both single genes and full solutions from PyGAD
        if isinstance(gene, (list, np.ndarray)) and len(gene) == 3:
            # Full solution provided
            solution = gene
        else:
            # Single gene provided - collect into solution buffer
            if solution_idx != optimization_state.current_solution_idx:
                optimization_state.current_solution = [0.0] * 3
                optimization_state.current_solution_idx = solution_idx
            
            # Convert gene to float scalar if it's a numpy array
            if isinstance(gene, np.ndarray):
                gene = float(gene.item())
            else:
                gene = float(gene)
            
            # Add gene to current solution buffer
            gene_idx = sum(1 for g in optimization_state.current_solution if g != 0.0)
            optimization_state.current_solution[gene_idx] = gene
            
            # If solution is not complete, return placeholder
            if any(g == 0.0 for g in optimization_state.current_solution):
                return 0.0
                
            # Solution is complete, use it
            solution = optimization_state.current_solution
        
        # Just append every solution to the batch
        optimization_state.pending_solutions.append(solution)
        # For compatibility, still append a dummy cache_key (not used for filtering)
        optimization_state.pending_cache_keys.append(None)
        
        # Process the batch when the number of solutions matches the expected batch size
        batch_size = optimization_state.generation_solutions_total
        if len(optimization_state.pending_solutions) < batch_size:
            # Not enough solutions yet, just return 0.0 as a placeholder
            return 0.0
        # Diagnostic: Log batch contents before processing
        logger.info(f"[BATCH-DEBUG] About to process batch: pending_solutions:")
        for idx, sol in enumerate(optimization_state.pending_solutions):
            int_cache_key = (
                int(round(sol[0])),
                int(round(sol[1])),
                int(round(sol[2]))
            )
            logger.info(f"[BATCH-DEBUG] idx={idx}, float_genes={sol}, int_cache_key={int_cache_key}")
        logger.info(f"[BATCH-DEBUG] pending_cache_keys: {optimization_state.pending_cache_keys}")
        # --- CHUNKED BATCHING LOGIC START ---
        actual_batch_size = min(batch_size, len(optimization_state.pending_solutions))
        optimization_state.current_batch_total = actual_batch_size
        optimization_state.current_batch_completed = 0
        optimal_workers = min(SOL_PER_POP, NUM_PROCESSES)
        optimization_state.active_solutions = min(optimal_workers, actual_batch_size)
        optimization_state.display_needs_update = True
        logger.info(f"Processing batch of {actual_batch_size} solutions for generation {current_gen} with {optimization_state.active_solutions} workers")
        results = [None] * actual_batch_size
        timeout = STATIC_ANALYSIS_TIMEOUT * MULTIPROCESS_TIMEOUT_MULTIPLIER
        # --- Force progress display update before starting the first chunk ---
        update_progress_bars(force_full_update=True)
        # --- End force progress display update ---
        for chunk_start in range(0, actual_batch_size, NUM_PROCESSES):
            chunk_end = min(chunk_start + NUM_PROCESSES, actual_batch_size)
            chunk_size = chunk_end - chunk_start
            logger.info(f"[BATCH-DEBUG] Generation {current_gen}: Starting chunk {chunk_start // NUM_PROCESSES + 1} (solutions {chunk_start}-{chunk_end - 1}), chunk_size={chunk_size}, num_workers={NUM_PROCESSES}")
            processes = []
            queues = []
            chunk_start_time = time.time()
            for i in range(chunk_start, chunk_end):
                solution = optimization_state.pending_solutions[i]
                cache_key = optimization_state.pending_cache_keys[i]
                if isinstance(solution, np.ndarray):
                    solution = solution.tolist()
                solution = copy.deepcopy(solution)
                queue = multiprocessing.Queue()
                logger.info(f"[BATCH-DEBUG] Generation {current_gen}: About to start process for idx={i}, solution={solution}")
                p = multiprocessing.Process(target=evaluate_solution_wrapper, args=({"solution": solution}, optimization_state.current_generation, queue))
                processes.append((p, queue))
                queues.append(queue)
                p.start()
                logger.info(f"[BATCH-DEBUG] Generation {current_gen}: Started process PID={p.pid} for idx={i}")
            # --- Start a background thread to update the display every 10 seconds while waiting for this chunk ---
            import threading
            import time as _time
            stop_display_update = False
            def periodic_display_update():
                while not stop_display_update:
                    # Only set display_needs_update, let display thread handle actual update
                    optimization_state.display_needs_update = True
                    _time.sleep(10)
            display_thread = threading.Thread(target=periodic_display_update)
            display_thread.daemon = True
            display_thread.start()
            # --- End background display thread setup ---
            # Track which processes have completed
            finished = [False] * chunk_size
            chunk_results = [None] * chunk_size
            start_time = time.time()
            timeout = STATIC_ANALYSIS_TIMEOUT * MULTIPROCESS_TIMEOUT_MULTIPLIER
            while not all(finished):
                for i, (p, queue) in enumerate(processes):
                    idx = chunk_start + i
                    if finished[i]:
                        continue
                    try:
                        # Wait for result from queue (with timeout)
                        result = queue.get(timeout=timeout)
                        logger.info(f"[BATCH-DEBUG] Generation {current_gen}: Received result from PID={p.pid} idx={idx}: {result}")
                        chunk_results[i] = result
                        results[idx] = result
                        finished[i] = True
                        # Per-process progress update
                        optimization_state.current_batch_completed += 1
                        optimization_state.display_needs_update = True
                        # Join the process after getting the result
                        p.join(timeout=1)
                        if p.is_alive():
                            logger.warning(f"[BATCH-DEBUG] Generation {current_gen}: Process PID={p.pid} idx={idx} still alive after join, terminating.")
                            p.terminate()
                            p.join(timeout=1)
                    except Exception as e:
                        logger.error(f"[BATCH-DEBUG] Generation {current_gen}: Exception while getting result from PID={p.pid} idx={idx}: {e}")
                        norm_genes = optimization_state.pending_solutions[idx]
                        bm_mass, start_arc_length, num_modules = denormalize_solution(norm_genes)
                        genes = {
                            "bm_mass": int(round(bm_mass)),
                            "num_modules": int(round(num_modules)),
                            "start_arc_length": int(round(start_arc_length))
                        }
                        metric_keys = ["max_tension", "min_bend_radius", "seabed_clearance", "surface_clearance"]
                        metrics_formatted = {k: float('nan') for k in metric_keys}
                        chunk_results[i] = (float('-inf'), metrics_formatted, genes, False, "timeout")
                        results[idx] = chunk_results[i]
                        finished[i] = True
                        # Per-process progress update
                        optimization_state.current_batch_completed += 1
                        optimization_state.display_needs_update = True
                        if p.is_alive():
                            logger.warning(f"[BATCH-DEBUG] Generation {current_gen}: Process PID={p.pid} idx={idx} still alive after failed queue.get, terminating.")
                            p.terminate()
                            p.join(timeout=1)
                    finally:
                        try:
                            p.join(timeout=2)
                            queue.close()
                        except Exception:
                            pass
                _time.sleep(0.1)
            chunk_elapsed = time.time() - chunk_start_time
            logger.info(f"[BATCH-DEBUG] Generation {current_gen}: Finished chunk {chunk_start // NUM_PROCESSES + 1} in {chunk_elapsed:.2f}s")
            # --- Stop the background display update thread for this chunk ---
            stop_display_update = True
            display_thread.join(timeout=1)
            # --- End background display thread ---
        # --- CHUNKED BATCHING LOGIC END ---
        logger.info(f"[BATCH-DEBUG] All results after collection: {results}")
        # --- Force progress display update after all chunks are done ---
        optimization_state.display_needs_update = True
        update_progress_bars(force_full_update=True)
        # --- End force progress display update ---
        logger.info(f"[BATCH-DEBUG] Generation {current_gen}: About to log main batch. pending_solutions={len(optimization_state.pending_solutions)}, results={len(results)}")
        for idx, result in enumerate(results):
            logger.info(f"[BATCH-DEBUG] Processing result idx={idx}: {result}")
            float_genes = optimization_state.pending_solutions[idx]
            cache_key = optimization_state.pending_cache_keys[idx]
            logger.info(f"[BATCH-DEBUG] float_genes={float_genes}, cache_key={cache_key}")
            if result is None or len(result) < 5:
                genes = {
                    "bm_mass": int(round(float_genes[0])) if idx < len(optimization_state.pending_solutions) else 0,
                    "num_modules": int(round(float_genes[2])) if idx < len(optimization_state.pending_solutions) else 0,
                    "start_arc_length": int(round(float_genes[1])) if idx < len(optimization_state.pending_solutions) else 0
                }
                metric_keys = ["max_tension", "min_bend_radius", "seabed_clearance", "surface_clearance"]
                metrics_formatted = {k: float('nan') for k in metric_keys}
                fitness = float('-inf')
                is_valid = False
                fail_reason = "no result"
                solution_time = None
            else:
                # Unpack solution_time from the result tuple (6th value)
                if len(result) == 6:
                    fitness, metrics_formatted, genes, is_valid, fail_reason, solution_time = result
                else:
                    fitness, metrics_formatted, genes, is_valid, fail_reason = result
                    solution_time = None
                # Map metrics to expected CSV columns
                if metrics_formatted is not None:
                    if "seabed_clearance" in metrics_formatted:
                        metrics_formatted["min_seabed_clearance"] = metrics_formatted["seabed_clearance"]
                    if "surface_clearance" in metrics_formatted:
                        metrics_formatted["min_surface_clearance"] = metrics_formatted["surface_clearance"]
                # Log to CSV
                csv_logger.log_solution(
                    generation=optimization_state.total_generations,
                    genes=genes,
                    metrics=metrics_formatted,
                    fitness=round(fitness, 2) if isinstance(fitness, float) else fitness,
                    is_valid=is_valid,
                    fail_reason=fail_reason,
                    solution_time=solution_time
                )
                
                # Update best solution if this one is better
                update_best_solution(fitness, optimization_state.pending_solutions[idx], log_prefix="[fitness_func_parallel] ")
                
                # Increment counters ONLY here
                optimization_state.completed_evaluations += 1
                if not is_valid:
                    optimization_state.failed_evaluations += 1
                if fail_reason == "timeout":
                    optimization_state.timed_out_evaluations += 1
                optimization_state.generation_solutions_evaluated += 1
        logger.info(f"[BATCH-DEBUG] Generation {current_gen}: Finished logging main batch. Solutions logged: {len(results)}. Completed evals: {optimization_state.completed_evaluations}")
        # Remove batch-level update_progress calls here to avoid double-counting
        logger.info(f"Batch processed: {actual_batch_size} solutions. Total evaluated: {optimization_state.completed_evaluations}/{optimization_state.total_evaluations}")
        logger.info(f"[BATCH] Clearing batch: pending_solutions, pending_cache_keys")
        optimization_state.pending_solutions = []
        optimization_state.pending_cache_keys = []
        return results[solution_idx][0]
    except Exception as e:
        logger.error(f"Error in fitness function: {str(e)}")
        logger.error(traceback.format_exc())
        optimization_state.active_solutions = max(0, optimization_state.active_solutions - 1)
        optimization_state.display_needs_update = True
        return float('-inf')

def show_fancy_terminal_results(ga_instance):
    import graphics
    import os
    import matplotlib.pyplot as plt
    import time

    colors = graphics.get_colors()
    display_width = 80
    graphics.clear_terminal()

    if optimization_state.display_thread is not None and optimization_state.display_thread.is_alive():
        optimization_state.stop_display_thread = True
        optimization_state.display_thread.join(timeout=2.0)

    if optimization_state.best_solution is not None:
        bm_mass, start_arc_length, num_modules = optimization_state.best_solution
        best_fitness = optimization_state.best_fitness
        graphics.print_header("OPTIMIZATION COMPLETE", display_width)
        graphics.print_single_floater()
        graphics.print_waterline(display_width)
        graphics.print_section_header("OPTIMIZATION RESULTS")
        print(f"  {colors['BOLD']}{colors['GREEN']}Best Solution Parameters:{colors['END']}")
        print(f"  {colors['BOLD']}Buoyancy Module Mass:{colors['END']} {colors['YELLOW']}{bm_mass:.1f} kg{colors['END']}")
        print(f"  {colors['BOLD']}Start Arc Length:{colors['END']} {colors['YELLOW']}{start_arc_length:.1f} m{colors['END']}")
        print(f"  {colors['BOLD']}Number of Modules:{colors['END']} {colors['YELLOW']}{int(num_modules)}{colors['END']}")
        print(f"  {colors['BOLD']}Fitness Score:{colors['END']} {colors['GREEN']}{best_fitness:.2f}{colors['END']}")

        best_solution_cache_key = (float(bm_mass), float(start_arc_length), int(num_modules))
        metrics = optimization_state.metrics_cache.get(best_solution_cache_key, {})
        if metrics:
            graphics.print_section_header("SOLUTION METRICS")
            print(f"  {colors['BOLD']}Seabed Clearance:{colors['END']} {colors['YELLOW']}{metrics.get('seabed_clearance', 0.0):.2f} m{colors['END']} {colors['GREEN']}(Min Required: {MIN_SEABED_CLEARANCE} m){colors['END']}")
            print(f"  {colors['BOLD']}Surface Clearance:{colors['END']} {colors['YELLOW']}{metrics.get('surface_clearance', 0.0):.2f} m{colors['END']} {colors['GREEN']}(Min Required: {MIN_SURFACE_CLEARANCE} m){colors['END']}")
            print(f"  {colors['BOLD']}Maximum Tension:{colors['END']} {colors['YELLOW']}{metrics.get('max_tension', 0.0):.2f} kN{colors['END']} {colors['GREEN']}(Max Allowed: {MAX_TENSION_LIMIT} kN){colors['END']}")
            print(f"  {colors['BOLD']}    Cable:{colors['END']} {colors['CYAN']}{metrics.get('tension_cable', 'None')}{colors['END']}")
            print(f"  {colors['BOLD']}Minimum Bend Radius:{colors['END']} {colors['YELLOW']}{metrics.get('min_bend_radius', float('inf')):.2f} m{colors['END']} {colors['GREEN']}(Min Required: {MIN_BEND_RADIUS_LIMIT} m){colors['END']}")
            print(f"  {colors['BOLD']}    Cable:{colors['END']} {colors['CYAN']}{metrics.get('bend_cable', 'None')}{colors['END']}")

        graphics.print_section_header("OPTIMIZATION STATISTICS")
        total_evaluations = optimization_state.total_evaluations
        completed_evals = min(optimization_state.completed_evaluations, total_evaluations)
        print(f"  {colors['BOLD']}Evaluations:{colors['END']} {colors['YELLOW']}{completed_evals}/{total_evaluations}{colors['END']}")
        print(f"  {colors['BOLD']}Timed Out Solutions:{colors['END']} {colors['YELLOW']}{optimization_state.timed_out_evaluations}{colors['END']}")
        print(f"  {colors['BOLD']}Total Generations:{colors['END']} {colors['YELLOW']}{optimization_state.total_generations}{colors['END']}")
        avg_time = optimization_state.get_avg_solution_time()
        print(f"  {colors['BOLD']}Average Solution Time:{colors['END']} {colors['YELLOW']}{avg_time:.2f} s{colors['END']}")
        elapsed = time.time() - optimization_state.start_time
        print(f"  {colors['BOLD']}Total Time:{colors['END']} {colors['YELLOW']}{graphics.format_time(elapsed)}{colors['END']}")

        output_dir = "output"
        graphics.print_section_header("OUTPUT FILES")
        print(f"  {colors['GREEN']}Files have been saved to: {colors['YELLOW']}{output_dir}{colors['END']}")
        created_files = []
        if os.path.exists(os.path.join(output_dir, "cable_profiles.png")):
            created_files.append("cable_profiles.png")
        if os.path.exists(os.path.join(output_dir, "optimization_report.txt")):
            created_files.append("optimization_report.txt")
        if os.path.exists("BestSolution.dat"):
            created_files.append("BestSolution.dat")
        for f in created_files:
            print(f"  {colors['BOLD']}Output:{colors['END']} {colors['YELLOW']}{f}{colors['END']}")
        if not created_files:
            print(f"  {colors['RED']}No output files were created.{colors['END']}")

        try:
            fig = ga_instance.plot_fitness()
            plt.tight_layout()
            fitness_plot_path = os.path.join(output_dir, "pygad_fitness.png")
            plt.savefig(fitness_plot_path)
            plt.close(fig)
            print(f"  {colors['BOLD']}Fitness History Plot:{colors['END']} {colors['YELLOW']}{fitness_plot_path}{colors['END']}")
        except Exception as e:
            print(f"{colors['RED']}Error generating fitness plot: {e}{colors['END']}")
    else:
        graphics.print_header("OPTIMIZATION COMPLETE", display_width)
        graphics.print_single_floater()
        graphics.print_waterline(display_width)
        graphics.print_section_header("OPTIMIZATION RESULTS")
        print(f"\n{colors['RED']}No valid solution was found during optimization.{colors['END']}")
        print(f"{colors['YELLOW']}Try relaxing constraints or increasing search space.{colors['END']}")
        graphics.print_section_header("CONSTRAINT INFORMATION")
        print(f"  {colors['BOLD']}Min Seabed Clearance:{colors['END']} {colors['YELLOW']}{MIN_SEABED_CLEARANCE} m{colors['END']}")
        print(f"  {colors['BOLD']}Min Surface Clearance:{colors['END']} {colors['YELLOW']}{MIN_SURFACE_CLEARANCE} m{colors['END']}")
        print(f"  {colors['BOLD']}Max Tension Limit:{colors['END']} {colors['YELLOW']}{MAX_TENSION_LIMIT} kN{colors['END']}")
        print(f"  {colors['BOLD']}Min Bend Radius:{colors['END']} {colors['YELLOW']}{MIN_BEND_RADIUS_LIMIT} m{colors['END']}")
        graphics.print_section_header("OPTIMIZATION STATISTICS")
        total_evaluations = optimization_state.total_evaluations
        completed_evals = min(optimization_state.completed_evaluations, total_evaluations)
        print(f"  {colors['BOLD']}Evaluations:{colors['END']} {colors['YELLOW']}{completed_evals}/{total_evaluations}{colors['END']}")
        print(f"  {colors['BOLD']}Timed Out Solutions:{colors['END']} {colors['YELLOW']}{optimization_state.timed_out_evaluations}{colors['END']}")
        print(f"  {colors['BOLD']}Total Generations:{colors['END']} {colors['YELLOW']}{optimization_state.total_generations}{colors['END']}")
        avg_time = optimization_state.get_avg_solution_time()
        print(f"  {colors['BOLD']}Average Solution Time:{colors['END']} {colors['YELLOW']}{avg_time:.2f} s{colors['END']}")
        elapsed = time.time() - optimization_state.start_time
        print(f"  {colors['BOLD']}Total Time:{colors['END']} {colors['YELLOW']}{graphics.format_time(elapsed)}{colors['END']}")
        graphics.print_seabed(display_width)
        graphics.print_footer(display_width)

def run_optimization(config_overrides=None):
    """
    Run genetic algorithm optimization with parallelization
    
    Args:
        config_overrides (dict, optional): Override default configuration
    
    Returns:
        tuple: (best_solution, best_fitness)
    """
    csv_logger.init_csv_logging()
    ga_instance = None  # Ensure it's available in `finally` even if something fails early
    
    # Update config from overrides if provided
    if config_overrides:
        logger.info("Using configuration overrides:")
        for key, value in config_overrides.items():
            if key in globals():
                old_value = globals()[key]
                globals()[key] = value
                logger.info(f"  {key}: {old_value} -> {value}")
            else:
                logger.warning(f"  Unknown configuration parameter: {key}")
    
    # Configure logging to file
    init_logging()
    
    # Variables to store best solution details for access in finally block
    final_solution = None
    final_fitness = float('-inf')
    final_metrics = None
    best_model_saved = False
    ga_instance = None  # Store GA instance for later access in finally block
    
    # Set up parameter bounds from config
    buoyancy_module_mass_bounds = PARAM_BOUNDS["bm_mass"]
    start_arc_length_bounds = PARAM_BOUNDS["start_arc_length"]
    num_modules_bounds = PARAM_BOUNDS["num_modules"]
    
    try:
        try:
            # Main optimization logic
            # Use all available workers regardless of population size to improve batching
            # This allows us to process solutions efficiently even with small population sizes
            optimal_workers = NUM_PROCESSES
            logger.info(f"Starting optimization process with {optimal_workers} workers")
            # Clear terminal at the start of optimization
            clear_terminal()
            # Calculate total evaluations:
            # Generation 0: SOL_PER_POP solutions
            # Subsequent generations: (SOL_PER_POP - KEEP_PARENTS) solutions each
            total_evaluations = SOL_PER_POP + (NUM_GENERATIONS - 1) * (SOL_PER_POP - KEEP_PARENTS)

            # Set up optimization state
            global optimization_state
            optimization_state.total_evaluations = total_evaluations
            optimization_state.total_generations = NUM_GENERATIONS
            optimization_state.current_generation = 0
            optimization_state.generation_solutions_evaluated = 0
            optimization_state.generation_solutions_total = SOL_PER_POP
            optimization_state.completed_evaluations = 0  # Always start with 0 evaluated solutions
            optimization_state.failed_evaluations = 0  # Add counter for failed evaluations
            optimization_state.invalid_evaluations = 0  # Add counter for invalid (constraint-violating) evaluations
            optimization_state.active_solutions = 0
            optimization_state.current_batch_completed = 0
            optimization_state.current_batch_total = 0
            optimization_state.start_time = time.time()
            optimization_state.logger = logger
            optimization_state.terminal_output_enabled = True
            optimization_state.solutions_cache = {}
            optimization_state.metrics_cache = {}
            optimization_state.solution_times = []
            optimization_state.pending_solutions = []
            optimization_state.pending_indices = []
            optimization_state.pending_cache_keys = []
            optimization_state.best_fitness = float('-inf')
            optimization_state.best_solution = None
            # Main optimization loop
            ga_instance = pygad.GA(
                num_generations=NUM_GENERATIONS,
                num_parents_mating=NUM_PARENTS_MATING,
                sol_per_pop=SOL_PER_POP,
                fitness_func=fitness_func_parallel,
                parent_selection_type=PARENT_SELECTION_TYPE,
                keep_parents=KEEP_PARENTS,
                crossover_type=CROSSOVER_TYPE,
                crossover_probability=CROSSOVER_PROBABILITY,
                mutation_type=MUTATION_TYPE,
                mutation_percent_genes=MUTATION_PERCENT_GENES,
                mutation_probability=MUTATION_PROBABILITY,
                mutation_by_replacement=MUTATION_BY_REPLACEMENT,
                gene_type=GENE_TYPE,
                init_range_low=INIT_RANGE_LOW,
                init_range_high=INIT_RANGE_HIGH,
                allow_duplicate_genes=ALLOW_DUPLICATE_GENES,
                stop_criteria=STOP_CRITERIA,
                num_genes=3,
                gene_space=[
                    {'low': 0.0, 'high': 1.0},
                    {'low': 0.0, 'high': 1.0},
                    {'low': 0.0, 'high': 1.0}
                ],
                on_generation=on_generation,
            )
            ga_instance.run()
            # --- Ensure all pending solutions are processed and display thread is stopped before showing results ---
            # Stop the display update thread
            if optimization_state.display_thread is not None and optimization_state.display_thread.is_alive():
                optimization_state.stop_display_thread = True
                optimization_state.display_thread.join(timeout=2.0)
            # Process any remaining pending solutions (final batch)
            if len(optimization_state.pending_solutions) > 0:
                if optimization_state.completed_evaluations >= optimization_state.total_evaluations:
                    logger.warning(f"[FINAL-BATCH-DEBUG] Skipping {len(optimization_state.pending_solutions)} leftover pending solutions because completed_evaluations ({optimization_state.completed_evaluations}) >= total_evaluations ({optimization_state.total_evaluations})")
                    optimization_state.pending_solutions = []
                    optimization_state.pending_cache_keys = []
                else:
                    logger.info(f"Processing final batch of {len(optimization_state.pending_solutions)} pending solutions after GA run.")
                    batch_size = min(len(optimization_state.pending_solutions), optimization_state.total_evaluations - optimization_state.completed_evaluations)
                    results = [None] * batch_size
                    processes = []
                    queues = []
                    timeout = STATIC_ANALYSIS_TIMEOUT * MULTIPROCESS_TIMEOUT_MULTIPLIER
                    for i in range(batch_size):
                        solution = optimization_state.pending_solutions[i]
                        cache_key = optimization_state.pending_cache_keys[i]
                        if isinstance(solution, np.ndarray):
                            solution = solution.tolist()
                        solution = copy.deepcopy(solution)
                        queue = multiprocessing.Queue()
                        p = multiprocessing.Process(target=evaluate_solution_wrapper, args=({"solution": solution}, optimization_state.current_generation, queue))
                        processes.append((p, queue))
                        p.start()
                        queues.append(queue)
                    for i, (p, queue) in enumerate(processes):
                        p.join(timeout)
                        try:
                            result = queue.get(timeout=1)
                            results[i] = result
                            queue.close()
                        except Exception as e:
                            norm_genes = optimization_state.pending_solutions[i]
                            bm_mass, start_arc_length, num_modules = denormalize_solution(norm_genes)
                            genes = {
                                "bm_mass": int(round(bm_mass)),
                                "num_modules": int(round(num_modules)),
                                "start_arc_length": int(round(start_arc_length))
                            }
                            metric_keys = ["max_tension", "min_bend_radius", "seabed_clearance", "surface_clearance"]
                            metrics_formatted = {k: float('nan') for k in metric_keys}
                            results[i] = (float('-inf'), metrics_formatted, genes, False, "timeout")
                            try:
                                queue.close()
                            except Exception:
                                pass
                    logger.info(f"[FINAL-BATCH-DEBUG] About to log final batch. pending_solutions={batch_size}, results={len(results)}")
                    for idx, result in enumerate(results):
                        float_genes = optimization_state.pending_solutions[idx]
                        cache_key = optimization_state.pending_cache_keys[idx]
                        if result is None or len(result) < 5:
                            genes = {
                                "bm_mass": int(round(float_genes[0])) if idx < len(optimization_state.pending_solutions) else 0,
                                "num_modules": int(round(float_genes[2])) if idx < len(optimization_state.pending_solutions) else 0,
                                "start_arc_length": int(round(float_genes[1])) if idx < len(optimization_state.pending_solutions) else 0
                            }
                            metric_keys = ["max_tension", "min_bend_radius", "seabed_clearance", "surface_clearance"]
                            metrics_formatted = {k: float('nan') for k in metric_keys}
                            fitness = float('-inf')
                            is_valid = False
                            fail_reason = "no result"
                            solution_time = None
                        else:
                            fitness, metrics_formatted, genes, is_valid, fail_reason, solution_time = result
                            # Map metrics to expected CSV columns
                            if metrics_formatted is not None:
                                if "seabed_clearance" in metrics_formatted:
                                    metrics_formatted["min_seabed_clearance"] = metrics_formatted["seabed_clearance"]
                                if "surface_clearance" in metrics_formatted:
                                    metrics_formatted["min_surface_clearance"] = metrics_formatted["surface_clearance"]
                        # Log to CSV
                        csv_logger.log_solution(
                            generation=optimization_state.total_generations,
                            genes=genes,
                            metrics=metrics_formatted,
                            fitness=round(fitness, 2) if isinstance(fitness, float) else fitness,
                            is_valid=is_valid,
                            fail_reason=fail_reason,
                            solution_time=solution_time
                        )
                        
                        # Update best solution if this one is better
                        update_best_solution(fitness, optimization_state.pending_solutions[idx], log_prefix="[run_optimization] ")
                        
                        # Increment counters ONLY here
                        optimization_state.completed_evaluations += 1
                        if not is_valid:
                            optimization_state.failed_evaluations += 1
                        if fail_reason == "timeout":
                            optimization_state.timed_out_evaluations += 1
                        optimization_state.generation_solutions_evaluated += 1
                    logger.info(f"[FINAL-BATCH-DEBUG] Finished logging final batch. Solutions logged: {len(results)}. Completed evals: {optimization_state.completed_evaluations}")
                # Clear pending solutions regardless
                optimization_state.pending_solutions = []
                optimization_state.pending_cache_keys = []
            
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Cleaning up...")
            if optimization_state.pool:
                try:
                    optimization_state.pool.terminate()
                    optimization_state.pool.join(timeout=2.0)
                except Exception:
                    pass
                optimization_state.pool = None
            cleanup()
            print("All worker processes terminated. Exiting.")
            return None, float('-inf')
        # ... rest of your code ...
        # (existing post-optimization reporting and cleanup logic)
    finally:
        try:
            df = pd.read_csv('output/solutions.csv')
            logger.info(f"[FINAL-VERIFY] CSV rows: {len(df)}, completed_evaluations counter: {optimization_state.completed_evaluations}")
        except Exception as e:
            logger.error(f"[FINAL-VERIFY] Could not read CSV for row count: {e}")

        try:
            show_fancy_terminal_results(ga_instance)
        except Exception as e:
            logger.warning(f"[UI-FALLBACK] Could not display final UI: {e}")

        if optimization_state.best_solution is not None:
            return optimization_state.best_solution, optimization_state.best_fitness
        else:
            return None, float('-inf')


# --- Display Thread Diagnostics ---
def _display_updater():
    """Thread function to update the display regularly"""
    try:
        import os
        import platform
        supports_ansi = True
        system = platform.system()
        if system == "Windows":
            has_wt_session = os.environ.get('WT_SESSION') is not None
            has_term_env = os.environ.get('TERM') is not None
            is_cmd = os.environ.get('PROMPT') is not None and 'cmd.exe' in os.environ.get('COMSPEC', '')
            is_powershell = 'powershell' in os.environ.get('PSModulePath', '').lower() or 'pwsh' in os.environ.get('PSModulePath', '').lower()
            supports_ansi = has_wt_session or has_term_env or is_powershell
            if optimization_state.logger:
                optimization_state.logger.info(f"Terminal detection - Windows Terminal: {has_wt_session}, TERM: {has_term_env}, PowerShell: {is_powershell}")
        if optimization_state.logger:
            optimization_state.logger.info(f"Terminal ANSI support detected: {supports_ansi}")
            optimization_state.logger.info(f"Platform: {system}")
        if system == "Windows":
            supports_ansi = True
            if optimization_state.logger:
                optimization_state.logger.info("Forcing ANSI support on Windows (overriding detection)")
        if not supports_ansi:
            optimization_state.terminal_output_enabled = False
            if optimization_state.logger:
                optimization_state.logger.warning("Disabling terminal-based progress display due to limited ANSI support")
            print("Terminal-based progress display disabled due to limited ANSI support.")
            print("Progress will be logged to the log file instead.")
            return
        # Do one full display update at the start
        update_progress_bars(force_full_update=True)
        last_full_update = time.time()
        # Main update loop
        while not optimization_state.stop_display_thread:
            try:
                current_time = time.time()
                # Always update every 10 seconds, regardless of display_needs_update
                if (current_time - last_full_update > 10):
                    logger.info(f"[DISPLAY-DEBUG] Triggering display update: periodic 10s interval")
                    update_progress_bars(force_full_update=True)
                    last_full_update = current_time
                    optimization_state.display_needs_update = False
                elif optimization_state.display_needs_update:
                    logger.info(f"[DISPLAY-DEBUG] Triggering display update: forced")
                    update_progress_bars(force_full_update=True)
                    last_full_update = current_time
                    optimization_state.display_needs_update = False
                time.sleep(1.0)
            except Exception as e:
                if optimization_state.logger:
                    optimization_state.logger.error(f"Error in display updater: {str(e)}")
                time.sleep(2.0)
    except Exception as e:
        if optimization_state.logger:
            optimization_state.logger.error(f"Display updater thread failed to initialize: {str(e)}")

def update_best_solution(fitness, solution, log_prefix=""):
    """
    Update and save the best solution if the given fitness is better than the current best.
    
    Args:
        fitness (float): The fitness value of the solution
        solution (list or numpy.ndarray): The normalized solution genes
        log_prefix (str, optional): Prefix for log messages
    
    Returns:
        bool: True if this was a new best solution, False otherwise
    """
    if fitness > optimization_state.best_fitness:
        optimization_state.best_fitness = fitness
        
        # Get the original solution values
        if isinstance(solution, (list, np.ndarray)):
            bm_mass, start_arc_length, num_modules = denormalize_solution(solution)
            optimization_state.best_solution = (bm_mass, start_arc_length, num_modules)
            
            # Log the new best solution
            logger.info(f"{log_prefix}New best solution found: BM Mass={bm_mass}kg, Start Arc={start_arc_length}m, Modules={num_modules}")
            logger.info(f"{log_prefix}New best fitness: {fitness}")
            
            # Save the best solution model
            try:
                best_solution_dict = {
                    "bm_mass": bm_mass,
                    "start_arc_length": start_arc_length,
                    "num_modules": int(num_modules)
                }
                save_best_solution(best_solution_dict)
                logger.info(f"{log_prefix}Best solution model saved successfully")
                return True
            except Exception as e:
                logger.error(f"{log_prefix}Error saving best solution model: {e}")
                logger.error(traceback.format_exc())
        
    return False
