import os
import threading
import config
import pandas as pd
import logging
from orcaflex_utils import get_total_net_buoyancy, get_total_net_buoyancy_sol, get_total_net_buoyancy_eol
from config import BM_EoL_MASS_FACTOR, SEAWATER_DENSITY

OUTPUT_DIR = "output"
CSV_FILE = os.path.join(OUTPUT_DIR, "solutions.csv")
CSV_HEADER = [
    "solution_id", "bm_mass", "num_modules", "start_arc_length",
    "total_buoyancy_SoL", "total_buoyancy_EoL",
    "max_tension", "min_bend_radius", "min_seabed_clearance", "min_surface_clearance", "fitness", "is_valid", "fail_reason", "solution_time"
]

_csv_initialized = False

def get_solution_id(bm_mass, num_modules, start_arc_length):
    """Generate a unique ID for a solution based on its gene values."""
    return f"{bm_mass}_{num_modules}_{start_arc_length}"

def init_csv_logging():
    """
    Ensure the output directory exists and the CSV file has headers if it doesn't exist.
    If CSV_APPEND_MODE is False, the file will be reset (overwritten) at the start of each run.
    """
    global _csv_initialized
    if _csv_initialized:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not config.CSV_APPEND_MODE or not os.path.isfile(CSV_FILE):
        df = pd.DataFrame(columns=CSV_HEADER)
        df.to_csv(CSV_FILE, index=False)
    _csv_initialized = True

def log_solution(generation, genes, metrics, fitness, is_valid=True, fail_reason="", solution_time=None):
    """
    Log a solution (valid or invalid) to the CSV file using pandas.
    Args:
        generation (int): Generation number
        genes (dict): Dictionary with keys 'bm_mass', 'num_modules', 'start_arc_length'
        metrics (dict): Metrics dict with keys: max_tension, min_bend_radius, min_seabed_clearance, min_surface_clearance
        fitness (float): Fitness value
        is_valid (bool): True if solution is valid, False if constraint-violating
        fail_reason (str): Short string explaining why the solution is invalid
        solution_time (float, optional): Time taken to evaluate the solution in seconds
    """
    init_csv_logging()
    bm_mass = genes.get("bm_mass")
    num_modules = genes.get("num_modules")
    start_arc_length = genes.get("start_arc_length")
    total_buoyancy_SoL = None
    total_buoyancy_EoL = None
    if bm_mass is not None and num_modules is not None:
        total_buoyancy_SoL = get_total_net_buoyancy(bm_mass, num_modules, SEAWATER_DENSITY)
        eol_mass = bm_mass * BM_EoL_MASS_FACTOR
        total_buoyancy_EoL = get_total_net_buoyancy(eol_mass, num_modules, SEAWATER_DENSITY)
        total_buoyancy_SoL = round(total_buoyancy_SoL, 3)
        total_buoyancy_EoL = round(total_buoyancy_EoL, 3)
    # Only log solution_time if not a timeout
    if fail_reason == "timeout":
        solution_time_csv = ""
    else:
        solution_time_csv = round(solution_time, 1) if solution_time is not None else ""
    solution_id = get_solution_id(bm_mass, num_modules, start_arc_length)
    logging.info(f"[CSV_LOGGER] Logging solution: {solution_id} (generation {generation})")
    def fmt(x):
        return round(x, 3) if isinstance(x, float) else x
    row = {
        "solution_id": solution_id,
        "bm_mass": bm_mass,
        "num_modules": num_modules,
        "start_arc_length": start_arc_length,
        "total_buoyancy_SoL": total_buoyancy_SoL,
        "total_buoyancy_EoL": total_buoyancy_EoL,
        "max_tension": fmt(metrics.get("max_tension")),
        "min_bend_radius": fmt(metrics.get("min_bend_radius")),
        "min_seabed_clearance": fmt(metrics.get("min_seabed_clearance")),
        "min_surface_clearance": fmt(metrics.get("min_surface_clearance")),
        "fitness": fitness,
        "is_valid": is_valid,
        "fail_reason": fail_reason,
        "solution_time": solution_time_csv
    }
    df = pd.DataFrame([row])
    df.to_csv(CSV_FILE, mode="a", header=False, index=False) 