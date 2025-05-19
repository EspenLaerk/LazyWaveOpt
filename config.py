"""
Configuration for OrcaFlex buoyancy module configuration
"""

PLOT_SHOW_METRICS_BOX = False
PLOT_SHOW_TITLE = False
PLOT_SHOW_CABLE_LEGEND = False

# Model file path
MODEL_PATH = "LazyWave.dat"

# Cable names configuration
CABLE_NAMES = [
    "SoL-near", 
    "SoL", 
    "SoL-far",
    "EoL-near", 
    "EoL", 
    "EoL-far"
]

# Buoyancy module names
BM_NAMES = {
    "SoL": "BM-SoL",    # Buoyancy module for Start of Life cables
    "EoL": "BM-EoL"     # Buoyancy module for End of Life cables
}

# EoL factor for mass
BM_EoL_MASS_FACTOR = 1.05        # EoL mass is 5% more than SoL mass

# Default working configuration for UI plotting
DEFAULT_WORKING_BM_MASS = 489.0
DEFAULT_WORKING_START_ARC_LENGTH = 79.0
DEFAULT_WORKING_NUM_MODULES = 9

# ======================================================
# OPTIMIZATION CONFIGURATION
# ======================================================

# Parameter Bounds for Optimization
# These define the search space for each parameter
PARAM_BOUNDS = {
    "bm_mass": [300.0, 800.0],
    "start_arc_length": [10.0, 200.0],
    "num_modules": [1.0, 20.0],
}

# ------------------------------------------------------
# TIER 1: Safety Constraints (Hard Constraints)
# Any solution that violates these constraints is considered invalid
# ------------------------------------------------------
MIN_SEABED_CLEARANCE = 15.0
MIN_SURFACE_CLEARANCE = 15.0
MAX_TENSION_LIMIT = 1000.0         # Maximum acceptable cable tension in kN
MIN_BEND_RADIUS_LIMIT = 3.5      # Minimum acceptable bend radius in meters

# ------------------------------------------------------
# TIER 2: Cost Optimization Priority
# After meeting all safety constraints, minimize cost
# ------------------------------------------------------
# Base score is calculated as: 1000 - (num_modules * MODULE_COST_FACTOR)
MODULE_COST_FACTOR = 100         # Higher values prioritize fewer modules more strongly

# ------------------------------------------------------
# TIER 3: Performance Optimization Priority
# For solutions with the same module count, optimize performance
# ------------------------------------------------------
# Performance weights determine the relative importance of each metric
# when comparing solutions with the same number of modules
PERFORMANCE_WEIGHTS = {
    "seabed_clearance": 1.0,     # Higher values prioritize greater seabed clearance
    "surface_clearance": 0.5,    # Higher values prioritize greater surface clearance
    "max_tension": -0.1,         # Negative values prioritize lower tension
    "min_bend_radius": 0.5       # Higher values prioritize greater bend radius
}

# ------------------------------------------------------
# Fitness Weightings (User-configurable)
# ------------------------------------------------------
# These weights determine the relative importance of each fitness component (must sum to 1.0 for best results)
SAFETY_WEIGHT = 0.5         # Weight for safety (constraint satisfaction)
COST_WEIGHT = 0.3           # Weight for cost (module count/cost)
OPTIMIZATION_WEIGHT = 0.2   # Weight for performance optimization

# ------------------------------------------------------
# Genetic Algorithm Parameters
# ------------------------------------------------------
NUM_GENERATIONS = 100
NUM_PARENTS_MATING = 40
SOL_PER_POP = 100
PARENT_SELECTION_TYPE = "tournament"
KEEP_PARENTS = 10
CROSSOVER_TYPE = "uniform"
CROSSOVER_PROBABILITY = 0.9
MUTATION_TYPE = "random"
MUTATION_PERCENT_GENES = 33
MUTATION_PROBABILITY = 0.2
MUTATION_BY_REPLACEMENT = True
GENE_TYPE = float
INIT_RANGE_LOW = 0.0
INIT_RANGE_HIGH = 1.0
ALLOW_DUPLICATE_GENES = False
STOP_CRITERIA = None


# ------------------------------------------------------
# Execution Configuration
# ------------------------------------------------------

# Multiprocessing configuration
USE_MULTIPROCESSING = True       # Whether to use multiprocessing for evaluations
NUM_PROCESSES = 14

# Cache configuration
ENABLE_SOLUTION_CACHE = True     # Whether to cache solution evaluations
MAX_CACHE_SIZE = 1000           # Maximum number of solutions to keep in cache

# Progress tracking
PROGRESS_UPDATE_INTERVAL = 1.0   # Seconds between progress updates

# Static analysis timeout (seconds)
STATIC_ANALYSIS_TIMEOUT = 10.0
# Multiplier for multiprocess static analysis timeout (e.g., 2.0 means 2x the base timeout)
MULTIPROCESS_TIMEOUT_MULTIPLIER = 3.4

# ------------------------------------------------------
# CSV Logging Configuration
# ------------------------------------------------------
# If False, the solutions.csv file will be reset (overwritten) at the start of each optimization run.
# If True, new results will be appended to the existing file.
CSV_APPEND_MODE = False

# Seawater density (kg/m^3), used for net buoyancy calculations
SEAWATER_DENSITY = 1025.0

# Buoyancy and fluid properties
BM_VOLUME = 1.0
BM_SPACING = 5.0