# OrcaFlex Genetic Optimisation Tool

This repository contains a Python-based optimisation tool for lazy-wave cable configurations in floating offshore wind. The tool combines a genetic algorithm (GA) with automated static simulations in OrcaFlex to evaluate buoyancy module placement under multiple offset and lifecycle conditions. It was developed to support engineering exploration of cable feasibility in a simplified but configurable environment.

---

## Features

- Static evaluation of dynamic power cables in OrcaFlex using OrcFxAPI
- Genetic algorithm-based optimisation of buoyancy module mass, count, and position
- Parallel simulation using Python multiprocessing
- Terminal-based user interface for runtime control and parameter configuration
- Caching and timeout handling for efficient batch execution
- CSV logging and advanced plotting for post-processing

---

## Requirements

- Python 3.8 or newer
- OrcaFlex 11.0 or newer with licensed OrcFxAPI
- Python packages: numpy, pandas, matplotlib, seaborn, scipy, pygad, tqdm

To install the required Python packages:
1. Open the folder containing this repository
2. Hold Shift and right-click inside the folder, then choose "Open PowerShell window here" or "Open in Terminal"
3. Type this command and press Enter:
```shell
pip install numpy pandas matplotlib seaborn scipy pygad tqdm
```

---

## Quickstart: Terminal Interface

This is the easiest and recommended way to use the tool:

1. Make sure you have Python and OrcaFlex installed and working
2. Download or clone this repository to your computer
3. Open the folder containing the files
4. Hold Shift and right-click inside the folder, then select "Open PowerShell window here" or "Open in Terminal"
5. Run this command:
```shell
python terminal_ui.py
```

This will open a simple menu where you can:
- Change how the optimisation works (number of generations, limits, weights, etc.)
- Start the optimisation
- Try a single configuration
- View results and generate plots

All settings are saved automatically in `config.py`.

---

## Alternative Command-Line Usage

These are additional commands for advanced users:

- Run full optimisation with multiprocessing:
  ```shell
  python main.py --optimize --multi
  ```

- Evaluate a single configuration:
  ```shell
  python main.py --single-analysis --mass 500 --start 100 --num 12
  ```

- Generate plots only:
  ```shell
  python main.py --plot-only
  ```

---

## File Structure

- `main.py` – Execution entry point
- `terminal_ui.py` – User interface for configuration and control
- `config.py` – All GA and constraint settings
- `orcaflex_utils.py` – API logic for model manipulation and statics
- `multiprocess_optimization_module.py` – Custom GA implementation with multiprocessing
- `optimization_utils.py` – Caching, denormalisation, and helper routines
- `scatterPlots.py`, `results_processing.py` – Post-processing and visualisation
- `output/` – CSV logs, OrcaFlex files, and static plots

---

## Output Overview

- `output/solutions.csv` – All candidates and metrics
- `output/solutions_filtered.csv` – Constraint-satisfying configurations
- `output/cable_profiles.png` – Static shapes of best solution
- `output/pygad_fitness.png` – GA convergence tracking
- `output/BestSolution.dat` – OrcaFlex file for top-ranked configuration
- `plots/` – Scatter and density plots

---

## Notes and Limitations

- Full functionality for filtering optimisation results and generating detailed plots requires manual execution of the `results_processing.py` and `scatterPlots.py` scripts.


- Tool is intended for design-stage exploration and static-only analysis
- This repository is a prototype developed for research purposes and should not be regarded as a complete or fully developed engineering tool.
- Fitness structure and constraint values are not design-certified and should be tailored to the application
- OrcaFlex license is required and not included

---

## License

MIT License. See `LICENSE` for terms.

---

## Citation

This tool was developed for academic research and may be referenced in the context of simulation-based design optimisation of offshore power cables. Please cite the associated bachelor thesis if used in published work. A link to the full thesis will be provided here when (and if) it becomes publicly available.
