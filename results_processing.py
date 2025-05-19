# IMPORTANT: This script must NEVER modify or overwrite 'solutions.csv'.
# All processing and output must use separate files (e.g., 'solutions_filtered.csv').
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
import matplotlib.ticker as ticker

# --- Configuration ---
# These should match the config used during optimization
POPULATION_SIZE = config.SOL_PER_POP
GENERATIONS = config.NUM_GENERATIONS
KEEP_PARENTS = config.KEEP_PARENTS

# Derived values
NUM_POPULATIONS = GENERATIONS + 1
EVALS_PER_GEN = [POPULATION_SIZE] + [POPULATION_SIZE - KEEP_PARENTS] * GENERATIONS
TOTAL_EVALS = sum(EVALS_PER_GEN)

SOLUTIONS_CSV = os.path.join('output', 'solutions.csv')
FITNESS_HISTORY_CSV = os.path.join('output', 'fitnessHistory.csv')
FITNESS_PLOT_PATH = os.path.join('output', 'fitness_history_plot.png')


def map_evaluations_to_generations(num_evals, evals_per_gen):
    """
    Map each evaluation index to its generation number.
    Returns a list of generation numbers for each evaluation.
    """
    gen_map = []
    idx = 0
    for gen, n in enumerate(evals_per_gen):
        for _ in range(n):
            if idx >= num_evals:
                break
            gen_map.append(gen)
            idx += 1
    return gen_map


def extract_fitness_history(solutions_csv, fitness_history_csv, evals_per_gen):
    """
    Extract the best fitness per generation from the solutions CSV and write to fitnessHistory.csv.
    Ensures every generation from 0 to GENERATIONS is included, with cumulative best fitness.
    """
    # Read the solutions CSV
    df = pd.read_csv(solutions_csv)
    num_evals = len(df)
    # Map each row to its generation
    gen_map = map_evaluations_to_generations(num_evals, evals_per_gen)
    df['generation'] = gen_map
    # Prepare cumulative best fitness for each generation
    generations = list(range(GENERATIONS + 1))
    best_so_far = float('-inf')
    fitness_history = []
    for gen in generations:
        gen_df = df[df['generation'] == gen]
        valid_fitness = gen_df['fitness'][gen_df['fitness'] != float('-inf')]
        if not valid_fitness.empty:
            best_in_gen = valid_fitness.max()
            if best_in_gen > best_so_far:
                best_so_far = best_in_gen
        # Append the best so far (could be -inf if no valid fitness yet)
        fitness_history.append({'generation': gen, 'best_fitness': best_so_far})
    # Write to CSV
    hist_df = pd.DataFrame(fitness_history)
    hist_df.to_csv(fitness_history_csv, index=False)
    print(f"Wrote fitness history to {fitness_history_csv}")


def plot_fitness_history_cumulative(fitness_history_csv, save_path=None):
    """
    Plot fitness vs. generations, where Y is the best fitness found up to and including each generation (cumulative best).
    """
    df = pd.read_csv(fitness_history_csv)
    # Compute cumulative best fitness (max so far)
    cum_best = df['best_fitness'].cummax()
    fig, ax = plt.subplots(figsize=(8, 5))
    # Vibrant green line, no markers, thicker line
    ax.plot(df['generation'], cum_best, linestyle='-', color='#00C853', linewidth=2.5)
    # Set X axis ticks in steps of 10 and Y axis ticks in steps of 50
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cumulative Best Fitness')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved fitness history plot to {save_path}")
    else:
        plt.show()


def filter_solutions(input_csv, output_csv):
    """
    Filter the solutions CSV to remove duplicates and rows where fail_reason is 'timeout'.
    Duplicates are removed based on bm_mass, num_modules, start_arc_length.
    Appends a 'generation' column to the output CSV, using the mapping from the original solutions.csv.
    """
    df = pd.read_csv(input_csv)
    # Assign generation number to each row based on original order
    num_evals = len(df)
    gen_map = map_evaluations_to_generations(num_evals, EVALS_PER_GEN)
    df['generation'] = gen_map
    # Remove rows where fail_reason is 'timeout' (string match, ignore case and NaN)
    filtered = df[~df['fail_reason'].fillna('').str.lower().eq('timeout')]
    # Remove duplicates based on bm_mass, num_modules, start_arc_length (keep first occurrence)
    filtered = filtered.drop_duplicates(subset=['bm_mass', 'num_modules', 'start_arc_length'])
    filtered.to_csv(output_csv, index=False)
    print(f"Filtered solutions written to {output_csv}")


def main():
    # Filter solutions first
    filtered_csv = os.path.join('output', 'solutions_filtered.csv')
    filter_solutions(SOLUTIONS_CSV, filtered_csv)
    # Use the filtered CSV for further processing
    extract_fitness_history(filtered_csv, FITNESS_HISTORY_CSV, EVALS_PER_GEN)
    plot_fitness_history_cumulative(FITNESS_HISTORY_CSV, save_path=FITNESS_PLOT_PATH)
    # (Other plotting/analysis functions can be added here later)

if __name__ == "__main__":
    main() 