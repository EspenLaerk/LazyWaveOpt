"""
Module for tracking optimization progress
"""

import time
import logging
import graphics

class OptimizationState:
    def __init__(self):
        # Basic state
        self.completed_evaluations = 0
        self.failed_evaluations = 0
        self.invalid_evaluations = 0
        self.total_evaluations = None
        self.current_generation = 0
        self.total_generations = None
        self.generation_solutions_evaluated = 0
        self.generation_solutions_total = 0
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.start_time = time.time()
        self.logger = None
        self.terminal_output_enabled = True
        
        # Solution timing tracking
        self.solution_times = []
        self.solution_times_window = 10
        self._last_display_update = 0

        # New attributes
        self.solutions_cache = {}
        self.metrics_cache = {}
        self.pending_solutions = []
        self.pending_indices = []
        self.pending_cache_keys = []
        self.timed_out_evaluations = 0
        self._last_avg_solution_time = 0.0
        self._last_est_remaining_time = 0.0

    def update_progress(self, count=1, cached=False, failed=False, eval_time=None, metrics=None):
        """
        Update progress tracking.
        
        Args:
            count: Number of solutions to add to progress
            cached: Whether this was a cached solution
            failed: Whether this was a failed evaluation
            eval_time: Time taken to evaluate solution (None for cached)
            metrics: Metrics associated with the evaluation
        """
        # Track evaluation time for calculating averages
        if eval_time is not None and not cached:
            self.solution_times.append(eval_time)
            # Keep only the most recent times in the window
            if len(self.solution_times) > self.solution_times_window:
                self.solution_times = self.solution_times[-self.solution_times_window:]
        
        # Always update completed evaluations count
        self.completed_evaluations += count
        
        # For generation progress, we count all solutions towards generation progress,
        # even cached ones, since they represent solutions that have been "processed"
        # in this generation, whether through evaluation or cache retrieval
        self.generation_solutions_evaluated += count
        
        # Log status to the log file - only if logger is available
        if self.logger is not None:
            status = "cached" if cached else "failed" if failed else "evaluated"
            # Protect against division by zero
            gen_percent = 0
            if self.generation_solutions_total > 0:
                gen_percent = min(100, (self.generation_solutions_evaluated / self.generation_solutions_total) * 100)
                
            total_percent = 0
            if self.total_evaluations > 0:
                total_percent = min(100, (self.completed_evaluations / self.total_evaluations) * 100)
                
            # Log detailed information about the progress state
            self.logger.info(f"Progress update: {count} solution(s) {status}. "
                         f"Total: {self.completed_evaluations}/{self.total_evaluations} ({total_percent:.1f}%), "
                         f"Generation: {self.generation_solutions_evaluated}/{self.generation_solutions_total} ({gen_percent:.1f}%)")

        if failed:
            self.failed_evaluations += count
            # Check if this failure was due to a timeout
            if metrics and (isinstance(metrics, dict) and metrics.get("timeout")):
                self.timed_out_evaluations += count

    def update_timing_stats(self, completed_evaluations, total_evaluations, start_time):
        elapsed = time.time() - start_time
        if completed_evaluations > 0 and elapsed > 0:
            self._last_avg_solution_time = round(elapsed / completed_evaluations, 1)
        else:
            self._last_avg_solution_time = 0.0
        remaining_evals = max(0, total_evaluations - completed_evaluations)
        self._last_est_remaining_time = remaining_evals * self._last_avg_solution_time

    def get_avg_solution_time(self):
        return self._last_avg_solution_time

    def get_est_remaining_time(self):
        return self._last_est_remaining_time

    def format_time_exact(self, seconds):
        """Format time with more precision for reports"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.2f}s"
        else:
            hours = int(seconds // 3600)
            remaining = seconds % 3600
            minutes = int(remaining // 60)
            seconds = remaining % 60
            return f"{hours}h {minutes}m {seconds:.2f}s"

    def update_terminal_info(self, force=False):
        """
        Update the terminal display with current optimization status.
        """
        # Only update if terminal output is enabled
        if not self.terminal_output_enabled:
            return
        # Optionally, throttle updates unless forced
        now = time.time()
        if not force and now - self._last_display_update < 1.0:
            return
        self._last_display_update = now

        # Clear the terminal before printing progress
        graphics.clear_terminal()

        # Calculate progress percentages
        gen_percent = min((self.generation_solutions_evaluated / self.generation_solutions_total) * 100 if self.generation_solutions_total else 0, 100)
        total_percent = min((self.completed_evaluations / self.total_evaluations) * 100 if self.total_evaluations else 0, 100)
        elapsed = time.time() - self.start_time
        avg_time = self.get_avg_solution_time()
        est_remaining = self.get_est_remaining_time()

        # Dynamically calculate generation and population from completed_evaluations
        if self.total_generations and self.total_evaluations and self.generation_solutions_total:
            # At the end, show all generations complete
            if self.completed_evaluations >= self.total_evaluations:
                current_gen = self.total_generations
                current_pop = (self.total_generations or 0) + 1
            else:
                # First generation: population_size
                pop_size = self.generation_solutions_total
                keep_parents = self.total_evaluations - pop_size - (self.total_generations - 1) * (pop_size - 1)
                # Calculate which generation we're in
                if self.completed_evaluations < pop_size:
                    current_gen = 0
                else:
                    current_gen = min(self.total_generations - 1, (self.completed_evaluations - pop_size) // (pop_size - keep_parents) + 1)
                current_pop = current_gen + 1
        else:
            current_gen = self.current_generation
            current_pop = self.current_generation + 1

        graphics.print_optimization_progress(
            gen_percent=gen_percent,
            total_percent=total_percent,
            elapsed=elapsed,
            avg_time=avg_time,
            est_remaining=est_remaining,
            current_gen=current_gen,
            total_generations=self.total_generations or 0,
            current_pop=current_pop,
            total_populations=(self.total_generations or 0) + 1,
            completed_evaluations=self.completed_evaluations,
            total_evaluations=self.total_evaluations or 0
        )

    def get_timed_out_evaluations(self):
        return self.timed_out_evaluations

# Create a global optimization state object
optimization_state = OptimizationState() 