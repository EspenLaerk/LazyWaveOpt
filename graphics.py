"""
Graphics for optimization output display with ANSI terminal colors
"""

import os
import sys
import time

# ANSI color codes
BLUE = "\033[94m"
GREEN = "\033[92m"
WHITE = "\033[97m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
BROWN = "\033[33m"  # Actually more like amber/orange
RED = "\033[91m"
CYAN = "\033[96m"
END = "\033[0m"

# Ensure these are accessible for import
__all__ = [
    "BLUE", "GREEN", "WHITE", "YELLOW", "BOLD", "BROWN", "RED", "CYAN", "END",
    "get_colors", "clear_terminal",
    "progress_bar", "format_time",
    "print_header", "print_footer", "print_section_header",
    "print_single_floater", "print_three_floaters", "single_floater_bottom", "three_floater_bottoms",
    "print_waterline", "print_seabed",
    "print_underwater_graphic",
    "print_optimization_progress", "print_multiprocess_optimization_progress", "print_optimization_cancelled"
]

def get_colors():
    """Return a dictionary of color codes"""
    return {
        "BLUE": BLUE,
        "GREEN": GREEN,
        "WHITE": WHITE,
        "YELLOW": YELLOW,
        "BOLD": BOLD,
        "BROWN": BROWN,
        "RED": RED,
        "CYAN": CYAN,
        "END": END
    }

def clear_terminal():
    """
    Clear the terminal screen in a cross-platform way.
    Uses ANSI escape sequences for most terminals and os-specific commands as fallback.
    """
    # First try ANSI escape sequences
    print("\033[H\033[J", end="", flush=True)
    
    # Then use platform-specific clear command
    if sys.platform == "win32":
        os.system('cls')
    else:
        os.system('clear')
    
    # Add a small delay to ensure the clear command completes
    time.sleep(0.1)

# --- Basic Drawing/Formatting Utilities ---
def progress_bar(percent, width=20, color=None):
    filled_width = int(width * percent / 100)
    if color:
        return color + "█" * filled_width + "\033[0m" + "-" * (width - filled_width)
    else:
        return "█" * filled_width + "-" * (width - filled_width)

def format_time(seconds):
    if seconds < 60:
        rounded_seconds = 10 * round(seconds / 10)
        return f"{int(rounded_seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        rounded_seconds = 10 * round(remaining_seconds / 10)
        if rounded_seconds == 60:
            return f"{minutes+1}m00s"
        else:
            sec_str = f"{int(rounded_seconds):02d}"
            return f"{minutes}m{sec_str}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = (seconds % 3600) // 60
        rounded_minutes = round(remaining_minutes)
        if rounded_minutes == 60:
            return f"{hours+1}h00m"
        else:
            min_str = f"{rounded_minutes:02d}"
            return f"{hours}h{min_str}m"

# --- Header, Footer, and Section Titles ---
def print_header(title="OPTIMIZATION COMPLETE", width=80):
    print("\n")
    print(f"{BLUE}" + "=" * width + f"{END}")
    print(f"{BOLD}{GREEN}" + " " * ((width - len(title)) // 2) + title + " " * ((width - len(title)) // 2) + f"{END}")
    print(f"{BLUE}" + "=" * width + f"{END}")

def print_footer(width=80):
    print(f"\n  {BOLD}{GREEN}All data saved. Optimization complete.{END}")
    print(f"  {BLUE}" + "=" * width + f"{END}\n")

def print_section_header(title, width=80):
    print(f"\n  {BOLD}{CYAN}{title}{END}\n")

# --- Floater Graphics ---
def print_single_floater():
    """Print a single floating wind turbine graphic using the same style as the center floater in print_three_floaters."""
    turbine_lines = [
        f"           {WHITE} \\\\      //     {END}",
        f"            {WHITE} \\\\    //       {END}",
        f"             {WHITE} \\\\  //        {END}",
        f"             {WHITE}   ||            {END}",
        f"              {WHITE}//||\\\\        {END}",
        f"             {WHITE}// || \\\\       {END}",
        f"            {WHITE}//  ||  \\\\      {END}",
        f"                {WHITE}||            {END}",
        f"          {YELLOW}___   ||_   ___   {END}",
        f"         {YELLOW}|   |_|   |_|   |  {END}"
    ]
    print("\n")
    for line in turbine_lines:
        print(f"  {line}")

def print_three_floaters():
    """Print three floating wind turbines graphic"""
    turbine_lines = [
        f"           {WHITE} \\\\      //               \\\\      //              \\\\      //    {END}",
        f"            {WHITE} \\\\    //                 \\\\    //                \\\\    //     {END}",
        f"             {WHITE} \\\\  //                   \\\\  //                  \\\\  //      {END}",
        f"             {WHITE}   ||                       ||                      ||            {END}",
        f"              {WHITE}//||\\\\                   //||\\\\                  //||\\\\      {END}",
        f"             {WHITE}// || \\\\                 // || \\\\                // || \\\\         {END}",
        f"            {WHITE}//  ||  \\\\               //  ||  \\\\              //  ||  \\\\            {END}",
        f"                {WHITE}||                       ||                      ||                     {END}",
        f"          {YELLOW}___   ||_   ___          ___   ||_   ___         ___   ||_   ___             {END}",
        f"         {YELLOW}|   |_|   |_|   |        |   |_|   |_|   |       |   |_|   |_|   |            {END}"
    ]
    print("\n")
    for line in turbine_lines:
        print(f"  {line}")

def single_floater_bottom():
    """Return the underwater bottom block for a single floater as a list of lines."""
    colors = get_colors()
    return [
        f"         {colors['YELLOW']}|   | |   | |   |{colors['END']}",
        f"         {colors['YELLOW']}|   |_|   |_|   |{colors['END']}",
        f"         {colors['YELLOW']}|___| |___| |___|{colors['END']}"
    ]

def three_floater_bottoms():
    """Return the underwater graphic block as a list of lines (for use in display buffers)."""
    colors = get_colors()
    return [
        f"         {colors['YELLOW']}|   | |   | |   |        |   | |   | |   |       |   | |   | |   |            {colors['END']}",
        f"         {colors['YELLOW']}|   |_|   |_|   |        |   |_|   |_|   |       |   |_|   |_|   |            {colors['END']}",
        f"         {colors['YELLOW']}|___| |___| |___|        |___| |___| |___|       |___| |___| |___|            {colors['END']}"
    ]

# --- Water and Seabed Graphics ---
def print_waterline(width=80):
    print(f"  {BLUE}" + "∿" * width + f"{END}")

def print_seabed(width=80):
    print(f"\n  {BROWN}" + "~" * width + f"{END}")

# --- Composite/Scene Graphics ---
def print_underwater_graphic():
    for line in three_floater_bottoms():
        print(line)

# --- Progress and Status Displays ---
def print_optimization_progress(
    gen_percent, total_percent, elapsed, avg_time, est_remaining,
    current_gen, total_generations, current_pop, total_populations,
    completed_evaluations, total_evaluations, optimization_complete=False,
    execution_info=None, bar_width=60
):
    colors = get_colors()
    gen_str = f"{current_gen}/{total_generations}"
    pop_str = f"{current_pop}/{total_populations}"
    eval_str = f"{min(completed_evaluations, total_evaluations)}/{total_evaluations}"
    elapsed_str = format_time(elapsed)
    avg_str = f"{avg_time:.1f}s"
    remain_str = format_time(est_remaining)

    # Progress bars (vertically aligned)
    overall_bar = f"  {colors['BOLD']}Overall    {colors['END']}[" + progress_bar(total_percent, bar_width, color=colors['BLUE']) + f"] {total_percent:5.1f}%"
    gen_bar     = f"  {colors['BOLD']}Generation {colors['END']}[" + progress_bar(gen_percent, bar_width, color=colors['GREEN']) + f"] {gen_percent:5.1f}%"

    # Add completion message if optimization is complete
    status_message = ""
    if optimization_complete:
        status_message = f"  {colors['BOLD']}{colors['GREEN']}Optimization complete!{colors['END']}"

    display_buffer = []
    display_buffer.append("")
    # Use the single floater
    print_single_floater()
    print_waterline(bar_width + 20)
    display_buffer.append("")
    display_buffer.append(f"  {colors['BOLD']}OPTIMIZATION STATUS{colors['END']}")
    display_buffer.append("")
    display_buffer.append(f"  {colors['BOLD']}Generation:{colors['END']} {gen_str:<5}     {colors['BOLD']}Population:{colors['END']} {pop_str:<5}     {colors['BOLD']}Time:{colors['END']} {elapsed_str}")
    display_buffer.append(f"  {colors['BOLD']}Evaluations:{colors['END']} {eval_str:<10}  {colors['BOLD']}Avg Solution:{colors['END']} {avg_str:<6}   {colors['BOLD']}Remaining:{colors['END']} {remain_str}")
    display_buffer.append("")
    display_buffer.append(overall_bar)
    if status_message:
        display_buffer.append("")
        display_buffer.append(status_message)
    if execution_info:
        display_buffer.append("")
        display_buffer.append(f"  {colors['BOLD']}Recent Execution:{colors['END']}")
        for line in execution_info:
            display_buffer.append(f"  {line}")
    display_buffer.append("")
    print("\n".join(display_buffer), flush=True)
    print_seabed(bar_width + 20)

def print_optimization_cancelled(width=80):
    """Print a user-facing message about optimization cancellation."""
    print("\n" + BLUE + "=" * width + END)
    print(f"  {BOLD}{RED}Optimization cancelled by user.{END}\n")
    print(BLUE + "=" * width + END + "\n")

def print_multiprocess_optimization_progress(
    gen_percent, total_percent, elapsed, avg_time, est_remaining,
    current_gen, total_generations, current_pop, total_populations,
    completed_evaluations, total_evaluations, workers_str, batch_str,
    bar_width=70, line_width=100
):
    colors = get_colors()
    gen_str = f"{current_gen}/{total_generations}"
    pop_str = f"{current_pop}/{total_populations}"
    eval_str = f"{min(completed_evaluations, total_evaluations)}/{total_evaluations}"
    elapsed_str = format_time(elapsed)
    avg_str = f"{avg_time:.1f}s"
    remain_str = format_time(est_remaining)

    # Progress bars
    overall_bar = f"  {colors['BOLD']}Overall Progress{colors['END']}  [" + progress_bar(total_percent, bar_width, color=colors['BLUE']) + f"] {total_percent:5.1f}%"

    display_buffer = []
    display_buffer.append("")
    print_three_floaters()
    print_waterline(line_width)
    display_buffer.append("")
    display_buffer.append(f"  {colors['BOLD']}OPTIMIZATION STATUS{colors['END']}")
    display_buffer.append("")
    display_buffer.append(f"  {colors['BOLD']}Generation:{colors['END']} {gen_str}     {colors['BOLD']}Population:{colors['END']} {pop_str}     {colors['BOLD']}Solutions Evaluated:{colors['END']} {eval_str}")
    display_buffer.append(f"  {colors['BOLD']}Elapsed Time:{colors['END']} {elapsed_str}     {colors['BOLD']}Avg Solution Time:{colors['END']} {avg_str}     {colors['BOLD']}Remaining Time:{colors['END']} {remain_str}")
    display_buffer.append(f"  {colors['BOLD']}Active Workers:{colors['END']} {workers_str}     {colors['BOLD']}Current Batch Progress:{colors['END']} {batch_str}")
    display_buffer.append("")
    display_buffer.append(overall_bar)
    seabed = f"  {colors['BROWN']}" + "~" * line_width + f"{colors['END']}"
    display_buffer.append(seabed)
    print("\n".join(display_buffer), flush=True) 