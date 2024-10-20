# scripts/update_readme_benchmarks.py

import json
import sys
from pathlib import Path
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_json(json_path):
    """Load JSON data from a file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded data from {json_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {json_path}: {e}")
        sys.exit(1)

def extract_system_specs(machine_info):
    """Extract and format system specifications into Markdown."""
    specs_md = "### 🖥️ **System Specifications**\n\n"
    specs_md += "| Specification         | Details                                 |\n"
    specs_md += "|-----------------------|-----------------------------------------|\n"
    
    # Removed "Node Name"
    
    # Processor Information
    specs_md += f"| **Processor**         | {machine_info.get('processor', 'N/A')} |\n"
    
    # Architecture
    specs_md += f"| **Architecture**      | {machine_info.get('machine', 'N/A')} |\n"
    
    # Python Version and Build
    python_info = machine_info.get('python_version', 'N/A')
    python_impl = machine_info.get('python_implementation', 'N/A')
    python_build = ' '.join(machine_info.get('python_build', []))
    specs_md += f"| **Python Version**    | {python_info} ({python_impl}) |\n"
    specs_md += f"| **Python Build**      | {python_build} |\n"
    
    # Operating System
    specs_md += f"| **Operating System**  | {machine_info.get('system', 'N/A')} {machine_info.get('release', '')} |\n"
    
    # CPU Details
    cpu_info = machine_info.get('cpu', {})
    specs_md += f"| **CPU Brand**         | {cpu_info.get('brand_raw', 'N/A')} |\n"
    specs_md += f"| **CPU Frequency**     | {cpu_info.get('hz_actual_friendly', 'N/A')} |\n"
    specs_md += f"| **L2 Cache Size**     | {cpu_info.get('l2_cache_size', 'N/A') // 1024} KB |\n"
    specs_md += f"| **L3 Cache Size**     | {cpu_info.get('l3_cache_size', 'N/A') // 1024} KB |\n"
    specs_md += f"| **Number of Cores**   | {cpu_info.get('count', 'N/A')} |\n"
    
    return specs_md

def generate_markdown_table(data, total_frames_mapping):
    """
    Generates a Markdown table including FPS.
    
    Args:
        data (dict): Benchmark results from pytest-benchmark.
        total_frames_mapping (dict): Mapping of benchmark names to total frames.
    
    Returns:
        str: Markdown-formatted table.
    """
    benchmarks = {}
    for bench in data['benchmarks']:
        name = bench['name']
        stats = bench['stats']
        benchmarks[name] = {
            'mean': stats['mean'],      # Mean time in seconds
            'stddev': stats['stddev']   # Std dev in seconds
        }

    # Dynamically determine benchmark order based on data
    benchmark_order = [bench['name'] for bench in data['benchmarks']]

    # Generate Markdown table with FPS
    table_md = "| Benchmark                      | Mean Time (s) | Std Dev (s) | FPS    |\n"
    table_md += "|--------------------------------|---------------|------------|--------|\n"
    for bench_name in benchmark_order:
        if bench_name in benchmarks and bench_name in total_frames_mapping:
            mean_time = benchmarks[bench_name]['mean']      # Already in seconds
            stddev = benchmarks[bench_name]['stddev']        # Already in seconds
            total_frames = total_frames_mapping[bench_name]
            fps = total_frames / mean_time if mean_time > 0 else 0
            table_md += f"| {bench_name.replace('_', ' ').title()} | {mean_time:.2f}          | {stddev:.2f}       | {fps:.2f} |\n"
            logger.info(f"Benchmark '{bench_name}': Mean Time = {mean_time:.2f}s, Std Dev = {stddev:.2f}s, FPS = {fps:.2f}")
        else:
            if bench_name not in benchmarks:
                logger.warning(f"Benchmark '{bench_name}' not found in benchmark data.")
            if bench_name not in total_frames_mapping:
                logger.warning(f"Total frames for benchmark '{bench_name}' not found.")
            table_md += f"| {bench_name.replace('_', ' ').title()} | N/A           | N/A        | N/A    |\n"

    return table_md

def generate_fps_plot(data, total_frames_mapping, output_path):
    """
    Generates a bar chart for FPS of each benchmark.
    
    Args:
        data (dict): Benchmark results from pytest-benchmark.
        total_frames_mapping (dict): Mapping of benchmark names to total frames.
        output_path (str): Path to save the generated plot.
    """
    benchmarks = {}
    for bench in data['benchmarks']:
        name = bench['name']
        stats = bench['stats']
        benchmarks[name] = {
            'mean': stats['mean'],      # Mean time in seconds
            'stddev': stats['stddev']   # Std dev in seconds
        }

    # Prepare data for plotting
    bench_names = []
    fps_values = []
    for bench_name, stats in benchmarks.items():
        if bench_name in total_frames_mapping:
            mean_time = stats['mean']
            fps = total_frames_mapping[bench_name] / mean_time if mean_time > 0 else 0
            bench_names.append(bench_name.replace('_', ' ').title())
            fps_values.append(fps)
            logger.info(f"Plotting Benchmark '{bench_name}': FPS = {fps:.2f}")
        else:
            logger.warning(f"Total frames for benchmark '{bench_name}' not found. Skipping plot for this benchmark.")

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bench_names, fps_values, color='skyblue')
    plt.xlabel('Benchmark')
    plt.ylabel('Frames Per Second (FPS)')
    plt.title('FPS Comparison Across Benchmarks')
    plt.ylim(0, max(fps_values) * 1.2 if fps_values else 1)

    # Annotate bars with FPS values
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout()
    try:
        plt.savefig(output_path)
        logger.info(f"Generated FPS comparison plot at {output_path}")
    except Exception as e:
        logger.error(f"Failed to save FPS plot: {e}")
        sys.exit(1)
    plt.close()

def generate_mean_time_plot(data, total_frames_mapping, output_path):
    """
    Generates a bar chart for Mean Time of each benchmark.
    
    Args:
        data (dict): Benchmark results from pytest-benchmark.
        total_frames_mapping (dict): Mapping of benchmark names to total frames.
        output_path (str): Path to save the generated plot.
    """
    benchmarks = {}
    for bench in data['benchmarks']:
        name = bench['name']
        stats = bench['stats']
        benchmarks[name] = {
            'mean': stats['mean'],      # Mean time in seconds
            'stddev': stats['stddev']   # Std dev in seconds
        }

    # Prepare data for plotting
    bench_names = []
    mean_times = []
    for bench_name, stats in benchmarks.items():
        if bench_name in total_frames_mapping:
            mean_time = stats['mean']
            bench_names.append(bench_name.replace('_', ' ').title())
            mean_times.append(mean_time)
            logger.info(f"Plotting Benchmark '{bench_name}': Mean Time = {mean_time:.2f}s")
        else:
            logger.warning(f"Total frames for benchmark '{bench_name}' not found. Skipping plot for this benchmark.")

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bench_names, mean_times, color='salmon')
    plt.xlabel('Benchmark')
    plt.ylabel('Mean Time (s)')
    plt.title('Mean Time Comparison Across Benchmarks')
    plt.ylim(0, max(mean_times) * 1.2 if mean_times else 1)

    # Annotate bars with Mean Time values
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}s',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout()
    try:
        plt.savefig(output_path)
        logger.info(f"Generated Mean Time comparison plot at {output_path}")
    except Exception as e:
        logger.error(f"Failed to save Mean Time plot: {e}")
        sys.exit(1)
    plt.close()

def update_readme(system_specs_md, table_md, plots_md, readme_path):
    """
    Updates the README.md with system specs, benchmark table, and plots.
    
    Args:
        system_specs_md (str): Markdown-formatted system specifications.
        table_md (str): Markdown-formatted benchmark table.
        plots_md (str): Markdown-formatted plots section.
        readme_path (str): Path to README.md.
    """
    try:
        readme = Path(readme_path).read_text(encoding='utf-8')
        logger.info(f"Read README.md from {readme_path}")
    except Exception as e:
        logger.error(f"Failed to read README.md: {e}")
        sys.exit(1)
    
    start_marker = "<!-- BENCHMARKS_START -->"
    end_marker = "<!-- BENCHMARKS_END -->"

    start_idx = readme.find(start_marker)
    end_idx = readme.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        logger.error("Benchmark markers not found in README.md")
        sys.exit(1)

    # Extract the content before and after the markers
    before = readme[:start_idx + len(start_marker)]
    after = readme[end_idx:]

    # Combine all sections
    new_section = f"\n\n{system_specs_md}\n\n{table_md}\n\n{plots_md}\n\n"

    # Insert the new section between the markers
    new_readme = f"{before}{new_section}{after}"

    try:
        # Write back to README.md with UTF-8 encoding
        Path(readme_path).write_text(new_readme, encoding='utf-8')
        logger.info("README.md updated with system specs, benchmark table, and plots.")
    except Exception as e:
        logger.error(f"Failed to write README.md: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python update_readme_benchmarks.py <benchmark_json_path> <readme_path>")
        sys.exit(1)

    benchmark_json_path = sys.argv[1]
    readme_path = sys.argv[2]
    total_frames_json_path = "tests/benchmarks/total_frames.json"

    # Check if total_frames.json exists
    if not Path(total_frames_json_path).exists():
        logger.error(f"Error: {total_frames_json_path} does not exist.")
        sys.exit(1)

    # Load JSON data
    benchmark_data = load_json(benchmark_json_path)
    total_frames_mapping = load_json(total_frames_json_path)

    # Extract system specs
    machine_info = benchmark_data.get('machine_info', {})
    system_specs_md = extract_system_specs(machine_info)

    # Generate benchmark table
    table_md = generate_markdown_table(benchmark_data, total_frames_mapping)

    # Generate plots
    plots_dir = Path("scripts/benchmarks/")
    plots_dir.mkdir(parents=True, exist_ok=True)
    fps_plot_path = plots_dir / "fps_comparison.png"
    mean_time_plot_path = plots_dir / "mean_time_comparison.png"
    generate_fps_plot(benchmark_data, total_frames_mapping, str(fps_plot_path))
    generate_mean_time_plot(benchmark_data, total_frames_mapping, str(mean_time_plot_path))

    # Create Markdown for plots
    plots_md = "### 📊 **Benchmark Visualizations**\n\n"
    plots_md += f"![FPS Comparison]({fps_plot_path.as_posix()})\n\n"
    plots_md += f"![Mean Time Comparison]({mean_time_plot_path.as_posix()})\n\n"

    # Update README.md
    update_readme(system_specs_md, table_md, plots_md, readme_path)

if __name__ == "__main__":
    main()
