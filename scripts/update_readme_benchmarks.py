import json
import sys
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_gpu_info():
    """Retrieve GPU information using torch.cuda."""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu = {
                'name': gpu_props.name,
                'memory_total': gpu_props.total_memory  # in bytes
            }
            gpus.append(gpu)
    return gpus

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
    
    # Cache sizes and number of cores
    l2_cache = cpu_info.get('l2_cache_size', None)
    l3_cache = cpu_info.get('l3_cache_size', None)
    l2_cache_str = f"{l2_cache // 1024} KB" if isinstance(l2_cache, int) else "N/A"
    l3_cache_str = f"{l3_cache // 1024} KB" if isinstance(l3_cache, int) else "N/A"
    specs_md += f"| **L2 Cache Size**     | {l2_cache_str} |\n"
    specs_md += f"| **L3 Cache Size**     | {l3_cache_str} |\n"
    specs_md += f"| **Number of Cores**   | {cpu_info.get('count', 'N/A')} |\n"
    
    # GPU Details
    gpus = get_gpu_info()
    if gpus:
        for i, gpu in enumerate(gpus):
            gpu_name = gpu.get('name', 'N/A')
            gpu_memory = gpu.get('memory_total', 'N/A')
            if isinstance(gpu_memory, int):
                gpu_memory = f"{gpu_memory / (1024 ** 3):.2f} GB"
            specs_md += f"| **GPU #{i + 1}**           | {gpu_name} ({gpu_memory}) |\n"
    else:
        specs_md += f"| **GPU**               | N/A |\n"
    
    return specs_md

def generate_markdown_table(data, total_frames_mapping):
    """Generates a Markdown table with detailed benchmarks for BENCHMARKS.md."""
    benchmarks = {}
    for bench in data['benchmarks']:
        name = bench.get('name', 'N/A')
        stats = bench.get('stats', {})
        benchmarks[name] = {
            'mean': stats.get('mean', 'N/A'),      # Mean time in seconds
            'stddev': stats.get('stddev', 'N/A')   # Std dev in seconds
        }

    table_md = "| Benchmark                      | Video Size  | Threads | Mean Time (s) | Std Dev (s) | FPS    |\n"
    table_md += "|--------------------------------|-------------|---------|---------------|-------------|--------|\n"
    for bench_name, stats in benchmarks.items():
        mean_time = stats['mean']
        stddev = stats['stddev']
        total_frames_info = total_frames_mapping.get(bench_name, {})
        total_frames = total_frames_info.get("total_frames", None)
        video_size = total_frames_info.get("video_size", "N/A")
        num_threads = total_frames_info.get("num_threads", "N/A")
        
        if mean_time != 'N/A' and total_frames:
            fps = total_frames / mean_time if mean_time > 0 else 0
        else:
            fps = 'N/A'

        table_md += f"| {bench_name.replace('_', ' ').title()} | {video_size} | {num_threads} | {mean_time if mean_time != 'N/A' else 'N/A'} | {stddev if stddev != 'N/A' else 'N/A'} | {fps} |\n"
    
    return table_md

def generate_markdown_summary_table(data, total_frames_mapping):
    """Generates a brief Markdown summary table for the README."""
    benchmarks = {}
    for bench in data['benchmarks']:
        name = bench.get('name', 'N/A')
        stats = bench.get('stats', {})
        benchmarks[name] = {
            'mean': stats.get('mean', 'N/A'),      # Mean time in seconds
            'stddev': stats.get('stddev', 'N/A')   # Std dev in seconds
        }

    summary_md = "| Library  | Device       | Frames per Second (FPS) |\n"
    summary_md += "|----------|--------------|-------------------------|\n"
    
    for bench_name, stats in benchmarks.items():
        mean_time = stats['mean']
        total_frames_info = total_frames_mapping.get(bench_name, {})
        total_frames = total_frames_info.get("total_frames", None)
        
        if mean_time != 'N/A' and total_frames:
            fps = total_frames / mean_time if mean_time > 0 else 0
        else:
            fps = 'N/A'
        
        device = "CUDA" if "cuda" in bench_name.lower() else "CPU"
        library = "CeLux" if "celux" in bench_name.lower() else "PyAV" if "pyav" in bench_name.lower() else "OpenCV"
        
        summary_md += f"| {library} | {device}      | {fps}                  |\n"

    return summary_md

def generate_fps_plot(data, total_frames_mapping, output_path):
    """Generates a bar chart for FPS of each benchmark."""
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
            total_frames = total_frames_mapping[bench_name]["total_frames"]
            fps = total_frames / mean_time if mean_time > 0 else 0
            bench_names.append(bench_name.replace('_', ' ').title())
            fps_values.append(fps)

    if not fps_values:
        logger.warning("No FPS data available to plot.")
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bench_names, fps_values, color='skyblue')
    plt.xlabel('Benchmark')
    plt.ylabel('Frames Per Second (FPS)')
    plt.title('FPS Comparison Across Benchmarks')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_mean_time_plot(data, total_frames_mapping, output_path):
    """Generates a bar chart for Mean Time of each benchmark."""
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

    if not mean_times:
        logger.warning("No Mean Time data available to plot.")
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bench_names, mean_times, color='salmon')
    plt.xlabel('Benchmark')
    plt.ylabel('Mean Time (s)')
    plt.title('Mean Time Comparison Across Benchmarks')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def update_readme(summary_md, readme_path):
    """Updates the README.md with a brief summary table."""
    try:
        readme = Path(readme_path).read_text(encoding='utf-8')
        logger.info(f"Read README.md from {readme_path}")
    except Exception as e:
        logger.error(f"Failed to read README.md: {e}")
        sys.exit(1)

    # Markers
    summary_marker = "<!-- BENCHMARK_SUMMARY_START -->"
    summary_end_marker = "<!-- BENCHMARK_SUMMARY_END -->"

    summary_start_idx = readme.find(summary_marker)
    summary_end_idx = readme.find(summary_end_marker)

    if summary_start_idx != -1 and summary_end_idx != -1:
        before_summary = readme[:summary_start_idx + len(summary_marker)]
        after_summary = readme[summary_end_idx:]
        summary_section = f"\n\n## 📊 Benchmark Summary\n\n{summary_md}\n\nFor more details, see [Benchmarks](docs/BENCHMARKS.md).\n\n"
        new_readme = f"{before_summary}{summary_section}{after_summary}"

        try:
            Path(readme_path).write_text(new_readme, encoding='utf-8')
            logger.info("README.md updated with benchmark summary.")
        except Exception as e:
            logger.error(f"Failed to write README.md: {e}")
            sys.exit(1)

def update_benchmarks_doc(system_specs_md, table_md, plots_md, benchmarks_path):
    """Updates the BENCHMARKS.md with full details, including tables and plots."""
    try:
        benchmarks_doc = Path(benchmarks_path).read_text(encoding='utf-8')
        logger.info(f"Read BENCHMARKS.md from {benchmarks_path}")
    except Exception as e:
        logger.error(f"Failed to read BENCHMARKS.md: {e}")
        sys.exit(1)

    # Markers for full benchmarks section
    start_marker = "<!-- BENCHMARKS_START -->"
    end_marker = "<!-- BENCHMARKS_END -->"

    start_idx = benchmarks_doc.find(start_marker)
    end_idx = benchmarks_doc.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        logger.error("Benchmark markers not found in BENCHMARKS.md")
        sys.exit(1)

    # Insert main benchmarks section
    before = benchmarks_doc[:start_idx + len(start_marker)]
    after = benchmarks_doc[end_idx:]
    new_section = f"\n\n{system_specs_md}\n\n{table_md}\n\n{plots_md}\n\n"
    new_benchmarks_doc = f"{before}{new_section}{after}"

    try:
        Path(benchmarks_path).write_text(new_benchmarks_doc, encoding='utf-8')
        logger.info("BENCHMARKS.md updated with detailed benchmarks.")
    except Exception as e:
        logger.error(f"Failed to write BENCHMARKS.md: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 4:
        print("Usage: python update_readme_benchmarks.py <benchmark_json_path> <readme_path> <benchmarks_md_path>")
        sys.exit(1)

    benchmark_json_path = sys.argv[1]
    readme_path = sys.argv[2]
    benchmarks_md_path = sys.argv[3]
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

    # Generate tables and summary
    table_md = generate_markdown_table(benchmark_data, total_frames_mapping)
    summary_md = generate_markdown_summary_table(benchmark_data, total_frames_mapping)

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

    # Update README.md with summary
    update_readme(summary_md, readme_path)

    # Update BENCHMARKS.md with full details
    update_benchmarks_doc(system_specs_md, table_md, plots_md, benchmarks_md_path)

if __name__ == "__main__":
    main()
#python scripts\update_readme_benchmarks.py benchmark_results.json README.md docs/BENCHMARKS.md
