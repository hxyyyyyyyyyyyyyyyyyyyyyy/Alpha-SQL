import argparse
import json
import os
import pickle
import sys
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def find_pkl_folders(results_dir):
    """Find folders containing pkl files under results_dir."""
    pkl_folders = set()
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.pkl'):
                pkl_folders.add(root)
                break
    return sorted(list(pkl_folders))


def extract_node_type(node):
    """Extract node type name from node.node_type."""
    node_type = str(node.node_type)
    if "." in node_type:
        return node_type.split(".")[-1]
    return node_type


def process_pkl_folder(pkl_folder, generate_summary_paths=False):
    """Process one pkl folder and generate summary artifacts."""
    print(f"Processing folder: {pkl_folder}")

    pkl_files = sorted([f for f in os.listdir(pkl_folder) if f.endswith('.pkl')])

    if not pkl_files:
        print(f"No pkl files found in {pkl_folder}")
        return

    path_counter = defaultdict(int)
    summary_paths_repeat_counts = {}

    for pkl_file in pkl_files:
        pkl_path = os.path.join(pkl_folder, pkl_file)
        file_id = pkl_file.replace('.pkl', '')

        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            if not isinstance(data, list):
                print(f"Warning: {pkl_file} does not contain a list")
                continue

            per_file_path_counts = {}

            for path in data:
                if not path:
                    continue

                node_types = [extract_node_type(node) for node in path]
                path_signature = " -> ".join(node_types)

                if path_signature not in per_file_path_counts:
                    per_file_path_counts[path_signature] = 0
                per_file_path_counts[path_signature] += 1

                path_counter[path_signature] += 1

            if generate_summary_paths:
                summary_paths_repeat_counts[file_id] = list(per_file_path_counts.values())

        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
            continue

    output_dir = os.path.dirname(pkl_folder)

    if generate_summary_paths:
        json_path = os.path.join(output_dir, "summary_paths.json")
        sorted_repeat_counts = {}
        for file_id in sorted(summary_paths_repeat_counts.keys(), key=lambda x: int(x) if x.isdigit() else x):
            sorted_repeat_counts[file_id] = summary_paths_repeat_counts[file_id]

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_repeat_counts, f, indent=4, ensure_ascii=False)
        print(f"Generated: {json_path}")

    md_path = os.path.join(output_dir, "summary_statistics.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Path Statistics\n\n")
        f.write("## Path Patterns and Counts\n\n")

        sorted_paths = sorted(path_counter.items(), key=lambda x: x[1], reverse=True)

        for path_signature, count in sorted_paths:
            f.write(f"[{path_signature}]: {count}\n\n")

        f.write(f"\n## Summary\n\n")
        f.write(f"- Total unique paths: {len(path_counter)}\n")
        f.write(f"- Total path instances: {sum(path_counter.values())}\n")
        f.write(f"- Total processed files: {len(pkl_files)}\n")
    print(f"Generated: {md_path}")

    print(f"Processed {len(pkl_files)} pkl files in {pkl_folder}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize pkl outputs under results folders."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory to scan (default: <project_root>/results)",
    )
    parser.add_argument(
        "--generate-summary-paths",
        action="store_true",
        help="Generate summary_paths.json (disabled by default).",
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    if args.results_dir:
        if os.path.isabs(args.results_dir):
            results_dir = args.results_dir
        else:
            results_dir = os.path.join(project_root, args.results_dir)
    else:
        results_dir = os.path.join(project_root, 'results')

    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    pkl_folders = find_pkl_folders(results_dir)

    if not pkl_folders:
        print("No folders with pkl files found in results directory")
        return

    print(f"Found {len(pkl_folders)} folders with pkl files:\n")
    for folder in pkl_folders:
        print(f"  - {folder}")
    print()

    processed_count = 0
    for pkl_folder in pkl_folders:
        process_pkl_folder(
            pkl_folder,
            generate_summary_paths=args.generate_summary_paths,
        )
        processed_count += 1

    print(f"All done. Processed: {processed_count}")


if __name__ == "__main__":
    main()
