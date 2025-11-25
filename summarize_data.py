import os
import pickle
import json
from collections import defaultdict
from pathlib import Path


def find_pkl_folders(results_dir):
    """
    查找 results 目录下包含 pkl 文件的文件夹
    """
    pkl_folders = set()
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.pkl'):
                pkl_folders.add(root)
                break
    return sorted(list(pkl_folders))


def extract_node_type(node):
    """
    提取节点类型的字符串表示
    """
    node_type = str(node.node_type)
    # 从 "MCTSNodeType.ROOT" 提取 "ROOT"
    if "." in node_type:
        return node_type.split(".")[-1]
    return node_type


def process_pkl_folder(pkl_folder):
    """
    处理包含 pkl 文件的文件夹，生成三个文件：
    1. txt 文件：按 id 顺序提取 END_NODE 的 final_sql_query
    2. json 文件：记录每个 id 的所有路径
    3. md 文件：统计各种路径的出现次数
    """
    print(f"Processing folder: {pkl_folder}")
    
    # 获取所有 pkl 文件
    pkl_files = sorted([f for f in os.listdir(pkl_folder) if f.endswith('.pkl')])
    
    if not pkl_files:
        print(f"No pkl files found in {pkl_folder}")
        return
    
    # 获取所有存在的 id
    existing_ids = set()
    for pkl_file in pkl_files:
        file_id = pkl_file.replace('.pkl', '')
        if file_id.isdigit():
            existing_ids.add(int(file_id))
    
    # 确定 id 范围（从最小到最大）
    if existing_ids:
        min_id = min(existing_ids)
        max_id = max(existing_ids)
        all_ids = set(range(min_id, max_id + 1))
    else:
        all_ids = set()
    
    # 存储数据
    sql_results = {}  # id -> final_sql
    path_results = {}  # id -> {path_id: [node_types]}
    path_counter = defaultdict(int)  # path_signature -> count
    
    # 处理每个 pkl 文件
    for pkl_file in pkl_files:
        pkl_path = os.path.join(pkl_folder, pkl_file)
        file_id = pkl_file.replace('.pkl', '')
        
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # data 是一个列表，每个元素是一条路径（包含多个节点）
            if not isinstance(data, list):
                print(f"Warning: {pkl_file} does not contain a list")
                continue
            
            # 为当前 id 创建路径字典
            path_results[file_id] = {}
            
            # 提取所有路径
            for path_idx, path in enumerate(data, 1):
                if not path:
                    continue
                
                # 获取最后一个节点（END_NODE）的 final_sql_query
                end_node = path[-1]
                if hasattr(end_node, 'final_sql_query') and end_node.final_sql_query:
                    sql_results[file_id] = end_node.final_sql_query
                
                # 提取路径中的节点类型
                node_types = [extract_node_type(node) for node in path]
                path_results[file_id][str(path_idx)] = node_types
                
                # 统计路径出现次数
                path_signature = " -> ".join(node_types)
                path_counter[path_signature] += 1
        
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
            continue
    
    # 生成输出文件到文件夹的上一级目录
    output_dir = os.path.dirname(pkl_folder)
    
    # 1. 生成 txt 文件：按 id 顺序提取 final_sql
    txt_path = os.path.join(output_dir, "summary_sqls.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        # 按数字顺序遍历所有 id（包括缺失的）
        for file_id in sorted(all_ids):
            file_id_str = str(file_id)
            if file_id_str in sql_results:
                # 删除 SQL 中的换行符
                sql_query = sql_results[file_id_str].replace('\n', ' ').strip()
                f.write(f"{sql_query}\n")
            else:
                # 没有对应 id 的 pkl 文件时输出 SELECT
                f.write("SELECT\n")
    print(f"Generated: {txt_path}")
    
    # 2. 生成 json 文件：记录所有路径
    json_path = os.path.join(output_dir, "summary_paths.json")
    # 按数字顺序排序
    sorted_path_results = {}
    for file_id in sorted(path_results.keys(), key=lambda x: int(x) if x.isdigit() else x):
        sorted_path_results[file_id] = path_results[file_id]
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_path_results, f, indent=4, ensure_ascii=False)
    print(f"Generated: {json_path}")
    
    # 3. 生成 md 文件：统计路径出现次数
    md_path = os.path.join(output_dir, "summary_statistics.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Path Statistics\n\n")
        f.write("## Path Patterns and Counts\n\n")
        
        # 按出现次数降序排序
        sorted_paths = sorted(path_counter.items(), key=lambda x: x[1], reverse=True)
        
        for path_signature, count in sorted_paths:
            f.write(f"[{path_signature}]: {count}\n\n")
        
        # 添加总计信息
        f.write(f"\n## Summary\n\n")
        f.write(f"- Total unique paths: {len(path_counter)}\n")
        f.write(f"- Total path instances: {sum(path_counter.values())}\n")
        f.write(f"- Total processed files: {len(pkl_files)}\n")
    print(f"Generated: {md_path}")
    
    print(f"Processed {len(pkl_files)} pkl files in {pkl_folder}\n")


def main():
    """
    主函数：扫描 results 文件夹并处理所有包含 pkl 文件的子文件夹
    """
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # 查找所有包含 pkl 文件的文件夹
    pkl_folders = find_pkl_folders(results_dir)
    
    if not pkl_folders:
        print("No folders with pkl files found in results directory")
        return
    
    print(f"Found {len(pkl_folders)} folders with pkl files:\n")
    for folder in pkl_folders:
        print(f"  - {folder}")
    print()
    
    # 处理每个文件夹
    for pkl_folder in pkl_folders:
        process_pkl_folder(pkl_folder)
    
    print("All folders processed successfully!")


if __name__ == "__main__":
    main()
