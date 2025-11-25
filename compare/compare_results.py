"""
对比两个SQL文件的执行结果
分别读取 compare/data 文件夹中的两份文件，通过 BIRD 数据集进行执行，并与 golden_sql 进行对比
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphasql.database.sql_execution import execute_sql_with_timeout, SQLExecutionResultType


class SQLComparator:
    """SQL执行结果对比器"""
    
    def __init__(self, bird_dev_path: str, db_base_path: str):
        """
        初始化对比器
        
        Args:
            bird_dev_path: BIRD dev.json 文件路径
            db_base_path: 数据库文件夹路径
        """
        self.bird_dev_path = bird_dev_path
        self.db_base_path = db_base_path
        self.golden_data = self._load_golden_data()
        
    def _load_golden_data(self) -> List[Dict]:
        """加载 BIRD 数据集的 golden SQL"""
        with open(self.bird_dev_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def read_sql_file(self, file_path: str) -> List[str]:
        """读取 SQL 文件，每行一个 SQL 语句"""
        with open(file_path, 'r', encoding='utf-8') as f:
            sqls = [line.strip() for line in f.readlines()]
        return sqls
    
    def get_db_path(self, db_id: str) -> str:
        """获取数据库文件路径"""
        db_path = os.path.join(self.db_base_path, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        return db_path
    
    def execute_and_compare(self, sql: str, golden_sql: str, db_path: str, timeout: int = 60) -> Dict:
        """
        执行 SQL 并与 golden SQL 的结果对比
        
        Returns:
            字典包含:
                - predicted_result: 预测SQL的执行结果
                - golden_result: golden SQL的执行结果
                - is_match: 结果是否匹配
                - error_message: 错误信息（如果有）
        """
        # 执行预测的 SQL
        pred_result = execute_sql_with_timeout(db_path, sql, timeout)
        
        # 执行 golden SQL
        golden_result = execute_sql_with_timeout(db_path, golden_sql, timeout)
        
        # 检查是否匹配
        is_match = False
        error_message = None
        
        if pred_result.result_type == SQLExecutionResultType.SUCCESS and \
           golden_result.result_type == SQLExecutionResultType.SUCCESS:
            # 比较结果
            is_match = self._compare_results(pred_result.result, golden_result.result)
        else:
            # 记录错误信息
            errors = []
            if pred_result.result_type != SQLExecutionResultType.SUCCESS:
                errors.append(f"Predicted SQL error: {pred_result.error_message}")
            if golden_result.result_type != SQLExecutionResultType.SUCCESS:
                errors.append(f"Golden SQL error: {golden_result.error_message}")
            error_message = "; ".join(errors)
        
        return {
            "predicted_result": pred_result.to_dict(),
            "golden_result": golden_result.to_dict(),
            "is_match": is_match,
            "error_message": error_message
        }
    
    def _compare_results(self, result1, result2) -> bool:
        """比较两个查询结果是否相同"""
        if result1 is None and result2 is None:
            return True
        if result1 is None or result2 is None:
            return False
        
        # 将结果转换为集合进行比较（忽略顺序）
        try:
            set1 = set(tuple(row) for row in result1)
            set2 = set(tuple(row) for row in result2)
            return set1 == set2
        except:
            # 如果无法转换为集合，直接比较列表
            return sorted(result1) == sorted(result2)
    
    def _normalize_sql(self, sql: str) -> str:
        """归一化 SQL 语句用于比较"""
        # 转换为小写
        normalized = sql.lower()
        # 去除多余空白
        normalized = ' '.join(normalized.split())
        # 去除末尾的分号
        normalized = normalized.rstrip(';')
        return normalized
    
    def compare_files(self, file1_path: str, file2_path: str, output_dir: str = None):
        """
        对比两个 SQL 文件的执行结果
        
        Args:
            file1_path: 第一个 SQL 文件路径
            file2_path: 第二个 SQL 文件路径
            output_dir: 输出目录（可选）
        """
        # 读取两个文件
        sqls_file1 = self.read_sql_file(file1_path)
        sqls_file2 = self.read_sql_file(file2_path)
        
        # 确保文件长度一致
        if len(sqls_file1) != len(self.golden_data):
            print(f"Warning: File 1 has {len(sqls_file1)} lines, but golden data has {len(self.golden_data)} entries")
        if len(sqls_file2) != len(self.golden_data):
            print(f"Warning: File 2 has {len(sqls_file2)} lines, but golden data has {len(self.golden_data)} entries")
        
        # 统计信息
        file1_stats = {
            "total": 0,
            "correct": 0,
            "execution_error": 0,
            "result_mismatch": 0,
            "sql_exact_match": 0  # 新增：SQL完全匹配的数量
        }
        file2_stats = {
            "total": 0,
            "correct": 0,
            "execution_error": 0,
            "result_mismatch": 0,
            "sql_exact_match": 0  # 新增：SQL完全匹配的数量
        }
        
        # 详细结果
        detailed_results = []
        
        # 使用进度条对比每一条 SQL
        print(f"\nComparing SQL queries...")
        print(f"Total questions: {len(self.golden_data)}")
        
        for idx in tqdm(range(len(self.golden_data)), desc="Processing", unit="query"):
            golden_entry = self.golden_data[idx]
            question_id = golden_entry["question_id"]
            db_id = golden_entry["db_id"]
            golden_sql = golden_entry["SQL"]
            question = golden_entry["question"]
            
            # 获取数据库路径
            try:
                db_path = self.get_db_path(db_id)
            except FileNotFoundError as e:
                continue
            
            # 获取预测的 SQL
            sql1 = sqls_file1[idx] if idx < len(sqls_file1) else "SELECT"
            sql2 = sqls_file2[idx] if idx < len(sqls_file2) else "SELECT"
            
            # 检查 SQL 是否完全匹配（归一化比较）
            sql1_normalized = self._normalize_sql(sql1)
            sql2_normalized = self._normalize_sql(sql2)
            golden_normalized = self._normalize_sql(golden_sql)
            
            sql1_exact_match = (sql1_normalized == golden_normalized)
            sql2_exact_match = (sql2_normalized == golden_normalized)
            
            # 执行并对比 File 1
            result1 = self.execute_and_compare(sql1, golden_sql, db_path)
            file1_stats["total"] += 1
            if result1["is_match"]:
                file1_stats["correct"] += 1
            elif result1["error_message"]:
                file1_stats["execution_error"] += 1
            else:
                file1_stats["result_mismatch"] += 1
            if sql1_exact_match:
                file1_stats["sql_exact_match"] += 1
            
            # 执行并对比 File 2
            result2 = self.execute_and_compare(sql2, golden_sql, db_path)
            file2_stats["total"] += 1
            if result2["is_match"]:
                file2_stats["correct"] += 1
            elif result2["error_message"]:
                file2_stats["execution_error"] += 1
            else:
                file2_stats["result_mismatch"] += 1
            if sql2_exact_match:
                file2_stats["sql_exact_match"] += 1
            
            # 保存详细结果
            detailed_results.append({
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "golden_sql": golden_sql,
                "file1": {
                    "sql": sql1,
                    "is_match": result1["is_match"],
                    "exact_match": sql1_exact_match,
                    "error": result1["error_message"]
                },
                "file2": {
                    "sql": sql2,
                    "is_match": result2["is_match"],
                    "exact_match": sql2_exact_match,
                    "error": result2["error_message"]
                }
            })
        
        print("\n")  # 进度条结束后换行
        
        # 打印统计信息
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        
        print(f"\nFile 1: {file1_path}")
        print(f"  Total: {file1_stats['total']}")
        print(f"  Correct (Execution Match): {file1_stats['correct']} ({file1_stats['correct']/file1_stats['total']*100:.2f}%)")
        print(f"  SQL Exact Match: {file1_stats['sql_exact_match']} ({file1_stats['sql_exact_match']/file1_stats['total']*100:.2f}%)")
        print(f"  Execution Error: {file1_stats['execution_error']}")
        print(f"  Result Mismatch: {file1_stats['result_mismatch']}")
        
        print(f"\nFile 2: {file2_path}")
        print(f"  Total: {file2_stats['total']}")
        print(f"  Correct (Execution Match): {file2_stats['correct']} ({file2_stats['correct']/file2_stats['total']*100:.2f}%)")
        print(f"  SQL Exact Match: {file2_stats['sql_exact_match']} ({file2_stats['sql_exact_match']/file2_stats['total']*100:.2f}%)")
        print(f"  Execution Error: {file2_stats['execution_error']}")
        print(f"  Result Mismatch: {file2_stats['result_mismatch']}")
        
        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存统计信息
            stats_path = os.path.join(output_dir, "comparison_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "file1": {
                        "path": file1_path,
                        "stats": file1_stats
                    },
                    "file2": {
                        "path": file2_path,
                        "stats": file2_stats
                    }
                }, f, indent=4)
            print(f"\nStatistics saved to: {stats_path}")
            
            # 保存详细结果
            details_path = os.path.join(output_dir, "comparison_details.json")
            with open(details_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=4, ensure_ascii=False)
            print(f"Detailed results saved to: {details_path}")


def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent
    bird_dev_path = project_root / "data" / "bird" / "dev" / "dev.json"
    db_base_path = project_root / "data" / "bird" / "dev" / "dev_databases"
    
    file1_path = project_root / "compare" / "data" / "ori.txt"
    file2_path = project_root / "compare" / "data" / "llm_guided.txt"
    output_dir = project_root / "compare" / "results"
    
    # 检查文件是否存在
    if not bird_dev_path.exists():
        print(f"Error: BIRD dev.json not found at {bird_dev_path}")
        return
    
    if not file1_path.exists():
        print(f"Error: File 1 not found at {file1_path}")
        return
    
    if not file2_path.exists():
        print(f"Error: File 2 not found at {file2_path}")
        return
    
    # 创建对比器
    comparator = SQLComparator(str(bird_dev_path), str(db_base_path))
    
    # 执行对比
    comparator.compare_files(str(file1_path), str(file2_path), str(output_dir))


if __name__ == "__main__":
    main()
