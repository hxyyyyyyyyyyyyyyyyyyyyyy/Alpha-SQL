# import pickle
# res = pickle.load(open('/home/cbacoding/Alpha-SQL/results/llm_guided/Qwen2.5-Coder-7B-Instruct/1029_20:00(success)/sds/1063.pkl', 'rb'))
# print(res)

string = """
```json
{
    "reasoning": "We need to identify the relevant columns from both the 'client' and 'district' tables to answer the question. The question specifies conditions related to gender and region, which correspond to columns in these tables. We also need to ensure that we are using the correct function to calculate the average salary. Therefore, the next logical step is to identify the column values and functions needed for the SQL query.",
    "selected_action_number": 2
}
```
"""

import re
import json
from typing import Any
def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            normalized_value = re.sub(r"\s+", "", str(value))
            normalized_match = re.search(r"[-+]?\d+", normalized_value)
            if normalized_match:
                try:
                    return int(normalized_match.group(0))
                except (TypeError, ValueError):
                    pass
            return default

def _parse_selected_action_idx(response: str) -> int:
        # 1) 优先解析 ```json ... ``` 代码块
        json_match = re.search(r"```json\n(.*?)```(.*?)", response, flags=re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return _safe_int(result.get("selected_action_number", 0), 0) - 1
            except Exception:
                pass

        # 2) 再尝试从整段文本中直接匹配字段（兼容字符串/数字）
        selected_action_match = re.search(
            r'"selected_action_number"\s*:\s*"?(\d+)"?',
            response,
            flags=re.DOTALL,
        )
        if selected_action_match:
            return _safe_int(selected_action_match.group(1), 0) - 1

        return -1
    
print(_parse_selected_action_idx(string))