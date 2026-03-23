# Result Folder Comparison

Compared folders:
- origin1: results/origin1/Qwen2.5-Coder-7B-Instruct/bird/dev
- llm_guided_fix_1_1: results/llm_guided_fix_1_1/Qwen2.5-Coder-7B-Instruct/bird/dev
- llm_guided1_3: results/llm_guided1_3/Qwen2.5-Coder-7B-Instruct/bird/dev

## 1) Basic Scale

| folder | pkl_files | loaded_files | total_paths | total_nodes | avg_nodes_per_path |
|---|---:|---:|---:|---:|---:|
| origin1 | 147 | 147 | 222 | 1298 | 5.847 |
| llm_guided_fix_1_1 | 147 | 147 | 147 | 907 | 6.170 |
| llm_guided1_3 | 147 | 147 | 147 | 944 | 6.422 |

## 2) Node Trigger Frequency

| node_type | origin1 count(%) | llm_guided_fix_1_1 count(%) | llm_guided1_3 count(%) |
|---|---:|---:|---:|
| end | 222 (17.10%) | 147 (16.21%) | 147 (15.57%) |
| identify_column_functions | 174 (13.41%) | 147 (16.21%) | 144 (15.25%) |
| identify_column_values | 167 (12.87%) | 147 (16.21%) | 142 (15.04%) |
| rephrase_question | 61 (4.70%) | 4 (0.44%) | 12 (1.27%) |
| root | 222 (17.10%) | 147 (16.21%) | 147 (15.57%) |
| schema_selection | 155 (11.94%) | 145 (15.99%) | 134 (14.19%) |
| sql_generation | 222 (17.10%) | 147 (16.21%) | 147 (15.57%) |
| sql_revision | 75 (5.78%) | 23 (2.54%) | 71 (7.52%) |

## 3) Node Order Pattern (Top 10 Path Sequences)

### origin1
| rank | node_sequence | count | share_in_paths |
|---:|---|---:|---:|
| 1 | root -> identify_column_functions -> identify_column_values -> schema_selection -> sql_generation -> end | 16 | 7.21% |
| 2 | root -> schema_selection -> identify_column_values -> identify_column_functions -> sql_generation -> end | 12 | 5.41% |
| 3 | root -> identify_column_functions -> schema_selection -> identify_column_values -> sql_generation -> end | 11 | 4.95% |
| 4 | root -> schema_selection -> identify_column_values -> identify_column_functions -> sql_generation -> sql_revision -> end | 8 | 3.60% |
| 5 | root -> identify_column_functions -> identify_column_values -> sql_generation -> end | 7 | 3.15% |
| 6 | root -> identify_column_functions -> sql_generation -> end | 7 | 3.15% |
| 7 | root -> identify_column_values -> identify_column_functions -> schema_selection -> sql_generation -> end | 7 | 3.15% |
| 8 | root -> identify_column_values -> sql_generation -> end | 7 | 3.15% |
| 9 | root -> schema_selection -> identify_column_functions -> identify_column_values -> sql_generation -> end | 7 | 3.15% |
| 10 | root -> schema_selection -> sql_generation -> end | 7 | 3.15% |

### llm_guided_fix_1_1
| rank | node_sequence | count | share_in_paths |
|---:|---|---:|---:|
| 1 | root -> schema_selection -> identify_column_values -> identify_column_functions -> sql_generation -> end | 86 | 58.50% |
| 2 | root -> identify_column_values -> identify_column_functions -> schema_selection -> sql_generation -> end | 21 | 14.29% |
| 3 | root -> schema_selection -> identify_column_values -> identify_column_functions -> sql_generation -> sql_revision -> end | 17 | 11.56% |
| 4 | root -> identify_column_values -> schema_selection -> identify_column_functions -> sql_generation -> end | 11 | 7.48% |
| 5 | root -> identify_column_values -> identify_column_functions -> schema_selection -> sql_generation -> sql_revision -> end | 5 | 3.40% |
| 6 | root -> rephrase_question -> identify_column_values -> identify_column_functions -> schema_selection -> sql_generation -> end | 3 | 2.04% |
| 7 | root -> identify_column_values -> identify_column_functions -> sql_generation -> end | 2 | 1.36% |
| 8 | root -> identify_column_values -> schema_selection -> identify_column_functions -> sql_generation -> sql_revision -> end | 1 | 0.68% |
| 9 | root -> rephrase_question -> identify_column_values -> schema_selection -> identify_column_functions -> sql_generation -> end | 1 | 0.68% |

### llm_guided1_3
| rank | node_sequence | count | share_in_paths |
|---:|---|---:|---:|
| 1 | root -> schema_selection -> identify_column_values -> identify_column_functions -> sql_generation -> sql_revision -> end | 28 | 19.05% |
| 2 | root -> schema_selection -> identify_column_values -> identify_column_functions -> sql_generation -> end | 25 | 17.01% |
| 3 | root -> identify_column_values -> identify_column_functions -> schema_selection -> sql_generation -> end | 15 | 10.20% |
| 4 | root -> identify_column_values -> identify_column_functions -> schema_selection -> sql_generation -> sql_revision -> end | 12 | 8.16% |
| 5 | root -> identify_column_functions -> schema_selection -> identify_column_values -> sql_generation -> end | 10 | 6.80% |
| 6 | root -> identify_column_values -> schema_selection -> identify_column_functions -> sql_generation -> end | 8 | 5.44% |
| 7 | root -> identify_column_values -> schema_selection -> identify_column_functions -> sql_generation -> sql_revision -> end | 6 | 4.08% |
| 8 | root -> identify_column_functions -> identify_column_values -> schema_selection -> sql_generation -> sql_revision -> end | 5 | 3.40% |
| 9 | root -> identify_column_functions -> schema_selection -> identify_column_values -> sql_generation -> sql_revision -> end | 4 | 2.72% |
| 10 | root -> identify_column_values -> identify_column_functions -> sql_generation -> end | 4 | 2.72% |

## 4) Action Path Frequency (Top 10)

### origin1
| rank | action_path | count | share_in_paths |
|---:|---|---:|---:|
| 1 | IdentifyColumnFunctionsAction -> IdentifyColumnValuesAction -> SchemaSelectionAction -> SQLGenerationAction -> EndAction | 16 | 7.21% |
| 2 | SchemaSelectionAction -> IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> EndAction | 12 | 5.41% |
| 3 | IdentifyColumnFunctionsAction -> SchemaSelectionAction -> IdentifyColumnValuesAction -> SQLGenerationAction -> EndAction | 11 | 4.95% |
| 4 | SchemaSelectionAction -> IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> SQLRevisionAction -> EndAction | 8 | 3.60% |
| 5 | IdentifyColumnFunctionsAction -> IdentifyColumnValuesAction -> SQLGenerationAction -> EndAction | 7 | 3.15% |
| 6 | IdentifyColumnFunctionsAction -> SQLGenerationAction -> EndAction | 7 | 3.15% |
| 7 | IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SchemaSelectionAction -> SQLGenerationAction -> EndAction | 7 | 3.15% |
| 8 | IdentifyColumnValuesAction -> SQLGenerationAction -> EndAction | 7 | 3.15% |
| 9 | SchemaSelectionAction -> IdentifyColumnFunctionsAction -> IdentifyColumnValuesAction -> SQLGenerationAction -> EndAction | 7 | 3.15% |
| 10 | SchemaSelectionAction -> SQLGenerationAction -> EndAction | 7 | 3.15% |

### llm_guided_fix_1_1
| rank | action_path | count | share_in_paths |
|---:|---|---:|---:|
| 1 | SchemaSelectionAction -> IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> EndAction | 86 | 58.50% |
| 2 | IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SchemaSelectionAction -> SQLGenerationAction -> EndAction | 21 | 14.29% |
| 3 | SchemaSelectionAction -> IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> SQLRevisionAction -> EndAction | 17 | 11.56% |
| 4 | IdentifyColumnValuesAction -> SchemaSelectionAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> EndAction | 11 | 7.48% |
| 5 | IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SchemaSelectionAction -> SQLGenerationAction -> SQLRevisionAction -> EndAction | 5 | 3.40% |
| 6 | RaphraseQuestionAction -> IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SchemaSelectionAction -> SQLGenerationAction -> EndAction | 3 | 2.04% |
| 7 | IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> EndAction | 2 | 1.36% |
| 8 | IdentifyColumnValuesAction -> SchemaSelectionAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> SQLRevisionAction -> EndAction | 1 | 0.68% |
| 9 | RaphraseQuestionAction -> IdentifyColumnValuesAction -> SchemaSelectionAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> EndAction | 1 | 0.68% |

### llm_guided1_3
| rank | action_path | count | share_in_paths |
|---:|---|---:|---:|
| 1 | SchemaSelectionAction -> IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> SQLRevisionAction -> EndAction | 28 | 19.05% |
| 2 | SchemaSelectionAction -> IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> EndAction | 25 | 17.01% |
| 3 | IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SchemaSelectionAction -> SQLGenerationAction -> EndAction | 15 | 10.20% |
| 4 | IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SchemaSelectionAction -> SQLGenerationAction -> SQLRevisionAction -> EndAction | 12 | 8.16% |
| 5 | IdentifyColumnFunctionsAction -> SchemaSelectionAction -> IdentifyColumnValuesAction -> SQLGenerationAction -> EndAction | 10 | 6.80% |
| 6 | IdentifyColumnValuesAction -> SchemaSelectionAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> EndAction | 8 | 5.44% |
| 7 | IdentifyColumnValuesAction -> SchemaSelectionAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> SQLRevisionAction -> EndAction | 6 | 4.08% |
| 8 | IdentifyColumnFunctionsAction -> IdentifyColumnValuesAction -> SchemaSelectionAction -> SQLGenerationAction -> SQLRevisionAction -> EndAction | 5 | 3.40% |
| 9 | IdentifyColumnFunctionsAction -> SchemaSelectionAction -> IdentifyColumnValuesAction -> SQLGenerationAction -> SQLRevisionAction -> EndAction | 4 | 2.72% |
| 10 | IdentifyColumnValuesAction -> IdentifyColumnFunctionsAction -> SQLGenerationAction -> EndAction | 4 | 2.72% |

## 5) Position-wise Node Distribution

### origin1
- step 0: root: 222 (100.0%)
- step 1: identify_column_functions: 69 (31.1%), rephrase_question: 61 (27.5%), schema_selection: 46 (20.7%), identify_column_values: 38 (17.1%), sql_generation: 8 (3.6%)
- step 2: identify_column_values: 76 (34.2%), schema_selection: 53 (23.9%), identify_column_functions: 51 (23.0%), sql_generation: 34 (15.3%), end: 6 (2.7%)
- step 3: sql_generation: 49 (22.7%), schema_selection: 45 (20.8%), identify_column_values: 44 (20.4%), identify_column_functions: 42 (19.4%), end: 26 (12.0%)
- step 4: sql_generation: 99 (52.1%), end: 42 (22.1%), sql_revision: 17 (8.9%), identify_column_functions: 12 (6.3%), schema_selection: 11 (5.8%)
- step 5: end: 82 (55.4%), sql_revision: 34 (23.0%), sql_generation: 32 (21.6%)
- step 6: end: 54 (81.8%), sql_revision: 12 (18.2%)
- step 7: end: 12 (100.0%)

### llm_guided_fix_1_1
- step 0: root: 147 (100.0%)
- step 1: schema_selection: 103 (70.1%), identify_column_values: 40 (27.2%), rephrase_question: 4 (2.7%)
- step 2: identify_column_values: 107 (72.8%), identify_column_functions: 28 (19.0%), schema_selection: 12 (8.2%)
- step 3: identify_column_functions: 118 (80.3%), schema_selection: 27 (18.4%), sql_generation: 2 (1.4%)
- step 4: sql_generation: 141 (95.9%), schema_selection: 3 (2.0%), end: 2 (1.4%), identify_column_functions: 1 (0.7%)
- step 5: end: 118 (81.4%), sql_revision: 23 (15.9%), sql_generation: 4 (2.8%)
- step 6: end: 27 (100.0%)

### llm_guided1_3
- step 0: root: 147 (100.0%)
- step 1: schema_selection: 59 (40.1%), identify_column_values: 49 (33.3%), identify_column_functions: 25 (17.0%), rephrase_question: 12 (8.2%), sql_generation: 2 (1.4%)
- step 2: identify_column_values: 64 (43.5%), identify_column_functions: 41 (27.9%), schema_selection: 39 (26.5%), end: 2 (1.4%), sql_generation: 1 (0.7%)
- step 3: identify_column_functions: 70 (48.3%), schema_selection: 35 (24.1%), identify_column_values: 26 (17.9%), sql_generation: 13 (9.0%), sql_revision: 1 (0.7%)
- step 4: sql_generation: 119 (82.1%), end: 8 (5.5%), identify_column_functions: 8 (5.5%), sql_revision: 6 (4.1%), identify_column_values: 3 (2.1%)
- step 5: end: 67 (48.9%), sql_revision: 58 (42.3%), sql_generation: 12 (8.8%)
- step 6: end: 64 (91.4%), sql_revision: 6 (8.6%)
- step 7: end: 6 (100.0%)

## 6) High-level Delta vs origin1

### llm_guided_fix_1_1 vs origin1
- total_paths: 147 vs 222 (-75)
- avg_nodes_per_path: 6.170 vs 5.847 (+0.323)
- node ratio delta (percentage points):
  - end: -0.90 pp
  - identify_column_functions: +2.80 pp
  - identify_column_values: +3.34 pp
  - rephrase_question: -4.26 pp
  - root: -0.90 pp
  - schema_selection: +4.05 pp
  - sql_generation: -0.90 pp
  - sql_revision: -3.24 pp

### llm_guided1_3 vs origin1
- total_paths: 147 vs 222 (-75)
- avg_nodes_per_path: 6.422 vs 5.847 (+0.575)
- node ratio delta (percentage points):
  - end: -1.53 pp
  - identify_column_functions: +1.85 pp
  - identify_column_values: +2.18 pp
  - rephrase_question: -3.43 pp
  - root: -1.53 pp
  - schema_selection: +2.25 pp
  - sql_generation: -1.53 pp
  - sql_revision: +1.74 pp
