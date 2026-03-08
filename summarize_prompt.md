请在summarize_data.py中写一个脚本，读取results文件夹里面包含pkl文件的文件夹，在文件夹的同级目录总结一份pkl文件的内容。
要求：
1. 生成一份txt文档，按照id的顺序提取出END_NODE中的final_sql字段，并且只保留该字段
2. 生成一份json文档，用id作为键值，用每次生成的路径作为value，格式如下：
```json
"1":{
    "1":[START_NODE, ..., END_NODE],
    "2":[START_NODE, ..., END_NODE],
    ...
},
"2":{
    ...
},
...
```
3.生成一份md文档，总结各种路径的总数，格式如下：
[START_NODE, ..., END_NODE]: 100,
[START_NODE, ..., END_NODE]: 200,
...

请实现一个独立组件，通过llm_selector选择多条路径，之后筛选出SQL能够成功执行的高质量路径，并将这些路径作为遗传算法的初始路径。最后通过遗传算法产生足量的路径。要求llm_selector产生的路径数以及最终路径数可变