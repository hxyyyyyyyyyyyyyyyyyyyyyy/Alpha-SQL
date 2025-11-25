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