import pandas as pd
import numpy as np
import json
# 读取Parquet文件
df = pd.read_parquet('MBPP/test.parquet')  # 替换为实际文件路径

# 合并所有solution列的代码
all_code = " ".join(df['solution'].astype(str).tolist())

# 按空格分割成单词列表
tokens = all_code.split()

# 按64个单词一组划分
chunk_size = 64
chunks = {
    f"chunk_{i+1}": {
        "size": len(chunk),
        "content": " ".join(chunk)
    }
    for i, chunk in enumerate([tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)])
}

# 写入JSON文件
with open('MBPP.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print("数据已成功写入output.json文件")

