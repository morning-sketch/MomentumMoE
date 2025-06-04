import json
import pandas as pd

def process_jsonl(file_path):
    solutions = []
    difficulties = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            solutions.append(data.get('solutions', []))
            difficulties.append(data.get('difficulty', None))



    # 创建DataFrame并保存为Parquet
    df = pd.DataFrame({
        'solution': solutions,     # 表头保持单数形式
        'difficulty': difficulties
    })
    df.to_parquet('test.parquet', engine='pyarrow')
process_jsonl("test.jsonl")
df=pd.read_parquet('test.parquet')
print(df.head())