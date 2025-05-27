import pandas as pd
import json
# 读取特定列
df = pd.read_parquet('test.parquet', columns=['code'])

# 获取该列的所有数据
column_data = df['code'].values
# print(len(column_data))
unk=0
k_len=0
with open('tokenizer/vocab.json', 'r', encoding='utf-8') as file:
    vocab = json.load(file)
    for i in column_data:
        k_list=i.split()
        k_len+=len(k_list)
        for j in k_list:
            if j not in vocab:
                unk+=1
print(k_len)
print(unk)

