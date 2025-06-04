import pandas as pd
import json
# print(len(column_data))
unk=0
k_len=0
with open('tokenizer/vocab.json', 'r', encoding='utf-8') as file:
    vocab = json.load(file)


    df = pd.read_parquet('../APPS/train.parquet', columns=['solution'])
    column_data = df['solution'].values
    for i in column_data:
        k_list=i.split()
        k_len+=len(k_list)
        for j in k_list:
            if j not in vocab:
                unk+=1
    print(k_len)
    print(unk)

    # unk = 0
    # k_len = 0
    # df = pd.read_parquet('../codeforces-dataset/validation.parquet', columns=['solution'])
    # column_data = df['solution'].values
    # for i in column_data:
    #     k_list=i.split()
    #     k_len+=len(k_list)
    #     for j in k_list:
    #         if j not in vocab:
    #             unk+=1
    # print(k_len)
    # print(unk)

    unk = 0
    k_len = 0
    df = pd.read_parquet('../APPS/test.parquet', columns=['solution'])
    column_data = df['solution'].values
    for i in column_data:
        k_list=i.split()
        k_len+=len(k_list)
        for j in k_list:
            if j not in vocab:
                unk+=1
    print(k_len)
    print(unk)




# df = pd.read_parquet('test.parquet', columns=['code'])
# df = df.rename(columns={'code': 'solution'})
# df.to_parquet('test.parquet', engine='pyarrow')
# print(df.head())
# df = pd.read_parquet('train.parquet', columns=['code'])
# df = df.rename(columns={'code': 'solution'})
# df.to_parquet('train.parquet', engine='pyarrow')
# print(df.head())
# df = pd.read_parquet('validation.parquet', columns=['code'])
# df = df.rename(columns={'code': 'solution'})
# df.to_parquet('validation.parquet', engine='pyarrow')
# print(df.head())
