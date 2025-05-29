import os, sys
import argparse
import math, random
import torch
import tqdm
import json
from transformers import PreTrainedTokenizer
import pandas as pd


class ACCPreTrainedTokenizer(PreTrainedTokenizer):
    """自定义tokenizer类，继承自Hugging Face的PreTrainedTokenizer"""

    # 定义特殊token
    pad_token = "<pad>"
    unk_token = "<unk>"
    additional_special_tokens = []

    def __init__(
            self,
            vocab=None,
            pad_token="<pad>",
            unk_token="<unk>",
            **kwargs
    ):
        # 确保vocab始终是字典
        if vocab is None:
            self.vocab = {}
        else :
            self.vocab = vocab
        super().__init__(pad_token=pad_token, unk_token=unk_token, **kwargs)

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        # 设置特殊token的ID
        self.pad_token_id = self.vocab.get(pad_token, 0)
        self.unk_token_id = self.vocab.get(unk_token, 1)

    @property
    def vocab_size(self) -> int:
        """返回词汇表大小"""
        return len(self.vocab)

    def _tokenize(self, text: str):
        """将文本拆分为token列表"""
        return text.split()

    def _convert_token_to_id(self, token):
        """将token转换为ID"""
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        """将ID转换为token"""
        return self.ids_to_tokens.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        """保存词汇表到文件"""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        # 构建完整的文件名
        if filename_prefix is not None:
            vocab_file = os.path.join(save_directory, f"{filename_prefix}-vocab.json")
        else:
            vocab_file = os.path.join(save_directory, "vocab.json")

        # 保存词汇表
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        return (vocab_file,)

    @classmethod
    def from_vocabulary(cls, vocab, **kwargs):
        """从词汇表创建tokenizer实例"""
        return cls(vocab=vocab, **kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """添加特殊token到输入中"""
        return token_ids_0

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """返回特殊token的掩码"""
        return [0] * len(token_ids_0)

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """创建token类型ID"""
        return [0] * len(token_ids_0)

    def get_vocab(self):
        """返回完整词汇表字典"""
        return self.vocab.copy()

    def __getstate__(self):
        """序列化tokenizer时调用"""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, d):
        """反序列化tokenizer时调用"""
        self.__dict__ = d
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        """从保存的文件加载tokenizer"""
        # 获取保存目录
        if os.path.isdir(pretrained_model_name_or_path):
            vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
        else:
            raise ValueError(f"未找到保存的tokenizer目录: {pretrained_model_name_or_path}")

        # 加载词汇表
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"未找到词汇表文件: {vocab_file}")

        # 加载配置（从kwargs中获取或使用默认值）
        config = kwargs.pop("config", None)
        if config and hasattr(config, "tokenizer_class") and config.tokenizer_class != cls.__name__:
            raise ValueError(
                f"配置中的tokenizer类({config.tokenizer_class})与当前类({cls.__name__})不匹配"
            )

        # 创建tokenizer实例
        return cls(vocab=vocab, *init_inputs, **kwargs)



def _tokenize(text_path, tokenizer):
    """Tokenizes a text file."""
    print("Tokenizing {}".format(text_path))
    assert os.path.exists(text_path)
    dictionary_to_update=tokenizer.vocab
    nb_tokens_in_dictionary = len(dictionary_to_update)

    # Count nb of tokens in text and update the dictionary
    with open(text_path, "r", encoding="utf8") as f:
        for line in f:
            tokens = line.split() + ["<eos>"]
            for token in tokens:
                if token not in dictionary_to_update:
                    dictionary_to_update[token] = nb_tokens_in_dictionary
                    nb_tokens_in_dictionary += 1
    tokenizer.vocab = dictionary_to_update
    tokenizer.ids_to_tokens = {v: k for k, v in dictionary_to_update.items()}
    tokenizer.pad_token_id = dictionary.get(tokenizer.pad_token, 0)
    tokenizer.unk_token_id = dictionary.get(tokenizer.unk_token, 1)
    # Assign to each token its identifier
    ids = []
    with open(text_path, "r", encoding="utf8") as f:
        for line in f:
            tokens = line.split() + ["<eos>"]
            for token in tokens:
                ids.append(dictionary_to_update[token])
    ids = torch.LongTensor(ids)
    return ids

def _tokenize_parquet(text_path, tokenizer):
    """Tokenizes a text file."""
    print("Tokenizing {}".format(text_path))
    assert os.path.exists(text_path)
    dictionary_to_update=tokenizer.vocab
    ids = []
    df = pd.read_parquet(text_path, columns=['code'])
    column_data = df['code'].values
    for i in column_data:
        k_list=i.split()+ ["<eos>"]
        for token in k_list:
            if token not in dictionary_to_update:
                ids.append(dictionary_to_update["<unk>"])
            else:
                ids.append(dictionary_to_update[token])
    ids = torch.LongTensor(ids)
    return ids

def _tokenize_json(text_path, tokenizer):
    """Tokenizes a text file."""
    print("Tokenizing {}".format(text_path))
    assert os.path.exists(text_path)
    dictionary_to_update=tokenizer.vocab
    ids = []
    with open(text_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # 提取所有sql字段内容
        sql_contents = [i['sql'] for i in data]
    for i in sql_contents:
        i=str(i)
        k_list=i.split()+ ["<eos>"]
        for token in k_list:
            if token not in dictionary_to_update:
                ids.append(dictionary_to_update["<unk>"])
            else:
                ids.append(dictionary_to_update[token])
    ids = torch.LongTensor(ids)
    return ids

#spider
class Corpus:
    def __init__(self, data_path):
        self.tokenizer = ACCPreTrainedTokenizer._from_pretrained("my_custom_tokenizer")
        self.train = _tokenize_json(
            tokenizer=self.tokenizer,
            text_path=os.path.join(data_path, "train.json"),
        )
        self.valid = _tokenize_json(
            tokenizer=self.tokenizer,
            text_path=os.path.join(data_path, "validation.json"),
        )
        self.test = _tokenize_json(
            tokenizer=self.tokenizer,
            text_path=os.path.join(data_path, "test.json"),
        )

    # tokenizer.save_pretrained("my_custom_tokenizer")
    @property
    def vocab_size(self):
        return len(self.tokenizer.vocab)


#mbpp
# class Corpus:
#     def __init__(self, data_path):
#         self.tokenizer = ACCPreTrainedTokenizer._from_pretrained("my_custom_tokenizer")
#         self.train = _tokenize_parquet(
#             tokenizer=self.tokenizer,
#             text_path=os.path.join(data_path, "train.parquet"),
#         )
#         self.valid = _tokenize_parquet(
#             tokenizer=self.tokenizer,
#             text_path=os.path.join(data_path, "validation.parquet"),
#         )
#         self.test = _tokenize_parquet(
#             tokenizer=self.tokenizer,
#             text_path=os.path.join(data_path, "test.parquet"),
#         )
#
#     # tokenizer.save_pretrained("my_custom_tokenizer")
#     @property
#     def vocab_size(self):
#         return len(self.tokenizer.vocab)
#wiki103
# class Corpus:
#     def __init__(self, data_path):
#         tokenizer = ACCPreTrainedTokenizer()
#         self.train = _tokenize(
#             tokenizer=tokenizer,
#             text_path=os.path.join(data_path, "train.txt"),
#         )
#         self.valid = _tokenize(
#             tokenizer=tokenizer,
#             text_path=os.path.join(data_path, "valid.txt"),
#         )
#         self.test = _tokenize(
#             tokenizer=tokenizer,
#             text_path=os.path.join(data_path, "test.txt"),
#         )
#
#         tokenizer.save_pretrained("my_custom_tokenizer")
#     @property
#     def vocab_size(self):
#         return len(self._dictionary)


def _batchify(data_tensor, batch_size):
    nb_batches = data_tensor.size(0) // batch_size
    # trim away some tokens to make whole batches
    data_tensor = data_tensor.narrow(0, 0, nb_batches * batch_size)
    data_tensor = data_tensor.view(batch_size, -1).contiguous()
    return data_tensor


def _build_corpus(data_path, env_params, data_name=None):
    # save the corpus to a file so that it's faster next time
    corpus_path = os.path.join(data_path, "corpus_a1.pt")
    if os.path.exists(corpus_path):
        print("Loading an existing corpus file from {}".format(corpus_path))
        corpus = torch.load(corpus_path)
    else:
        print("Creating a corpus file at {}".format(corpus_path))
        if env_params["distributed"]:
            # only one process need to create a corpus file
            if env_params["rank"] == 0:
                corpus = Corpus(data_path)
                torch.save(corpus, corpus_path)
                # sync with other processes
                torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
            else:
                print("Waiting rank0 to create a corpus file.")
                # sync with rank0
                torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
                corpus = torch.load(corpus_path)
        else:
            corpus = Corpus(data_path)
            torch.save(corpus, corpus_path)
    return corpus


def _get_train_val_test_data(corpus, batch_size):
    return [
        _batchify(corpus.train, batch_size),
        _batchify(corpus.valid, batch_size),
        _batchify(corpus.test, batch_size),
    ]


def get_train_val_test_data(data_params, env_params, batch_size, device):
    corpus = _build_corpus(**data_params, env_params=env_params)
    data_params["vocab_size"] = corpus.vocab_size
    train_data, val_data, test_data = _get_train_val_test_data(
        corpus=corpus, batch_size=batch_size
    )

    if env_params["distributed"]:
        # split the data into equal parts
        assert batch_size % env_params["world_size"] == 0
        device_batch_size = batch_size // env_params["world_size"]
        slice_data = slice(
            device_batch_size * env_params["rank"],
            device_batch_size * (env_params["rank"] + 1),
        )
        train_data = train_data[slice_data]
        val_data = val_data[slice_data]
        test_data = test_data[slice_data]

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    return train_data, val_data, test_data
