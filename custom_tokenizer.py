import os
import json
import torch
from transformers import PreTrainedTokenizer

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
        # 初始化tokenizer
        super().__init__(pad_token=pad_token, unk_token=unk_token, **kwargs)
        
        # 如果提供了词汇表，则使用它，否则初始化一个空的
        self.vocab = vocab if vocab is not None else {}
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
        # 这里我们不添加额外的特殊token，因为原始代码中没有这个逻辑
        return token_ids_0
    
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """返回特殊token的掩码"""
        return [0] * len(token_ids_0)
    
    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """创建token类型ID"""
        return [0] * len(token_ids_0)


def train_tokenizer(data_x, data_y, block_size):
    """训练tokenizer并返回实例"""
    # 初始化词汇表和tokenizer
    dictionary = {}
    tokenizer = ACCPreTrainedTokenizer()
    
    # 从数据中构建词汇表
    for item1, item2 in zip(data_x, data_y):
        tokens = item1.split()
        # 添加输入文本中的token
        for token in tokens:
            if token not in dictionary:
                dictionary[token] = len(dictionary)
        # 添加目标文本中的token
        if item2 not in dictionary:
            dictionary[item2] = len(dictionary)
    
    # 更新tokenizer的词汇表
    tokenizer.vocab = dictionary
    tokenizer.ids_to_tokens = {v: k for k, v in dictionary.items()}
    tokenizer.pad_token_id = dictionary.get(tokenizer.pad_token, 0)
    tokenizer.unk_token_id = dictionary.get(tokenizer.unk_token, 1)
    
    return tokenizer


def acc_tokenize_with_tokenizer(tokenizer, data_x, data_y, block_size):
    """使用自定义tokenizer处理数据，替代原有的acc_tokenize函数"""
    ids_x = []
    ids_y = []
    
    for item1, item2 in zip(data_x, data_y):
        tokens = tokenizer.tokenize(item1)
        if len(tokens) > block_size:
            continue
        
        # 填充处理
        padding_length = block_size - len(tokens)
        tokens = [tokenizer.pad_token] * padding_length + tokens
        
        # 转换为ID
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        ids_x.extend(input_ids)
        
        # 处理目标token
        target_id = tokenizer.convert_tokens_to_ids([item2])[0]
        ids_y.extend([-100] * (block_size - 1) + [target_id])
    
    return torch.LongTensor(ids_x), torch.LongTensor(ids_y)    