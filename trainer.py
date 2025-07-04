import os, sys
import argparse
import math, random
import torch
import tqdm

from custom_gates import *
from data import ACCPreTrainedTokenizer
import torch.nn.functional as F


def has_module_path(model, path):
    """检查模型是否包含指定路径的子模块"""
    parts = path.split('.')
    current = model

    for part in parts:
        if not hasattr(current, part):
            return False
        current = getattr(current, part)

    return True


# 使用示例



def get_cosine_regularization(model):
    """计算第16个线性层与其他线性层的余弦相似度正则化项"""
    layer_nums=-1
    if has_module_path(model, "module.layers[-1].smoe"):
        layer_nums=-1
    else:
        layer_nums = -2
    weight_htoh4 = model.module.layers[layer_nums].smoe.experts.htoh4.weight.data  # 第16个线性层
    weight_h4toh = model.module.layers[layer_nums].smoe.experts.h4toh.weight.data
    weight_gate = model.module.layers[layer_nums].smoe.gate.gate.weight.data

    # 展平权重（保留专家维度，其他维度展平）
    flat_htoh4 = weight_htoh4.view(weight_htoh4.size(0), -1)  # 形状: (16, 352*352)
    flat_h4toh = weight_h4toh.view(weight_h4toh.size(0), -1)  # 形状: (16, 352*352)

    # 提取目标专家（第16个）的权重向量
    target_htoh4 = flat_htoh4[-1]  # 形状: (352*352,)
    target_h4toh = flat_h4toh[-1]  # 形状: (352*352,)

    # 其他专家的权重向量（排除目标专家）
    others_htoh4 = flat_htoh4[:-1]
    others_h4toh = flat_h4toh[:-1]# 形状: (15, 352*352)
    others_htoh4=others_htoh4[0]
    others_h4toh=others_h4toh[0]
    # print("type(flat_h4toh):",type(flat_h4toh))
    # 计算余弦相似度（自动广播计算）
    cos_sim_htoh4 = F.cosine_similarity(
        target_htoh4.unsqueeze(0),  # 增加维度 -> (1, D)
        others_htoh4,  # (15, D)
        dim=1
    )  # 返回形状: (15,)

    cos_sim_h4toh = F.cosine_similarity(
        target_h4toh.unsqueeze(0),
        others_h4toh,
        dim=1
    )  # 形状: (15,)

    # 对相似度取平均作为正则项
    reg_loss_htoh4 = cos_sim_htoh4.mean()
    reg_loss_h4toh = cos_sim_h4toh.mean()

    # 合并两个线性层的正则损失
    total_reg_loss = (reg_loss_htoh4 + reg_loss_h4toh)/2
    simple_loss = torch.mean(torch.norm(weight_gate, dim=0))
    return total_reg_loss + simple_loss
def _train_step(model, load_balance, X, Y, h_cache, eval_only, loss_div=1):
    """Single training step."""

    out, h_cache = model(X, h_cache)
    out = out.view(-1, out.size(-1))

    # 获取预测类别 (out中最大概率的索引)
    preds = out.argmax(dim=1)
    # 展平Y并过滤掉-100
    y_flat = Y.view(-1)
    mask = y_flat != -100

    # 计算统计值
    trick = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', 'A': '1', 'B': '2', 'C': '3', 'D': '4'}
    tokenizer = ACCPreTrainedTokenizer._from_pretrained("my_custom_tokenizer")
    correct=0
    for i,j in zip(preds[mask],y_flat[mask]):
        if i==j or(tokenizer._convert_id_to_token(i) in trick and trick[tokenizer._convert_id_to_token(i)]==tokenizer._convert_id_to_token(j))  or (tokenizer._convert_id_to_token(j) in trick and trick[tokenizer._convert_id_to_token(j)]==tokenizer._convert_id_to_token(i)):
            correct+=1
    # correct = (preds[mask] == y_flat[mask]).sum().item()
    # 计算比值 (避免除以0)
    ratio = correct / (mask.sum().item())

    loss = torch.nn.functional.nll_loss(out, y_flat)
    loss_value = loss.item() / loss_div
    reg = 0.01
    loss_value = loss_value + reg * get_cosine_regularization(model)

    if not eval_only:
        # loss term from adaptive-span
        if model.module.layers[0].attn.attn.adapt_span_enabled:
            loss += sum(
                model.module.layers[layer_i].attn.attn.adaptive_span.get_loss()
                for layer_i in range(model.module.attn_layer_count)
            )

        if load_balance > 0:
            balance_loss = 0
            for name, m in model.named_modules():
                if isinstance(m, CustomNaiveGate_Balance_SMoE) or isinstance(
                    m, CustomNaiveGate_Balance_XMoE
                ):
                    if m.loss is not None:
                        balance_loss += m.loss
            loss += load_balance * balance_loss
        (loss / loss_div).backward(retain_graph=True)
    return loss_value, h_cache,ratio


def _train_batch(
    model, load_balance, optimizer, scheduler, X, Y, h_cache, eval_only, batch_split
):
    """Train on a batch."""

    optimizer.zero_grad()

    if batch_split == 1:
        # process a batch in a single step (default behaviour)
        loss_value, h_cache,radio_value = _train_step(model, load_balance, X, Y, h_cache, eval_only)
    else:
        # split a batch into multiple pieces that each can fit in memory
        assert X.size(0) % batch_split == 0
        split_size = X.size(0) // batch_split
        loss_value = 0
        radio_value=0
        h_cache_list = []
        for split_ind in range(batch_split):
            split_slice = slice(split_ind * split_size, (split_ind + 1) * split_size)
            split_h_cache = [h[split_slice, :, :] for h in h_cache]
            split_loss_value, split_h_cache,radio = _train_step(
                model,
                load_balance,
                X[split_slice, :],
                Y[split_slice],
                split_h_cache,
                eval_only,
                batch_split,
            )
            loss_value += split_loss_value
            radio_value+=radio
            h_cache_list.append(split_h_cache)
        h_cache = [
            torch.cat([h_cache_list[i][l] for i in range(batch_split)], dim=0)
            for l in range(len(h_cache))
        ]
        radio_value/=batch_split
    if not eval_only:
        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        # make sure span parameters are in a correct range
        if model.module.layers[0].attn.attn.adapt_span_enabled:
            for layer in model.module.layers:
                if layer.use_attn:
                    layer.attn.attn.adaptive_span.clamp_param()
    return loss_value, h_cache,radio_value


def train_iteration(
    model,
    load_balance,
    optimizer,
    scheduler,
    data_x,
    data_y,
    nb_batches_per_iter,
    block_size,
    eval_only,
    train_pos,
    h_cache,
    batch_split,
    checkpoint_path,
):
    """Single training iteration."""
    if eval_only:
        model.eval()
    else:
        model.train()

    nb_batches_per_iter_max = nb_batches_per_iter
    if eval_only:
        # eval on fewer batches during training for speed-up
        nb_batches_per_iter_max = max(1, nb_batches_per_iter // 10)
        nb_batches_per_iter_max = min(
            nb_batches_per_iter_max, math.ceil(data_x.size(1) / block_size)
        )

    loss_all = 0
    actual_nb_batches_per_iter = 0
    radio_all=0
    for _ in tqdm.tqdm(range(nb_batches_per_iter_max)):
        actual_nb_batches_per_iter += 1
        X = data_x[:, train_pos : train_pos + block_size].contiguous()
        Y = data_y[:, train_pos: train_pos + block_size].contiguous()

        loss, h_cache,radio = _train_batch(
            model=model,
            load_balance=load_balance,
            optimizer=optimizer,
            scheduler=scheduler,
            X=X,
            Y=Y,
            h_cache=h_cache,
            eval_only=eval_only,
            batch_split=batch_split,
        )
        loss_all += loss
        train_pos += block_size
        radio_all+=radio
        if train_pos >= data_x.size(1) - block_size:
            # reached the end. randomize the offset to reduce overfitting
            train_pos = random.randrange(block_size)
            # reset the cache
            for h in h_cache:
                h.fill_(0)
    radio_all/=actual_nb_batches_per_iter
    loss_all = loss_all / actual_nb_batches_per_iter
    return loss_all, train_pos, h_cache,radio_all


# do full evaluation
def full_eval(model, optimizer, scheduler, data_x,data_y, block_size, hidden_size, batch_split,device):
    model.eval()
    train_pos = 0
    nb_batches_per_iter_max = math.ceil(data_x.size(1) / block_size)
    h_cache = [
        torch.zeros(
            data_x.size(0),
            model.module.layers[layer_i].attn.attn.get_cache_size(),
            hidden_size,
        ).to(device)
        for layer_i in range(model.module.attn_layer_count)
    ]

    loss_all = 0
    radio_all=0
    actual_nb_batches_per_iter = 0
    for _ in tqdm.tqdm(range(nb_batches_per_iter_max)):
        actual_nb_batches_per_iter += 1
        X = data_x[:, train_pos : train_pos + block_size].contiguous()
        Y = data_y[:, train_pos : train_pos + block_size].contiguous()

        loss, h_cache ,radio= _train_batch(
            model=model,
            load_balance=0,
            optimizer=optimizer,
            scheduler=scheduler,
            X=X,
            Y=Y,
            h_cache=h_cache,
            eval_only=True,
            batch_split=batch_split,
        )
        loss_all += loss
        train_pos += block_size
        radio_all+=radio
        if train_pos >= data_x.size(1) - block_size:
            # Skip the remaining tokens as it can't make a whole block.
            # An effect on performance should be negligable for a large data.
            break
    radio_all/=actual_nb_batches_per_iter
    loss_all = loss_all / actual_nb_batches_per_iter
    return loss_all,radio_all
