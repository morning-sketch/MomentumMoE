# 第一份工作完整笔记：CausalMoE

> 用途：给后续工作（第二份工作）快速了解第一份工作的研究问题、方法细节、代码位置、实验协议、已确认口径和已知坑。
> 论文源文件：`D:/GitHub/AAAI_HQA/anonymous-submission-latex-2026.tex`
> 论文题目：*Causal-aware Adaptive Routing Mechanism for Mixture of Experts*
> 主方法代码版本：`8a2f37c`（`deepseekmoe+loss` 线）
> 整理后的干净分支：`first-work-clean`
> 最后更新：2026-10-18

---

## 0. 一句话总结

CausalMoE 在最后一个 MoE 层用压缩块注意力构建 token 因果图，通过 CLEANN 找到关键块的最小解释集，把解释集块内的全部 token 额外路由给一个专用的因果专家；普通专家仍做 Top-2；因果专家与普通专家输出用可学习 sigmoid 门控融合；最后一层加 CAL，把因果专家与 15 个普通专家的参数余弦相似度压低，以提升路由可解释性和专家专业化。

第一份工作的核心主张是：**可解释路由 + 专家专业化**，而不是单纯刷 PPL。主要收益出现在代码数据集和显式因果推理任务上，自然语言 PPL 收益较弱。

---

## 1. 研究问题与贡献

### 1.1 问题

1. **MoE 路由不可解释**：Top-K/动态路由通常是黑箱，很难说明某个 token 为什么被送到某个专家，或该专家学到了什么知识。
2. **专家专业化不足 / model collapse**：少数专家接收绝大多数 token，其他专家利用不足，专家之间功能重叠；负载均衡损失有时会加剧同质化。

### 1.2 贡献

1. **Causal Routing**：用最后一层注意力的压缩块构图，CLEANN 找解释集，把关键块路由给因果专家，使路由过程有因果结构依据。
2. **Causal Expert**：专门处理强因果结构的 token 块，提升专家专业化。
3. **Causal Auxiliary Loss (CAL)**：惩罚因果专家与普通专家的参数余弦相似度，鼓励功能解耦。
4. 在 SMoE / GLaM × 原版 / Momentum / Adam × medium / large 上做了 PPL、代码数据集和显式因果推理评测。

---

## 2. 方法细节

### 2.1 总体结构

- 专家总数 16：15 个普通专家 + 1 个因果专家。
- 普通专家仍用 Top-2；**因果专家不参与 Top-K 竞争**，是额外路由。
- 因果路由只作用于**最后一个 MoE 层**。
- 对 SMoE：最后一层是 `attention + causal MoE`；对 GLaM：最后一组是 `attention + causal MoE + attention + dense FFN`，CAL 取倒数第二层（`layers[-2]`）。
- 因果专家输出与普通专家输出融合；非因果 token 只走普通专家。

### 2.2 压缩块注意力

为了避免 O(n²) 的 token 级构图，先把序列压成块：

- medium：block_size 512，`cmp_size=64`，压成 8 个块（`graph_size=8`）；
- large：block_size 1024，`cmp_size=128`，压成 8 个块。

压缩 MLP 沿序列维工作：`CompressionMLP(n=block_size, c=cmp_size, d=1)`，把 `n` 个 token 的 key/query 压成 `n//c` 个块向量。

核心代码在 `models_causal.py`：

| 位置 | 内容 |
|---|---|
| `models_causal.py:14` | `class CompressionMLP` |
| `models_causal.py:127-128` | `q_compress_mlp` / `k_compress_mlp` 实例化 |
| `models_causal.py:145-146` | `cmp_key` / `cmp_query` |
| `models_causal.py:147` | `attn_cont_cmp = torch.matmul(cmp_query, cmp_key.transpose(-1, -2))` |
| `models_causal.py:148` | 压缩注意力 softmax |
| `models_causal.py:208` | 多头平均 `att_weight.mean(1)` |
| `models_causal.py:214` | 返回 `out, att_weight` |

压缩注意力只用于路由，不参与模型主输出；模型输出仍用完整 token 注意力。当前实现中，压缩分支在 `torch.no_grad()` 的路由路径里被消费，**没有梯度路径**。如果论文坚持“可学习的压缩 MLP”，后续需要确认是否另有训练路径，或把它改成固定随机投影 / 增加训练目标。

### 2.3 因果图与解释集

1. 对压缩后的块级注意力 `A`，按 ABCD 计算：
   - `C_z = A A^T`
   - `ρ_ij = C_z(i,j) / sqrt(C_z(i,i) C_z(j,j))`
2. CLEANN 内部用 ICD + 偏相关检验学因果图，再以目标节点为根建 PDS 树，得到解释集。
3. **目标节点**：取注意力行均值最大的块（`np.argmax(adj_matrix.mean(axis=1))`），并**强制把目标块加入解释集**。这是用户确认的正式设计，不是临时实现细节。
4. **最小解释集**：用户口径下，`CLEANN(..., explanation_tester=None)` 返回的就是最小解释集（PDS 解释集），不再把它当作“full explanation set”的差异。
5. **块级路由**：解释集是块级；每个块展开成 64 个 token（large 为 128），整块 token 全部送入因果专家。

核心代码在 `custom_layers_causal.py`：

| 位置 | 内容 |
|---|---|
| `custom_layers_causal.py:24` | `split_graph_into_equal_size_subgraphs` |
| `custom_layers_causal.py:38` | `explanation_tester=None` 调用 CLEANN |
| `custom_layers_causal.py:41-43` | 目标块 = 注意力行均值最大 + 强制加入解释集 |
| `custom_layers_causal.py:177` | `self.share_expert_num = 1` |
| `custom_layers_causal.py:253` | 解释集块映射到因果专家索引 `num_expert-1` |
| `custom_layers_causal.py:308` | `share_mask` |
| `custom_layers_causal.py:360-361` | sigmoid 融合 |

### 2.4 因果专家与融合

- gate 只在前 15 个普通专家上做 Top-K（`num_expert - share_expert_num`）。
- 因果专家是第 16 个专家，索引 `num_expert-1`。
- 融合**以代码为准**：

```python
self.sigmoid_mlp = nn.Sequential(
    nn.Linear(d_model * 2, 1),
    nn.Sigmoid()
)
...
sg = self.sigmoid_mlp(torch.cat((moe_outp[share_mask], share_fwd), dim=-1))
moe_outp[share_mask] = moe_outp[share_mask] * sg + (1 - sg) * share_fwd
```

注意：论文 eq.10 写的是逐元素 `σ(y+y_causal)` 凸组合；代码是 `[普通专家输出, 因果专家输出]` 拼接后过可学习 sigmoid MLP，且只对 `share_mask` 的 token 融合。**写作/复现时以代码为准。**

### 2.5 CAL：Causal Auxiliary Loss

- 只作用最后一层。
- 对最后一层因果专家与 15 个普通专家的 `htoh4` / `h4toh` 权重分别计算余弦相似度。
- 取绝对值、对 15 个专家求均值，再对两个权重矩阵求平均：

```python
reg_loss_htoh4 = torch.abs(cos_sim_htoh4).mean()
reg_loss_h4toh = torch.abs(cos_sim_h4toh).mean()
total_reg_loss = (reg_loss_htoh4 + reg_loss_h4toh) / 2
```

- 训练时 `loss = loss + 0.01 * get_cosine_regularization(model)`。
- 代码位置：`trainer.py:12`、`trainer.py:59-60`；reasoning 路径同逻辑在 `trainer_reasoning.py:12`、`trainer_reasoning.py:75-76`。

**重要修复**：原始代码版本把 CAL 加到了 `loss_value`（日志标量），没有加到传给 `backward()` 的 `loss` 上，因此 CAL 对梯度没有影响。`first-work-clean` 已改为真正进入 backward。用户确认这是正确行为；如果对比历史 CAL 数字，需要用修正版重跑。

### 2.6 复杂度与开销

- 构图复杂度从 token 级 O(n²) 降到块级；论文给出 4090 上 512 token、单样本构图约 0.03 秒。
- 这是**单样本**数字；当前实现对每个样本、每个图块做 Python/CLEANN 循环，batch 48 的串行开销上限约 1.44 秒/512 token。论文里写“negligible”时需要补 batch 级实测或说明并行方式。

---

## 3. 模型体系（21 个配置）

### 3.1 七个模型族

| 族 | 基础架构 | causal 后缀 | 说明 |
|---|---|---|---|
| SMoE-medium | `sgsgsgsgsgsp` | `...c` | 每层 attention + MoE；16 专家、Top-2 |
| MomentumSMoE-medium | `smsmsmsmsmsk` | `...d` | MoE 输出用 momentum 状态更新 |
| AdamSMoE-medium | `sasasasasasq` | `...t` | MoE 输出用 Adam 式状态更新 |
| GLaM-medium | `sgsfsgsf...psf` | `...c` | 每组 attention + MoE + attention + dense FFN；MoE 层数约为 SMoE 一半 |
| MomentumGLaM-medium | `smsfsmsf...ksf` | `...d` | GLaM + momentum |
| AdamGLaM-medium | `sasfsasf...qsf` | `...t` | GLaM + Adam |
| MomentumSMoE-large | `smsmsmsmsmsmsmsmsmsmsmsk` | `...d` | 12 层、hidden 512、388M |

- `p/q/k`：普通 / 带计数输出的 MoE 版本；`c/d/t`：causal SMoE / causal Momentum / causal Adam。
- 每个非 large 族有 3 个配置：base、causal w/o CAL、causal + CAL；large 同样 3 个。共 7 base + 7 causal w/o CAL + 7 CAL = 21。
- CAL 由 `--causal-loss` 控制；runner 对 `cal=true` 的配置自动加。
- 详细参数在 `configs/models.json`；任务默认在 `configs/tasks.json`。

### 3.2 21 个配置清单

| id | display name | architecture | 类型 |
|---|---|---|---|
| `smoe-m` | SMoE-medium | `sgsgsgsgsgsp` | base |
| `smoe-m-causal` | CausalSMoE-medium(w/o CAL) | `sgsgsgsgsgsc` | causal w/o CAL |
| `smoe-m-causal-loss` | CausalSMoE-medium | `sgsgsgsgsgsc` | CAL |
| `smoe-mom-m` | MomentumSMoE-medium | `smsmsmsmsmsk` | base |
| `smoe-mom-m-causal` | CausalMomentumSMoE-medium(w/o CAL) | `smsmsmsmsmsd` | causal w/o CAL |
| `smoe-mom-m-causal-loss` | CausalMomentumSMoE-medium | `smsmsmsmsmsd` | CAL |
| `smoe-adam-m` | AdamSMoE-medium | `sasasasasasq` | base |
| `smoe-adam-m-causal` | CausalAdamSMoE-medium(w/o CAL) | `sasasasasast` | causal w/o CAL |
| `smoe-adam-m-causal-loss` | CausalAdamSMoE-medium | `sasasasasast` | CAL |
| `smoe-mom-l` | MomentumSMoE-large | `smsmsmsmsmsmsmsmsmsmsmsk` | base |
| `smoe-mom-l-causal` | CausalMomentumSMoE-large(w/o CAL) | `smsmsmsmsmsmsmsmsmsmsmsd` | causal w/o CAL |
| `smoe-mom-l-causal-loss` | CausalMomentumSMoE-large | `smsmsmsmsmsmsmsmsmsmsmsd` | CAL |
| `glam-m` | GLaM-medium | `sgsfsgsfsgsfsgsfsgsfspsf` | base |
| `glam-m-causal` | CausalGLaM-medium(w/o CAL) | `sgsfsgsfsgsfsgsfsgsfscsf` | causal w/o CAL |
| `glam-m-causal-loss` | CausalGLaM-medium | `sgsfsgsfsgsfsgsfsgsfscsf` | CAL |
| `glam-mom-m` | MomentumGLaM-medium | `smsfsmsfsmsfsmsfsmsfsksf` | base |
| `glam-mom-m-causal` | CausalMomentumGLaM-medium(w/o CAL) | `smsfsmsfsmsfsmsfsmsfsdsf` | causal w/o CAL |
| `glam-mom-m-causal-loss` | CausalMomentumGLaM-medium | `smsfsmsfsmsfsmsfsmsfsdsf` | CAL |
| `glam-adam-m` | AdamGLaM-medium | `sasfsasfsasfsasfsasfsqsf` | base |
| `glam-adam-m-causal` | CausalAdamGLaM-medium(w/o CAL) | `sasfsasfsasfsasfsasfstsf` | causal w/o CAL |
| `glam-adam-m-causal-loss` | CausalAdamGLaM-medium | `sasfsasfsasfsasfsasfstsf` | CAL |

---

## 4. 实验协议

### 4.1 语言模型 PPL

- 预训练：WikiText-103。
- 评测：WikiText-103、WikiText-2、PTB-Text；指标 PPL。
- 论文设置：16 专家、Top-2、medium 6 层 / hidden 352、large 12 层 / hidden 512；batch size 48、80,000 updates；每个数据集重复 10 次。
- 代码脚本常用 batch 24、batch_split 8；复现时以实际脚本/配置为准。

### 4.2 代码数据集

- HumanEval、MBPP、APPS、Codeforces；指标 PPL。
- HumanEval 采用固定 seed=42 的分层抽样：60% 训练、40% 测试。
- 代码数据天然包含函数依赖、条件分支、控制流等强因果结构，是第一份工作的主要收益来源。

### 4.3 Corr2Cause / CausalBench

- 都是**在预训练权重上继续微调**；每个域单独微调。
- Corr2Cause：形式化因果图推理（d-separation 等），Acc。
- CausalBench：Code / Math / Text 三个 scenario，multiple-choice，Acc；每个域有 cause→effect / effect→cause × with/without intervention 等视角。
- 论文/论文补充表使用 3 epochs、bs 24 / split 8、lr 0.0007（GLaM 0.00007），CAL 模型加 `--causal-loss`。
- **数据以用户最新表格为准**；旧的 `causal-bench详细分析` 汇总存在标签/epoch 对齐差异，后续若要复现，先确认每个模型对应的 checkpoint 和 epoch 选择，不要直接拿旧汇总逐格对齐。

---

## 5. 主要结果与结论

### 5.1 PPL

- 代码数据集提升显著，论文摘要给出平均 18.42%（代码）和整体 11.40%。
- 自然语言提升有限，论文正文约 2%，且个别配置变差。例如：
  - SMoE-medium：WT103 38.836 → Causal 39.436（变差）；WT2 29.868 → 29.104；PTB 62.653 → 59.504。
  - CausalMomentumSMoE-large：PTB 62.587 → 55.535。
- 因此第一份工作的“普适性”主要靠代码/强因果结构数据；毕业论文里不宜过度强调所有自然语言数据集都提升。

### 5.2 代码数据集

- Causal 模型在所有代码数据集上优于对应 base；CAL 消融有效。
- 例：CausalSMoE-medium vs SMoE-medium：HumanEval 67.11 vs 69.654，MBPP 8.544 vs 10.068，APPS 3.826 vs 4.818，Codeforces 3.081 vs 3.316（PPL，越低越好）。
- CausalMomentumSMoE-large 在 HumanEval/MBPP/APPS/Codeforces 上进一步下降。

### 5.3 显式因果推理（用户最新表，以此为准）

代表行（Corr2Cause / CausalBench Code / Math / Text，Acc，越高越好）：

| 模型 | Corr2Cause | Code | Math | Text |
|---|---|---|---|---|
| SMoE-medium | 85.018 | 48.802 | 84.343 | 71.401 |
| CausalSMoE-medium(w/o CAL) | 91.726 | 73.354 | 86.450 | 72.441 |
| CausalSMoE-medium | 92.294 | 76.269 | 87.408 | 73.295 |
| MomentumSMoE-medium | 83.729 | 66.306 | 75.592 | 70.736 |
| CausalMomentumSMoE-medium(w/o CAL) | 92.634 | 73.532 | 84.214 | 71.346 |
| CausalMomentumSMoE-medium | 93.476 | 76.735 | 85.382 | 72.870 |
| MomentumSMoE-large | 82.678 | 66.473 | 84.179 | 71.600 |
| CausalMomentumSMoE-large(w/o CAL) | 91.726 | 72.203 | 84.216 | 71.450 |
| CausalMomentumSMoE-large | 96.108 | 76.269 | 87.870 | 74.218 |

趋势：

- 因果路由本身（w/o CAL）已经大幅提升 Corr2Cause 和 CausalBench Code；CAL 通常再加一点。
- Code 提升最大，Math 次之，Text 最小；这与“代码/数学的因果结构更强”的叙事一致。
- CAL 的增益因模型而异，不是所有模型/指标都单调提升。
- 完整 21 行表见毕业论文/用户最新表；本笔记不重复旧明细的核对过程。

### 5.4 其他分析

- **解释集比例**：代码数据集上解释集 token 占比明显大于自然语言，说明路由在代码上分配了更多因果 token。
- **Top-K 敏感性**：Top-K 从 2 到 12，整体平滑；WikiText 上略降，代码任务几乎不变，支持因果专家的专业化。
- **Case study**：Codeforces 样例中，路由聚焦边界索引、关键转换等核心逻辑，忽略样板代码/异常处理。
- **CAL 消融**：去掉 CAL 后代码 PPL 变差，说明 CAL 对因果专家与普通专家的解耦有帮助（但注意历史代码版本的 CAL 未进入 backward，见 §8）。

---

## 6. 代码位置与版本映射

### 6.1 版本关系

| 版本 | 内容 | 与第一份工作的关系 |
|---|---|---|
| `main`（旧） | 早期 MomentumSMoE + 部分 causal 代码 | 没有压缩、单因果专家、CAL，不能覆盖论文 |
| `8a2f37c`（`deepseekmoe+loss`） | 压缩 + 15 普通 + 1 因果 + CAL；补齐 `p/q/k` 与 `c/d/t` 类 | **论文主方法基准版本** |
| `causal-bench-deepseekmoe+loss` | 在 `8a2f37c` 之后加 `--causal-loss` 和 Corr2Cause/CausalBench 评测 | **新表/评测代码版本** |
| `DeepseekMoe` | 删除压缩层，5%/6% token 选择，余弦损失/其他专家正则 | 第一份工作的**无压缩实验线**，不是论文主方法 |
| `first-work-clean` | 从 `main` 整理出的干净实现 + 文档 | 对外可读、可运行的版本 |

### 6.2 代码地图

| 文件 | 作用 |
|---|---|
| `models_causal.py` | CausalMoE 模型、压缩 MLP、`p/q/k` 普通层、`c/d/t` causal 层 |
| `custom_layers_causal.py` | 因果 MoE 层、CLEANN 路由、因果专家、sigmoid 融合 |
| `custom_transformer_causal.py` | causal Transformer block |
| `custom_layers.py` | FMoE 基础层、`_Expert` |
| `data.py` | LM/code 数据（txt/parquet/json）+ Corr2Cause/CausalBench 数据 |
| `train_causal.py` | 分发器：有 `train.json`/`valid.json` 走 reasoning，否则走 LM |
| `train_lm.py` | PPL 预训练 / 代码微调入口 |
| `train_reasoning.py` | Corr2Cause / CausalBench 入口 |
| `trainer.py` | LM/code trainer |
| `trainer_reasoning.py` | reasoning trainer |
| `configs/models.json` | 21 个模型配置 |
| `configs/tasks.json` | 四条实验线的任务默认值 |
| `scripts/run.py` | 参数化 runner |
| `causalitylab/` | CLEANN/ICD 等因果发现与解释代码 |

### 6.3 关键代码行

- 压缩注意力：`models_causal.py:147`
- 多头平均：`models_causal.py:208`
- 目标节点/解释集：`custom_layers_causal.py:41-43`
- 单因果专家：`custom_layers_causal.py:177`
- 因果专家路由：`custom_layers_causal.py:253`
- sigmoid 融合：`custom_layers_causal.py:360-361`
- CAL：`trainer.py:12`、`trainer.py:59-60`；`trainer_reasoning.py:12`、`trainer_reasoning.py:75-76`
- 数据加载修复：`data.py:135`（列选择）、`data.py:145`（txt/parquet/json）、`data.py:173`（valid/validation）、`data.py:186`（两遍建词表）、`data.py:208`（Corpus）、`data.py:219`（`self._dictionary`）
- 分发器：`train_causal.py:15`

---

## 7. 已确认的设计口径（不要重复争论）

1. 目标节点取注意力行均值最大的块，并强制把目标块加入解释集：**正式设计**。
2. `CLEANN(..., explanation_tester=None)` 返回的就是**最小解释集**（用户口径）。
3. 解释集是块级，**整块 64 token（large 128）全部送入因果专家**。
4. 融合公式**以代码为准**：concat + 可学习 sigmoid MLP，只对因果路由 token 融合。
5. CAL **只作用最后一层**。
6. 0.03s/512 token 是**单样本**构图时间，不是整批。
7. 第二份工作**不是**把 CausalMoE 迁到 DeepSeekMoE；希望是“方法递进”。
8. CausalBench/Corr2Cause 数据以最新表格为准，不再按旧汇总纠正。
9. 第一份工作更强调“可解释路由 + 专家专业化”，不是单纯追求 NL PPL。

---

## 8. 实现层已知问题与修复

### 8.1 CAL 未进入 backward（已修）

原始 `trainer.py` / `trainer_reasoning.py`：

```python
loss_value = loss.item() / loss_div
loss_value = loss_value + 0.01 * get_cosine_regularization(model)
...
(loss / loss_div).backward()
```

CAL 只加到日志标量 `loss_value`，没有加到 `loss`，对梯度无影响。`first-work-clean` 已改为：

```python
loss = nll_loss(...)
if causal_loss:
    loss = loss + 0.01 * get_cosine_regularization(model)
loss_value = loss.item() / loss_div
```

### 8.2 `Corpus` 词表问题（已修）

原始 `data.py`（`8a2f37c`）有四个问题：

1. `_tokenize` 最后用了未定义的 `dictionary.get(...)`，第一个 split 就 `NameError`；应为 `dictionary_to_update.get(...)`。
2. `Corpus.__init__` 从不给 `self._dictionary` 赋值，`vocab_size` 会 `AttributeError`；修复后 `self._dictionary = tokenizer.vocab`。
3. `Corpus` 只读 `train.txt/valid.txt/test.txt`，parquet 分支没被调用；`_tokenize_parquet` 还硬编码 `solution` 列，并且只编码、不建词表，导致所有 token 变 `<unk>`。
4. 原始 txt 编码 `dictionary_to_update[token]` 对未登录词会 `KeyError`；修复后 `.get(token, unk_id)`。

现在的 `data.py`：

- 支持 `.txt` / `.parquet` / `.json`；
- 自动选择 `text` / `solution` / `code` / `content` / `question` / `answer` 列；
- list 列用空格拼接；
- 支持 `valid.*` 和 `validation.*`；
- 两遍扫描：先建词表，再编码；
- 特殊 token `<pad>` / `<unk>` / `<eos>` 预先加入；
- 保存 `my_custom_tokenizer`，供 reasoning 路径加载。

### 8.3 其他清理

- `custom_layers_causal.py` 不再 `sys.path.append("/hy-tmp/causalitylab")`，改用仓库内 `causalitylab`。
- `g is "x"` / `s is "s"` / `f is "f"` 改成 `==`。
- `torch.tensor(已存在tensor)` 改成 `.to(device)`。
- `utils.py` 缺逗号的语法错误已修。
- 删除 IDE、缓存、build、egg-info、旧版 `custom_layers_causa_old.py`。

### 8.4 入口拆分

- `train_causal.py` 是分发器；旧的 `python ... train_causal.py $args` 仍可用。
- `train_lm.py` 和 `train_reasoning.py` 分开，因为 LM/code 与 reasoning 的 data/trainer API 不同，硬合并容易破坏原有行为。
- `trainer.py` 与 `trainer_reasoning.py` 分开，原因同上。

### 8.5 尚未验证

- 本机没有 `torch` / `fmoe_cuda`，只做了 `py_compile`、manifest/arch 静态检查、runner dry-run。
- 4090 上需要先跑 smoke test，确认 `fmoe_cuda`、数据加载、前向/反向、tokenizer 都正常。

---

## 9. 局限与可补的实验

1. **自然语言 PPL 收益弱**，部分配置变差；主张应聚焦代码/强因果结构数据和显式因果推理。
2. **单层、块级、注意力构图**：因果图来自注意力相关 + 偏相关检验，不是干预数据；论文中应明确假设与边界。
3. **压缩 MLP 是否可学习**：当前实现只用于 `no_grad` 路由；若论文称 learnable，需要补训练路径或改成固定投影。
4. **随机路由对照**：当前论文没有看到“随机选第 16 个专家”的对照；`w/o CAL` 不能完全排除“多一个专用专家/额外容量”的收益。第二份工作可以补这个 ablation。
5. **CausalBench 协议细节**：每域单独微调、epoch/checkpoint 选择、简单平均 vs 题目数加权，需要在论文/补充材料里写清楚；复现时以最新表格为准。
6. **开销**：0.03s 是单样本；batch 级构图开销和并行方式需要补测。
7. **规模**：最大 388M、约 103M tokens 训练语料，小于当前大规模 MoE；需要在更大模型上验证。

---

## 10. 对第二份工作的衔接点（暂定，未最终决定）

用户明确：第二份工作不是迁到 DeepSeekMoE 的细粒度/共享专家，而是**方法递进**。从第一份工作的限制出发，可能的递进方向：

1. **单层 → 跨层**：把因果路由从最后一层扩展到多层，研究跨层因果一致性。
2. **块级 → 更细粒度/可学习**：让压缩/路由端到端可学习，或做 token 级与块级的混合路由。
3. **启发式解释集 → 更可验证的因果解释**：引入干预/反事实或更强的因果识别约束。
4. **专家分工机制**：从“一个额外因果专家”推进到“因果专家与普通专家的动态分工/共享专家”。
5. **评测**：补随机路由对照、分视角 CausalBench、OOD/反事实任务。
6. **效率**：把 per-sample CLEANN 循环做成批量/GPU 友好实现。

这些只是候选，等第一份工作的版本和数据口径固定后再和用户确认主线。

---

## 11. 快速上手

### 11.1 环境

```bash
pip install -r requirements.txt
pip install -e .          # 构建 fmoe_cuda，需要 CUDA + ninja
```

### 11.2 数据

默认 `--data-root data_directory`。LM/code 支持 `.txt` / `.parquet` / `.json`；reasoning 需要：

```text
data_directory/
  corr2cause/{train.json, valid.json, test/*.json}
  causal-bench/{train.json, valid.json, test/*.json}
```

JSON 格式：`[{"text": ..., "label": ...}, ...]`。推理数据必须能生成/复用 `my_custom_tokenizer`；先跑一次 LM/code 预训练，或手动放好 tokenizer。

### 11.3 命令

```bash
python scripts/run.py --list
python scripts/run.py --model smoe-m --task pretrain --dry-run
python scripts/run.py --model smoe-m --task pretrain
python scripts/run.py --model smoe-m --task finetune_code --dataset MBPP
python scripts/run.py --model smoe-m-causal-loss --task corr2cause
python scripts/run.py --model smoe-m-causal-loss --task causalbench --eval
```

### 11.4 最小 smoke test

```bash
python scripts/run.py --model smoe-m --task pretrain --extra "--niter 1 --nbatches 2"
```

---

## 12. 快速结论（给未来接手的人）

- 第一份工作 = 压缩块注意力 + CLEANN 解释集 + 额外因果专家 + sigmoid 融合 + 最后一层 CAL。
- 论文主方法代码 = `8a2f37c`；干净整理版 = `first-work-clean`。
- 代码/显式因果推理收益明显；自然语言 PPL 收益弱。
- CAL 必须真正进入 backward；原始代码版本有 bug。
- 数据加载的 `Corpus` 原始版本有四个 bug；clean 版已修。
- 设计口径以用户确认的 §7 为准；不要再就“目标块启发式 / 最小解释集 / 块级路由 / 融合公式 / CAL 层数”反复争论。
- 第二份工作做“方法递进”，不是 DeepSeekMoE 迁移；从 §9/§10 的限制里选主线。
