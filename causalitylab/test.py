from causal_reasoning import CLEANN
from plot_utils import draw_graph
from experiment_utils.explanation import exhaustive_search_explanation

import torch
from transformers import BertForMaskedLM, BertConfig, BertTokenizer, logging

from itertools import combinations
logging.set_verbosity_error()

def explanation_definer(token_predictor, tokens_list, explanation_mask, prediction_mask):
    """
    A test if the given subset of the tokens can serve as an 'explanation' for the masked-token prediction.
    :param token_predictor: a function that predicts a masked token.
    :param tokens_list: a full list of tokens
    :param explanation_mask: a token that will replace the explanation tokens when testing the explanation
    :param prediction_mask: a masking token that will replace the target token when testing the explanation
    :return: a testing function with input parameters:
        explanation_token_pos: indexes of those tokens to be considered as explanation
        target_pos: index of the target node to be explained
    """
    def examine_explanation(explanation_token_pos: list, target_pos: int) -> bool:
        """
        A function for testing if a subset of tokens can serve as an explanation for the target token.
        :param explanation_token_pos: indexes of those tokens to be considered as explanation
        :param target_pos: index of the target node to be explained
        :return:
        """
        masked_tokens = tokens_list.copy()
        target_token = tokens_list[target_pos]
        masked_tokens[target_pos] = prediction_mask  # e.g., '[MASK]'
        for token_pos in explanation_token_pos:
            masked_tokens[token_pos] = explanation_mask  # e.g., '[PAD]'

        predicted_token = token_predictor(masked_tokens, target_pos)
        return predicted_token != target_token

    return examine_explanation

model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained("/hy-tmp/causality-lab/")
config = BertConfig.from_pretrained("/hy-tmp/causality-lab/", output_attentions=True)
bert_model = BertForMaskedLM.from_pretrained("/hy-tmp/causality-lab/", config=config)

def masked_token_predictor(model, tokenizer):
    """
    Creates a function that takes a list of tokens as input and predicts the masked token using the BERT model.
    :param model: a BERT model
    :param tokenizer: a corresponding tokenizer
    :return: a prediction function
    """
    def predictor(masked_tokens, target_pos):
        """
        A function that takes a list of tokens as input and predicts the masked token using the BERT model
        :param masked_tokens: list of tokens
        :param target_pos: index of the target node to be explained
        :return: the predicted token
        """
        in_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
        with torch.no_grad():
            logits = model(torch.tensor([in_ids])).logits
        predicted_token_id = logits[0, target_pos].argmax(axis=-1)
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
        return predicted_token
    return predictor


import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities
def split_graph_from_adjacency_matrix(adj_matrix, num_subgraphs):
    # 创建一个图
    G = nx.from_numpy_array(adj_matrix)

    # 使用 Louvain 社区检测算法进行划分
    communities = list(louvain_communities(G))
    if len(communities) > num_subgraphs:
        combined_communities = [sum(communities[i:i+num_subgraphs], []) for i in range(0, len(communities), num_subgraphs)]
    else:
        combined_communities = communities
    subgraphs = [G.subgraph(subgraph_nodes) for subgraph_nodes in combined_communities]

    return subgraphs


token_unmasker = masked_token_predictor(bert_model, bert_tokenizer)  # create a function that predicts a masked token given a list of tokens

txt = 'The blue whale is the largest animal on the planet.'

target_id = 6  # the token 'largest' will be masked
nodes_of_interest = {2, 3, 6, 7, 10}  # indexes of tokens for which a graph will be learned
encoded_input = bert_tokenizer(txt, return_tensors='pt')
tokens = bert_tokenizer.convert_ids_to_tokens(encoded_input.input_ids[0])

bert_explanation_tester = explanation_definer(token_predictor=token_unmasker, tokens_list=tokens,
                                              explanation_mask='[PAD]',
                                              prediction_mask='[MASK]')  # explanation tester

true_explanations_list = exhaustive_search_explanation(nodes_of_interest, target_id,
                                                          bert_explanation_tester, search_minimal=False)
for true_explanation in true_explanations_list:
    explanation_tokens = [tokens[pos] for pos in true_explanation]
    print('Exhaustive search result (ground truth):', explanation_tokens)

with torch.no_grad():
    out = bert_model(**encoded_input)
last_mh_attention = out.attentions[-1]

found_explanations = dict()
for head_id in range(last_mh_attention.shape[1]):
    attention_head = last_mh_attention[0, head_id, :, :]
    att = attention_head.numpy()  # attention matrix of current head

    # Extract explanations
    print("attention_matrix.shape:",att.shape)
    print("num_samples:",config.hidden_size)
    print("bert_explanation_tester:",bert_explanation_tester)
    explainer = CLEANN(attention_matrix=att, num_samples=config.hidden_size, p_val_th=1e-2,
                       explanation_tester=bert_explanation_tester,
                       nodes_set=nodes_of_interest, search_minimal=False)

    head_explanations_list = explainer.explain(target_node_idx=target_id)

    found_explanations[head_id] = {'explanations': head_explanations_list,
                                   'graph': explainer.graph}
    num_subgraphs=3
    subgraphs = split_graph_from_adjacency_matrix(explainer.graph.get_skeleton_mat(), num_subgraphs)

    for i, subgraph in enumerate(subgraphs):
        print(f"Subgraph {i + 1}:")
        print(subgraph.nodes())


    if len(head_explanations_list) > 0:
        print('Head', head_id)
        for head_explanation in head_explanations_list:
            explanation_tokens = [tokens[pos] for pos in sorted(head_explanation[0])]
            print('>>> Found explanation:', explanation_tokens)

node_labels = {node: tokens[node] for node in range(len(encoded_input.input_ids[0]))}
fig = draw_graph(found_explanations[8]['graph'], node_labels=node_labels, node_size_factor=2.0)

