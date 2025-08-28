import math
import numpy as np
from scipy.spatial.distance import jensenshannon

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def recall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    rec = hits / (1.0 * len(ground_list))
    return rec


def ndcg(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg


def IDCG(n):
    idcg=0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg


def js_topk(topk_items, sens, test_u2i, n_users, n_items, topk):
    # Build full recommendation matrices
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    correct_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    test_topk_items = topk_items.tolist()

    for uid in range(n_users):
        rank_topk_items[uid, test_topk_items[uid][:topk]] = 1
        correct_topk_items[uid, test_u2i[uid]] = 1

    # Keep only correctly recommended items
    correct_topk_items &= rank_topk_items

    # Aggregate distributions by sensitive attribute
    rank_dis_list = []
    correct_dis_list = []
    unique_sens = np.unique(sens)
    for s in unique_sens:
        index = (sens == s)
        rank_dis_list.append(np.sum(rank_topk_items[index], axis=0))
        correct_dis_list.append(np.sum(correct_topk_items[index], axis=0))

    # Compute the overall distribution
    rank_dis_avg = np.sum(rank_topk_items, axis=0)
    correct_dis_avg = np.sum(correct_topk_items, axis=0)

    # If sensitive attribute has only two groups
    if len(unique_sens) <= 2:
        index1 = (sens == unique_sens[1])
        index2 = ~index1

        js_dp = jensenshannon(
            np.sum(rank_topk_items[index1], axis=0),
            np.sum(rank_topk_items[index2], axis=0)
        )
        js_eo = jensenshannon(
            np.sum(correct_topk_items[index1], axis=0),
            np.sum(correct_topk_items[index2], axis=0)
        )
        return js_dp, js_eo

    # For multiple groups, compute JS distance for each group against the overall distribution
    js_dp_list = [jensenshannon(r, rank_dis_avg) for r in rank_dis_list]
    js_eo_list = [jensenshannon(r, correct_dis_avg) for r in correct_dis_list]

    js_dp = np.mean(js_dp_list)
    js_eo = np.mean(js_eo_list)

    return js_dp, js_eo