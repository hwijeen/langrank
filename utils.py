import numpy as np

def ndcg(y_score, y_true, k):
    gain = _ndcg_sample_scores(y_true, y_score, k=k, ignore_ties=False)
    return np.average(gain, weights=None)

def _ndcg_sample_scores(y_true, y_score, k, ignore_ties):
    gain = _dcg_sample_scores(y_true, y_score, k, ignore_ties=ignore_ties)
    # Here we use the order induced by y_true so we can ignore ties since
    # the gain associated to tied indices is the same (permuting ties doesn't
    # change the value of the re-ordered y_true)
    normalizing_gain = _dcg_sample_scores(y_true, y_true, k, ignore_ties=True)
    all_irrelevant = normalizing_gain == 0
    gain[all_irrelevant] = 0
    gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]
    return gain

def _dcg_sample_scores(y_true, y_score, k=None, ignore_ties=False):
    log_base = 2
    discount = 1 / (np.log(np.arange(y_true.shape[1]) + 2) / np.log(log_base))
    if k is not None:
        discount[k:] = 0
    if ignore_ties:
        ranking = np.argsort(y_score)[:, ::-1]
        ranked = y_true[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
        cumulative_gains = discount.dot(ranked.T)
    else:
        discount_cumsum = np.cumsum(discount)
        cumulative_gains = [_tie_averaged_dcg(y_t, y_s, discount_cumsum)
                            for y_t, y_s in zip(y_true, y_score)]
        cumulative_gains = np.asarray(cumulative_gains)
    return cumulative_gains


def _tie_averaged_dcg(y_true, y_score, discount_cumsum):
    _, inv, counts = np.unique(
        - y_score, return_inverse=True, return_counts=True)
    ranked = np.zeros(len(counts))
    np.add.at(ranked, inv, y_true)
    ranked /= counts
    groups = np.cumsum(counts) - 1
    discount_sums = np.empty(len(counts))
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = np.diff(discount_cumsum[groups])
    return (ranked * discount_sums).sum()