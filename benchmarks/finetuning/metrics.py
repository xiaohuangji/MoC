"""Original GLUE dev metrics; ties in Spearman receive average ranks."""
import numpy as np


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1)/2 + 1
        start = end
    return ranks


def stsb(gold, predicted):
    a, b = np.asarray(gold, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    assert a.ndim == b.ndim == 1 and a.shape == b.shape and len(a) >= 2
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.ptp(a) > 0 and np.ptp(b) > 0, 'Correlation undefined for constant values'
    pearson = float(np.corrcoef(a, b)[0, 1])
    spearman = float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])
    assert np.isfinite([pearson, spearman]).all()
    return dict(pearson=pearson, spearmanr=spearman, combined=(pearson+spearman)/2)


def f1(gold, predicted):
    assert len(gold) == len(predicted) and len(gold) > 0
    assert set(gold) <= {0, 1} and set(predicted) <= {0, 1}
    tp = sum(g == p == 1 for g, p in zip(gold, predicted))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, predicted))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, predicted))
    accuracy = sum(g == p for g, p in zip(gold, predicted))/len(gold)
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    return dict(accuracy=accuracy, f1=f1, combined=(accuracy+f1)/2)


def cola(gold, predicted):
    assert len(gold) == len(predicted) and len(gold) > 0
    assert set(gold) <= {0, 1} and set(predicted) <= {0, 1}
    tp = sum(g == p == 1 for g, p in zip(gold, predicted))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, predicted))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, predicted))
    accuracy = sum(g == p for g, p in zip(gold, predicted))/len(gold)
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    tn = sum(g == p == 0 for g, p in zip(gold, predicted))
    denominator = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** .5
    mcc = (tp*tn-fp*fn)/denominator if denominator else 0.0
    return dict(matthews_correlation=mcc)


def mnli(gold, predicted, matched_count=9815):
    assert len(gold) == len(predicted) and 0 < matched_count < len(gold)
    assert set(gold) <= {0, 1, 2} and set(predicted) <= {0, 1, 2}
    matched = sum(g == p for g, p in zip(gold[:matched_count], predicted[:matched_count]))/matched_count
    mismatched = sum(g == p for g, p in zip(gold[matched_count:], predicted[matched_count:]))/(len(gold)-matched_count)
    return dict(matched_accuracy=matched, mismatched_accuracy=mismatched, combined=(matched+mismatched)/2)


def accuracy(gold, predicted):
    assert len(gold) == len(predicted) and len(gold) > 0
    assert set(gold) <= {0, 1} and set(predicted) <= {0, 1}
    accuracy = sum(g == p for g, p in zip(gold, predicted))/len(gold)
    return dict(accuracy=accuracy)


SCORERS = {"stsb": stsb, "mnli": mnli, "mrpc": f1, "qqp": f1, "cola": cola,
           "sst2": accuracy, "rte": accuracy, "qnli": accuracy}
