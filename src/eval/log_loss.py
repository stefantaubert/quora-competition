import numpy as np
from math import log

#https://www.grund-wissen.de/informatik/python/scipy/matplotlib.html
def logloss(true_label, predicted):
    eps = 1e-10
    max = 1.0 - eps

    p = np.clip(predicted, eps, max)

    if true_label == 1:
        return -log(np.clip(p, eps, max))
    else:
        return -log(np.clip(1-p, eps, max))

def logloss_array(true_label, predicted):
    res = list(map(logloss, true_label, predicted))
    ret = sum(res) / len(res)
    return ret