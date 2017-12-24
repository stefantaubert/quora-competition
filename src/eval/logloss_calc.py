import numpy as np
from math import log 
import pandas as pd
from sklearn.model_selection import train_test_split

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


y = [1,1,1,1,1,1,0,0,0,0]
p = [1,1,1,1,1,0,0,0,1,1]
print(logloss_array(y,p))

x_train = pd.read_csv("D:/dev/Python/quora-bachelor-thesis/data/train.csv", encoding="ISO-8859-1")

# Ausgabedaten erstellen.
y_train = x_train.is_duplicate

# Trainings-Set und Validierungs-Set erstellen. Das Validierungs-Set enthält 10% aller Trainings-Daten.
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=2930)

p = len(y_valid.index) * [1]

print(logloss_array(y_valid,p))