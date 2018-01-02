import pandas as pd
from sklearn.model_selection import train_test_split
import data_paths
from log_loss import logloss_array

# y = [1,1,1,1,1,1,0,0,0,0]
# p = [1,1,1,1,1,0,0,0,1,1]
# print(logloss_array(y,p))

x_train = pd.read_csv(data_paths.train, encoding="ISO-8859-1")

# Ausgabedaten erstellen.
y_train = x_train.is_duplicate

# Trainings-Set und Validierungs-Set erstellen. Das Validierungs-Set enthält 10% aller Trainings-Daten.
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=2930)

# Logloss für Validierungsset berechnen wenn alles 0 oder 1 ist.
#p = len(y_valid.index) * [1]
p = len(y_valid.index) * [0]

print(logloss_array(y_valid, p))