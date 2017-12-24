import numpy as np
import matplotlib.pyplot as plt
from  math import  log
from  sklearn.metrics import  log_loss
#https://www.grund-wissen.de/informatik/python/scipy/matplotlib.html

def logloss(true_label, predicted):
  eps = 1e-15
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -log(p)
  else:
    return -log(1 - p)

def r(actual, predicted, eps = 1e-15):
    predicted = min(max(predicted, eps), 1-eps)
    return - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

#t1 = np.arange(0.6, 1, 0.0001) # detaillierte Ansicht
t1 = np.arange(0.01, 1, 0.0001)

print(t1)

y = [1]*len(t1)

res = list(map(logloss, y, t1))
avg = sum(res)/len(res)
print(avg)
# zum herausfinden was p ist
for i in range(len(t1)):
    if round(res[i], 3)== 0.553:
        print(t1[i], res[i])

plt.figure(1, figsize=(7,1))
plt.plot(t1,list(map(logloss, y, t1)), color="black")
#plt.title("Log-Loss Funktion für ein Fragen-Duplikat")
plt.xlabel("vorhergesagte Wahrscheinlichkeit p")
plt.ylabel("log-loss")
mean = 0.711 #0.34108284917889703
best = 0.889 #0.11765804346823845
#personal = 0.8519 #0.16028612993334826
personal = 0.5752 #0.553037472575306

plt.scatter([mean], logloss(1, mean), s=50, marker='o', color='black')
plt.annotate(r'Durchschnitt',
              xy=(mean, logloss(1, mean)), xycoords='data',
              xytext=(+0, +20), textcoords='offset points', fontsize=10,
              arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))

plt.scatter([personal], logloss(1, personal), s=50, marker='o', color='black')
plt.annotate(r'Final',
             xy=(personal, logloss(1, personal)), xycoords='data',
             xytext=(+10, +20), textcoords='offset points', fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))

plt.scatter([best], logloss(1, best), s=50, marker='o', color='black')
plt.annotate(r'Gewinner',
             xy=(best,  logloss(1, best)), xycoords='data',
             xytext=(+5, +20), textcoords='offset points', fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))

plt.show()

