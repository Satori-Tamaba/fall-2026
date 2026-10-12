import numpy as np

from dataset import *
from kernels import GaussianKernel
from metric_classifier import ParzenClassifier, PrototypeSelection
from utils import loo
import matplotlib.pyplot as plt
from sklearn.neighbors import  KNeighborsClassifier
if __name__ == "__main__":
    k_n = 30
    data = load_data("data/apple_quality.csv")
    data = prepare_data(data, drop_cols= DROP_COLS, target= TARGET)
    x, y =  data.drop(columns= TARGET).to_numpy(dtype=float) ,data[TARGET].to_numpy(dtype=float).ravel()
    rng = np.random.default_rng(43)
    idx = rng.permutation(len(x))
    split = int(0.8 * len(x))
    tr, te = idx[:split], idx[split:]

    X_tr, X_te = x[tr], x[te]
    y_tr, y_te = y[tr], y[te]
""""
    k_best, _,  emp_risk = loo(x,y, k_n=k_n)
    print(f"Лучший k {k_best}")
    plt.plot(np.arange(1,k_n,1),emp_risk)
    plt.title("Зависимость LOO от числа соседей")
    plt.xlabel("Число соседей")
    plt.ylabel("Частота ошибок")
    plt.show()
    ref_knn = KNeighborsClassifier(n_neighbors = k_best, weights='distance')
    ref_knn.fit(X_tr, y_tr)
    ref_pred = ref_knn.predict(X_te)
    my_clf = ParzenClassifier()
    my_clf.fit(X_tr, y_tr)
    ker = GaussianKernel()
    pred = my_clf.predict(X_te, ker, k=11)
    print(f"{np.sum(ref_pred != y_te)/ len(y_te)}")
    print(f"{np.sum(pred != y_te)/ len(y_te)}")
"""

ps = PrototypeSelection(k=7)
ps.fit(X_tr[:100], y_tr[:100])
alive = ps.select()
print("осталось эталонов:", alive.sum())


