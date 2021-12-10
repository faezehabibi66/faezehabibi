import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

LEARNING_RATE = 0.00000001


def lr(x, y):
    dm, db = 6, 5
    m, b = 1, 0
    error = float("INF")
    z = []

    while (dm > 0.2 or dm < -0.03) and (db > 0.002 or db < -0.2):
        dm = np.sum(np.sinh(m * x + b - y)*x/np.cosh(m * x + b - y))
        db = np.sum(np.sinh(m * x + b - y)/np.cosh(m * x + b - y))
        error = sum(np.log(np.cosh(m * x + b - y)))

        m -= LEARNING_RATE * dm
        b -= LEARNING_RATE * db

    for u in range(n_x):
        z.append(m * x[u] + b)

    xtest = 21
    ht = m * xtest + b
    print(ht, error, m, b)
    plt.scatter(x, y)
    plt.scatter(x, z)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(r"C:\\Users\\Faeze\\Documents\\ML\1\\test.csv")
    x = df.x.array
    x_1 = 5 * x
    x_2 = x*x
    n_x = x.size
    y = df.y.array

    lr(x, y)
    lr(x_1, y)
    lr(x_2, y)