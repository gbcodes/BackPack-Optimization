import pandas as pd
from glob import glob

def knapsack(V, W, C):
    return knapsack_aux(V, W, len(V)-1, C)

def knapsack_aux(V, W, i, aW):
    """

    :param V:
    :param W:
    :param i:
    :param aW:
    :return:
    """
    if i == -1 or aW == 0:
        return 0
    elif W[i] > aW:

        a= knapsack_aux(V, W, i-1, aW)
    else:
        a= max(knapsack_aux(V, W, i-1, aW),
                   V[i] + knapsack_aux(V, W, i-1, aW-W[i]))


files = glob('*.csv')
for file in files:
    test = pd.read_csv(file, names=['weights', 'values'], delimiter=";")
    N = test.shape[0]
    weights = test['weights'].tolist()
    values = test['values'].tolist()
    capacity = test['weights'].sum() / 2

    print(file+ ' ' + str(knapsack(values, weights, capacity)))