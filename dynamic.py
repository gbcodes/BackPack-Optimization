# Импорт либ
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# список файлов
tests = ["5", "8", "10", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"]
files = ['tests\\test_' + x + '.csv' for x in tests]
results = ['tests\\bpresult_' + x + '.csv' for x in tests]


def knapSack(items):
    """
    :param items: [capacity, weights, values, N]
    :return: best_value_cost, fractions
    """
    W = items[0]
    wt = items[1]
    val = items[2]
    n = items[3]

    K = [[0 for x in range(int(W) + 1)] for x in range(n + 1)]

    # Создаем таблицу кэша K[][] bottom up и далее динамическое программирование
    for i in range(n + 1):
        for w in range(int(W) + 1):

            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][int(w - wt[i - 1])],
                              K[i - 1][w])  # если вес предмета меньше веса столбца,
                # максимизируем значение суммарной ценности
            else:
                K[i][w] = K[i - 1][w]  # если площадь предмета больше площади столбца,
                # забираем значение ячейки из предыдущей строки

    best_combination = [0] * n
    result = K[n][int(W)]
    cap_copy = W
    for i in range(n, 0, -1):
        if round(result) != round(K[i - 1][int(cap_copy)]):
            best_combination[i - 1] = 1
            result -= val[i - 1]
            cap_copy -= wt[i - 1]
    return K[n][int(W)], best_combination


# читаем данные
data = []
for file in files:
    test = pd.read_csv(file, names=['weights', 'values'], delimiter=";")
    N = test.shape[0]
    weights = test['weights'].tolist()
    values = test['values'].tolist()
    capacity = test['weights'].sum() / 2
    data.append([capacity, weights, values, N])

from time import time

deltas = []
for job in range(1, 9):
    start = time()
    res = Parallel(n_jobs=job)(delayed(knapSack)(d) for d in data)  # параллелизация
    delta = time() - start
    deltas.append(delta)


fractions = [x[1] for x in res]
for r in range(len(results)):
    resu = pd.read_csv(results[r], sep=';', header=None)
    frac = resu.loc[0].values.tolist()
    try:
        np.testing.assert_allclose(frac, fractions[r])
    except AssertionError:
        print(results[r])
        print('result: ', frac)
        print('test: ', fractions[r], '\n')

differ = pd.DataFrame()
differ['time'] = deltas
differ['diff'] = differ['time'][0] / differ['time']

differ['time'].plot(title='Зависимость времени от числа потоков')
plt.xlabel('Поток')
plt.ylabel('Время')
plt.show()

differ['diff'][1:].plot(title='Уменьшение времени от числа потоков')
plt.xlabel('Поток')
plt.ylabel('Время')
plt.show()

data = []
for file in files:
    test = pd.read_csv(file, names=['weights', 'values'], delimiter=";")
    N = test.shape[0]
    weights = test['weights'].tolist()
    values = test['values'].tolist()
    capacity = test['weights'].sum() / 2
    data.append([capacity, weights, values, N])

# генерируем новые тесты
nums = 50
gen_w = []
gen_v = []
for i in range(27, nums):
    gen_w.append([np.random.uniform(10, 1000) for x in range(i)])
    gen_v.append([np.random.uniform(10, 1000) for x in range(i)])

data_gen = [[np.sum(gen_w[i]) / 2, gen_w[i], gen_v[i], len(gen_v[i])] for i in range(len(gen_v))]
for i in data_gen:
    data.append(i)

from time import time

deltas = []
for job in range(1, 9):
    start = time()
    res = Parallel(n_jobs=job)(delayed(knapSack)(d) for d in data)
    delta = time() - start
    deltas.append(delta)

differ1 = pd.DataFrame()
differ1['time'] = deltas
differ1['diff'] = differ1['time'][0] / differ1['time']

differ1['time'].plot(title='Зависимость времени от числа потоков')
plt.xlabel('Поток')
plt.ylabel('Время')
plt.show()

differ1['diff'][1:].plot(title='Уменьшение времени от числа потоков')
plt.xlabel('Поток')
plt.ylabel('Время')
plt.show()