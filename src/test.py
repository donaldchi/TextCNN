# -*- coding: utf-8 -*-
from time import time

from joblib import Parallel, delayed


def process(n):
    return sum([i*n for i in range(100000)])

start = time()

# 繰り返し計算 (並列化)
r = Parallel(n_jobs=-1)( [delayed(process)(i) for i in range(10000)] )
print(sum(r))

print('{}秒かかりました'.format(time() - start))
