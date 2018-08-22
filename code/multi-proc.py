from multiprocessing import Pool
import time
import numpy as np

def f(x):
    return x*x

# def fl():
# 	for i in range(1,50):
# 		return f(i)
a = np.arange(1,50000000)
start = time.time()
if __name__ == '__main__':
    p = Pool(5)
    o = p.map(f,a)
        # print(o)
end = time.time()
print(end-start)

# if __name__ == '__main__':
# 	start = time.time()
#     with Pool(3) as p:
#         print(p.map(f, [1, 2, 3]))
#     end = time.time()
#     print(end-start)
