import time
import numpy as np 
def f(x):
    return x*x

a = np.arange(1,50000000)
start = time.time()
for i in a:
	o = f(i)
	# print(o)
end = time.time()
print(end-start)