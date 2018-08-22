import pylab
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)

line, = ax.plot(fppi,miss_rate, color='blue', lw=2)

plt.semilogy(fppi, miss_rate)
plt.title('Average FPPI vs Log(Miss rate) on hand-labelled test set')
plt.grid(True)

plt.xlabel('FPPI')
plt.ylabel('Log(Miss Rate)')
plt.ylim([0.0, 1.00])

plt.legend()
plt.show()