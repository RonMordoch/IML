import numpy as np
from matplotlib import pyplot as plt

data = np.random.binomial(1, 0.25, (100000, 1000))
epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
tosses = np.arange(1, 1001)


def plot_means():
    for i in range(5):
        plt.plot(tosses, np.cumsum(data[i]) / tosses)
    plt.xlabel("Number of coins tosses")
    plt.ylabel("Mean value")
    plt.show()


def plot_variances():
    for eps in epsilon:
        plt.plot(tosses, np.minimum(1, 1 / (4 * tosses * (eps ** 2))), 'r',
                 label='Chebyshev Bound')
        plt.plot(tosses, np.minimum(1, 2 * np.exp(-2 * tosses * (eps ** 2))),
                 'b', label='Hoeffding Bound')

        plt.plot(tosses,
                 np.sum(abs((np.cumsum(data, axis=1) / tosses) - 0.25) >= eps,
                        axis=0) / 100000, 'g', label='Percentage')

        plt.xlabel("Number of coins tosses")
        plt.ylabel("Probability")
        plt.title(r"$\epsilon$  = " + str(eps))
        plt.legend()
        plt.show()


plot_means()
plot_variances()
