import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

N = np.array([1e7, 3e7, 1e8, 3e8, 1e9])
loss = np.array([3.2, 2.9, 2.6, 2.45, 2.35])

logN = np.log(N).reshape(-1, 1)
logL = np.log(loss)

reg = LinearRegression().fit(logN, logL)
alpha = -reg.coef_[0]

pred = np.exp(reg.predict(logN))

plt.scatter(N, loss)
plt.plot(N, pred)
plt.xscale("log")
plt.yscale("log")
plt.show()

print("estimated scaling exponent:", alpha)
