from scipy.io import loadmat
import matplotlib.pyplot as plt

LM = loadmat(r"data/LM.mat")
alpha = LM['alpha_res'].squeeze()
iter = LM['iter_res'].squeeze()
plt.plot(iter[1:], alpha[1:])
plt.show()
