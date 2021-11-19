from scipy.io import loadmat
import matplotlib.pyplot as plt

LM = loadmat(r"data/LM.mat")
alpha = LM['alpha_res'].squeeze()
iter = LM['iter_res'].squeeze()

plt.xlabel("alpha", fontsize=12)
plt.ylabel("iter", fontsize=12)
plt.plot(alpha[1:], iter[1:])
plt.savefig(r"data/result/alpha.png", bbox_inches='tight')
