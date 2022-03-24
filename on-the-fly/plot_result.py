import numpy as np
import matplotlib.pyplot as plt


ONE_hist = np.load("./error_hist_ONE.npy")
baseline_hist = np.load("./error_hist_baseline.npy")

plt.plot(ONE_hist, label='ONE')
plt.plot(baseline_hist, label='Baseline')
plt.legend()
plt.title("Top-1 Error in Test set")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()