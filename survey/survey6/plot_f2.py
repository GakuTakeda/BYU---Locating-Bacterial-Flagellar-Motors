import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

df = pd.read_csv("f2_results.csv")
plot = np.linspace(0, 69, 70)

plt.figure(figsize=(10, 6))
plt.plot(plot, df["max_f2"])
plt.show()
plt.savefig("f2_curve_th: 8000")