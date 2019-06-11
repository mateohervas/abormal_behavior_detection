import numpy as np
import matplotlib.pyplot as plt

data_MLP = np.array([
	0.883,
	0.881,
	0.879,
	0.877,
	0.875,
	0.873,
	0.872,
	0.869,
	0.867,
	0.865])

data_KN = np.array([
	0.833,
	0.842,
	0.836,
	0.850,
	0.820,
	0.843,
	0.838,
	0.839,
	0.829,
	0.814])

data_Bayes = np.array([
	0.664,
	0.660,
	0.667,
	0.665,
	0.653,
	0.655,
	0.665,
	0.668,
	0.661,
	0.671])

data_SVM = np.array([
	0.850,
	0.859,
	0.855,
	0.849,
	0.851,
	0.854,
	0.858,
	0.850,
	0.852,
	0.857])

data = [data_MLP,data_KN,data_SVM, data_Bayes]
fig1, ax1 = plt.subplots()


ax1.boxplot(data)

ax1.set_xticklabels(['MLP','K Neighbors','SVM','Naive Bayes'], fontsize=8)
ax1.set_xlabel('Algorithms')
ax1.set_ylabel('AUC')
plt.show()