import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.linalg import orth

df1 = pd.read_csv('chirper-happiness.csv')

scale = StandardScaler()
df2 = scale.fit_transform(df1)

df = pd.DataFrame(data=df2, columns=["Id2","totalGroup1","totalGroup2","percent_bachelorPlus","meanvalence","households_meanIncome"])
print(df)

corr = df.drop(['Id2'], axis=1).corr(method ='pearson')

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.suptitle('Ground Truth Model Correlation Heatmap')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(18, 7))
fig.suptitle('Ground Truth Model')

sns.scatterplot(ax=axes[0, 0], data=df, x="percent_bachelorPlus", y="meanvalence", hue="totalGroup1")
sns.scatterplot(ax=axes[0, 1], data=df, x="percent_bachelorPlus", y="meanvalence", hue="totalGroup2")
sns.scatterplot(ax=axes[1, 0], data=df, x="households_meanIncome", y="meanvalence", hue="totalGroup1")
sns.scatterplot(ax=axes[1, 1], data=df, x="households_meanIncome", y="meanvalence", hue="totalGroup2")

plt.show()