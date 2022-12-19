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

X = df.drop(['Id2', 'meanvalence'], axis=1)
y = df['meanvalence']

Z = df.drop(['Id2', 'meanvalence', 'totalGroup1', 'totalGroup2'], axis=1)
w = y.copy()

Z_train, Z_test, w_train, w_test = train_test_split(Z, w, test_size=0.3, random_state=1)
Z_train = sm.add_constant(Z_train)
results1 = sm.OLS(w_train, Z_train).fit()
print(results1.summary())
Z = sm.add_constant(Z)
wpred = results1.predict(Z)
# print(wpred)

result1 = pd.concat([X, wpred], axis=1)
result1 =  result1.rename(columns={0: "meanvalence"})
# print(result1)

new_threshold = (5.8 - df1['meanvalence'].mean())/(df1['meanvalence'].std(axis = 0))
print(new_threshold)

corr1 = result1.corr(method ='pearson')
sns.heatmap(corr1, xticklabels=corr1.columns, yticklabels=corr1.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.suptitle('Ethnic Group Blind Model Correlation Heatmap')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(18, 7))
fig.suptitle('Ethnic Group Blind Model')

graph = sns.scatterplot(ax=axes[0, 0], data=result1, x="percent_bachelorPlus", y="meanvalence", hue="totalGroup1")
graph.axhline(new_threshold)
graph1 = sns.scatterplot(ax=axes[0, 1], data=result1, x="percent_bachelorPlus", y="meanvalence", hue="totalGroup2")
graph1.axhline(new_threshold)
graph2 = sns.scatterplot(ax=axes[1, 0], data=result1, x="households_meanIncome", y="meanvalence", hue="totalGroup1")
graph2.axhline(new_threshold)
graph3 = sns.scatterplot(ax=axes[1, 1], data=result1, x="households_meanIncome", y="meanvalence", hue="totalGroup2")
graph3.axhline(new_threshold)
plt.show()