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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = LinearRegression()
model.fit(X_train, y_train)
print("Linear Regression model score: ", model.score(X_train,y_train))

X_train = sm.add_constant(X_train)
result = sm.OLS(y_train, X_train).fit() # this is a OLS object
print(result.summary())
X = sm.add_constant(X) # add again the constant
ypred = result.predict(X) # use the predict method of the object

result = pd.concat([X, ypred], axis=1)
result =  result.rename(columns={0: "meanvalence"})

corr = result.drop(['const'], axis=1).corr(method ='pearson')

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.suptitle('Ethnic Group Aware Model Correlation Heatmap')
plt.show()

new_threshold = (5.8 - df1['meanvalence'].mean())/(df1['meanvalence'].std(axis = 0))
print(new_threshold)

fig, axes = plt.subplots(2, 2, figsize=(18, 7))
fig.suptitle('Ethnic Group Aware Model')

graph = sns.scatterplot(ax=axes[0, 0], data=result, x="percent_bachelorPlus", y="meanvalence", hue="totalGroup1")
graph.axhline(new_threshold)
graph1 = sns.scatterplot(ax=axes[0, 1], data=result, x="percent_bachelorPlus", y="meanvalence", hue="totalGroup2")
graph1.axhline(new_threshold)
graph2 = sns.scatterplot(ax=axes[1, 0], data=result, x="households_meanIncome", y="meanvalence", hue="totalGroup1")
graph2.axhline(new_threshold)
graph3 = sns.scatterplot(ax=axes[1, 1], data=result, x="households_meanIncome", y="meanvalence", hue="totalGroup2")
graph3.axhline(new_threshold)
plt.show()