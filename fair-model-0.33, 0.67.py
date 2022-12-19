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

# corr = result.drop(['const'], axis=1).corr(method ='pearson')
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

new_threshold = (5.8 - df1['meanvalence'].mean())/(df1['meanvalence'].std(axis = 0))
# print(new_threshold)

protected_columns = list(result.columns)[1:3]
# print(protected_columns)

target_column = result.columns[5]
# print(target_column)

def gen_latent_fast(df0, prot_col, tar_col):
    """
    generate a representation for target column which is independent from any columns in prot_col
    df0: a data frame
    prot_col: list of strings, the protected columns
    tar_col: string, the target (outcome) column
    """
    df = df0.copy()
    for column in df.columns:
        df[column] = df[column] - df[column].mean()
    df_protect = df[prot_col]
    dfv_protect = df_protect.values
    dfv_target = df[tar_col].values
#     base_protect = scipy.linalg.orth(dfv_protect)
    base_protect = orth(dfv_protect)

    for i in range(base_protect.shape[1]):
        #print(base_protect[:,i].shape)
        dfv_target = dfv_target - np.inner(dfv_target, base_protect[:,i])*base_protect[:,i]
    return dfv_target
"""
dfv_target = gen_latent_fast(result, protected_columns, target_column)
# print(dfv_target)

dfv_target = pd.DataFrame(data=dfv_target, columns=["mean_valence"])

result_fair = pd.concat([result, dfv_target], axis=1)
result_fair = result_fair.drop(['meanvalence', 'const'], axis = 1)

corr_fair = result_fair.corr(method ='pearson')
sns.heatmap(corr_fair, xticklabels=corr_fair.columns, yticklabels=corr_fair.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

sns.scatterplot(data=result_fair, x="percent_bachelorPlus", y="mean_valence", hue="totalGroup1")
sns.scatterplot(data=result_fair, x="percent_bachelorPlus", y="mean_valence", hue="totalGroup2")
sns.scatterplot(data=result_fair, x="households_meanIncome", y="mean_valence", hue="totalGroup1")
sns.scatterplot(data=result_fair, x="households_meanIncome", y="mean_valence", hue="totalGroup2")
"""
def gen_latent_nonparam_regula(df0, prot_col, tar_col, lbd):
    """
    generate a fair representation at a certain level define by lbd
    df0: a data frame
    prot_col: list of strings, the protected columns
    tar_col: string, the target (outcome) column
    lbd: float number between 0 and 1, 0 means totally fair; 1 means same as outcome
    """
    dfv_target = df0[tar_col].values
    dfv_mean = df0[tar_col].mean()
    dfv_target = dfv_target - dfv_mean
    latent0 = gen_latent_fast(df0, prot_col, tar_col)
    lb = np.full(shape=2110,fill_value=lbd,dtype=np.int)
    z = latent0 + lb*(dfv_target - latent0)
    return z

gen_latent_nonparam_regula0 = gen_latent_nonparam_regula(result, protected_columns, target_column, 0.33)
# print(gen_latent_nonparam_regula0)

gen_latent_nonparam_regula0 = pd.DataFrame(data=gen_latent_nonparam_regula0, columns=["mean_valence"])

result_fair0 = pd.concat([result, gen_latent_nonparam_regula0], axis=1)
result_fair0 = result_fair0.drop(['meanvalence', 'const'], axis = 1)
# print(result_fair0)

corr_fair0 = result_fair0.corr(method ='pearson')
sns.heatmap(corr_fair0, xticklabels=corr_fair0.columns, yticklabels=corr_fair0.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.suptitle('Fair Model (lambda=0.33) Correlation Heatmap')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(18, 7))
fig.suptitle('Fair Model (lambda=0.33)')

sns.scatterplot(ax=axes[0, 0], data=result_fair0, x="percent_bachelorPlus", y="mean_valence", hue="totalGroup1")
sns.scatterplot(ax=axes[0, 1], data=result_fair0, x="percent_bachelorPlus", y="mean_valence", hue="totalGroup2")
sns.scatterplot(ax=axes[1, 0], data=result_fair0, x="households_meanIncome", y="mean_valence", hue="totalGroup1")
sns.scatterplot(ax=axes[1, 1], data=result_fair0, x="households_meanIncome", y="mean_valence", hue="totalGroup2")
plt.show()

gen_latent_nonparam_regula1 = gen_latent_nonparam_regula(result, protected_columns, target_column, 0.67)
# print(gen_latent_nonparam_regula1)

gen_latent_nonparam_regula1 = pd.DataFrame(data=gen_latent_nonparam_regula1, columns=["mean_valence"])

result_fair1 = pd.concat([result, gen_latent_nonparam_regula1], axis=1)
result_fair1 = result_fair1.drop(['meanvalence', 'const'], axis = 1)
# print(result_fair1)

corr_fair1 = result_fair1.corr(method ='pearson')
sns.heatmap(corr_fair1, xticklabels=corr_fair1.columns, yticklabels=corr_fair1.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.suptitle('Fair Model (lambda=0.67) Correlation Heatmap')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(18, 7))
fig.suptitle('Fair Model (lambda=0.67)')

sns.scatterplot(ax=axes[0, 0], data=result_fair1, x="percent_bachelorPlus", y="mean_valence", hue="totalGroup1")
sns.scatterplot(ax=axes[0, 1], data=result_fair1, x="percent_bachelorPlus", y="mean_valence", hue="totalGroup2")
sns.scatterplot(ax=axes[1, 0], data=result_fair1, x="households_meanIncome", y="mean_valence", hue="totalGroup1")
sns.scatterplot(ax=axes[1, 1], data=result_fair1, x="households_meanIncome", y="mean_valence", hue="totalGroup2")
plt.show()
