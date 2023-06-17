import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import sys
from sklearn import preprocessing
from function import *

# read csv file
df_train = pd.read_csv("v_train_data.csv")
df_test = pd.read_csv("v_test_data.csv")
# delete total cost
df_allX = pd.concat([df_train.loc[:, 'number of rooms':'exchange rate'],
                    df_test.loc[:]])

df_allX = df_allX.reset_index(drop=True)
print(df_allX.columns)
print(df_train.shape, df_test.shape, df_allX.shape)
# print(df_train.describe())

# numeric data
feats_numeric = df_allX.dtypes[df_allX.dtypes != "object"].index.values
# object data
feats_object = df_allX.dtypes[df_allX.dtypes == "object"].index.values
# result: (13,) (4,)
# print(feats_numeric.shape,feats_object.shape)
# discrete data
feats_numeric_discrete  = ['number of rooms','residence space','building space','noise level', 'waterfront', 'view', 'air quality level', 'aboveground space ','basement space','building year', 'decoration year','district','city', 'zip code', 'region']
# feats_numeric_discrete = ['number of rooms','residence space','building space','noise level', 'waterfront', 'view', 'air quality level', 'aboveground space ','basement space','building year', 'decoration year']

feats_continu = feats_numeric.copy()
feats_discrete = feats_object.copy()

for f in feats_numeric_discrete:
    feats_continu = np.delete(feats_continu,np.where(feats_continu == f))
    feats_discrete = np.append(feats_discrete,f)
# result : (2,) (19,) 2-continuous 19-discrete
# print(feats_continu.shape, feats_discrete.shape)
# -----------scatter
# continuous data scatter image
# plotfeats(df_train,feats_continu,kind='scatter',cols=6)
# continuous data numeric image
# feats_numeric_discrete = ['number of rooms','residence space','building space','noise level', 'waterfront', 'view', 'air quality level', 'aboveground space ','basement space','building year', 'decoration year']
# plotfeats(df_train,feats_numeric_discrete,kind='scatter',cols=4)

# ----------data analyze
# Normality test
# skewness
# result : 1.94835687547662
# print(df_train.skew()['total cost'])
# df_train['total cost'].plot(kind='hist',y='total cost',bins=100)
# stats.probplot(df_train['total cost'], plot=plt)
# df_train['total cost'].apply(lambda x: np.log1p(x)).plot(kind='hist',y='total cost',bins=100)
# plt.show()

# Calculate the deviation of each column itself
# skewed = df_allX[feats_numeric].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
# print(skewed)
# plotfeats(df_train,skewed[:6].index,kind='hs',cols=6)
# Kurtosis analysis
# print(df_train.kurt()['total cost'])
# Calculate the kurtosis of each column itself
# kurted = df_allX[feats_numeric].kurt().sort_values(ascending=False)
# print(kurted[:10])
# Comparison of histogram and scatter plot (between SalePrice) to show kurtosis
# plotfeats(df_train,kurted[:6].index,kind='hs',cols=6)

# Dispersion analysis
# plotfeats(df_train,feats_numeric,kind='box',cols=5)

# Dispersion
# plotfeats(df_train, feats_numeric_discrete, kind='boxp', cols=6)

# plotfeats(df_train, feats_object, kind='boxp', cols=6)

# chi-square test
# Analysis of Variance
# ========= Test =========

# a = np.random.random(size=(1000,))
# b = np.random.random(size=1000,)
# f,p = stats.f_oneway(a, b)
# print(f,p)
#
# a = np.random.randn(1000,)
# b = np.random.randn(1000,)
# f,p = stats.f_oneway(a, b)
# print(f,p)
#
# a = np.random.randint(1,10,size=1000,)
# b = np.random.randint(1,10,size=1000,)
# f,p = stats.f_oneway(a, b)
# print(f,p)
#
# a = np.random.randint(1,10,size=1000,)
# b = np.random.randint(5,15,size=1000,)
# f,p = stats.f_oneway(a, b)
# print(f,p)
#
# a = np.random.binomial(5,0.2,size=1000)
# b = np.random.randn(1000,)
# f,p = stats.f_oneway(a, b)
# print(f,p)

# ========= Test Over =========

# feature and label relationship by variation
# df = pd.DataFrame(columns=('feature','f','p','logp'))
# df['feature'] = feats_discrete
# for fe in feats_discrete:
#     data = pd.concat([df_train[fe],df_train['total cost']],axis=1)
#     f,p = anovaXY(data)
#     df.loc[df[df.feature==fe].index,'f'] = f
#     df.loc[df[df.feature==fe].index,'p'] = p
#     df.loc[df[df.feature==fe].index,'logp'] = 1000 if (p==0) else np.log(1./p)
#
# plt.figure(figsize=(10,4))
# sns.barplot(data=df.sort_values('p'), x='feature', y='logp')
# plt.xticks(rotation=90)
# plt.show()

# Spielman Rating Related
# spearman(df_train, np.delete(df_train.columns.values,-1))

# Analysis of covariance
corr_pearson = df_train.corr(method='pearson')
corr_spearman = df_train.corr(method='spearman')
# result: (14, 14)
# print(corr_pearson.shape)
# result: (14, 14)
# print(corr_spearman.shape)

# Covariance heatmap
# plt.figure(figsize=(20, 20))
# plt.subplot(211)
# sns.heatmap(corr_pearson, vmax=.8, square=True)
# sns.heatmap(corr_spearman, vmax=.8, square=True)
# plt.show()

# pairplot
feats_d = corr_pearson.nlargest(8,'total cost').index
# result: Index(['total cost', 'residence space', 'aboveground space ',
#        'security level of the community', 'number of rooms', 'basement space',
#        'view', 'noise level'],
#       dtype='object')
# print(feats_d)
# sns.pairplot(df_train[feats_d],size=2.5)
# plt.show()

# ----------data process
# useless data process

# Outlier Processing
# feats_away = ['building space','aboveground space ','basement space','building year']
# plotfeats(df_train,feats_away,kind='scatter')

# Normalization
# df_allX[feats_numeric] = df_allX[feats_numeric].apply(lambda x:(x-x.mean())/(x.std()))
# plt.figure(figsize=(16,10))
#
# plt.subplot(121)
# sns.boxplot(data=df_allX[feats_continu],orient="h")
#
# plt.subplot(122)
# sns.boxplot(data=df_allX[feats_discrete],orient="h")

# Discrete Volume Coding
# One-Hot Encoding
# result: (4401, 17)
# print(df_allX.shape)

# process city data
# result: {'Skykomish': 1, 'Pacific': 2, 'Algona': 3, 'Tukwila': 4, 'SeaTac': 5, 'Covington': 6, 'Milton': 7, 'Enumclaw': 8, 'Federal Way': 9, 'Auburn': 10, 'Kent': 11, 'Des Moines': 12, 'Maple Valley': 13, 'Black Diamond': 14, 'Ravensdale': 15, 'North Bend': 16, 'Renton': 17, 'Shoreline': 18, 'Duvall': 19, 'Burien': 20, 'Inglewood-Finn Hill': 21, 'Preston': 22, 'Vashon': 23, 'Kenmore': 24, 'Lake Forest Park': 25, 'Bothell': 26, 'Normandy Park': 27, 'Carnation': 28, 'Seattle': 29, 'Snoqualmie Pass': 30, 'Snoqualmie': 31, 'Issaquah': 32, 'Woodinville': 33, 'Kirkland': 34, 'Redmond': 35, 'Newcastle': 36, 'Sammamish': 37, 'Fall City': 38, 'Bellevue': 39, 'Mercer Island': 40, 'Clyde Hill': 41, 'Medina': 42, nan: 43}
# encode(df_train,'city')
# print(encode(df_train,'city'))

# ------Machine Learning
# training process
# randomly train

# select dataset
num_train = df_train.shape[0]
X_train = nd.array(df_train)
X_test = nd.array(df_test)
y_train = nd.array(np.ravel(df_train.iloc[:, -1].values))

# k=5
# epochs=50
# learning_rate=5
# weight_decay=0
# units=0
# dropout=0
#
# train_avg_loss, test_avg_loss, train_avg_loss_std, test_avg_loss_std = k_fold_cross_valid(
#     k, epochs, X_train, y_train, learning_rate, weight_decay, units, dropout, savejpg=False)







