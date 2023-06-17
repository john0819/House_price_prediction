import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import mxnet
from mxnet import nd, autograd, gluon

# scatter
#The function generates plots for the given features based on the plot type.
#The function plotfeats(frame, feats, kind, cols=4) takes a DataFrame frame, a list of feature names feats, a plot type kind, and an optional parameter cols which defaults to 4.
def plotfeats(frame,feats,kind,cols=4):
    rows = int(np.ceil((len(feats)) / cols))
    if rows == 1 and len(feats) < cols:
        cols = len(feats)
    # print("输入%d个特征，分%d行、%d列绘图" % (len(feats), rows, cols))
    if kind == 'hs':  # hs:hist and scatter
        fig, axes = plt.subplots(nrows=rows * 2, ncols=cols, figsize=(cols * 5, rows * 10))
    else:
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 5))
        fig.subplots_adjust(hspace=0.5)
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.reshape(rows, cols)  # 当 rows=1 时，axes.shape:(cols,)，需要reshape一下
    i = 0
    for f in feats:
        # print(int(i/cols),i%cols)
        if kind == 'hist':
            # frame.hist(f,bins=100,ax=axes[int(i/cols),i%cols])
            frame.plot.hist(y=f, bins=100, ax=axes[int(i / cols), i % cols])
        elif kind == 'scatter':
            frame.plot.scatter(x=f, y='total cost', ylim=(0, 4000000), ax=axes[int(i / cols), i % cols]) # 800000
        elif kind == 'hs':
            frame.plot.hist(y=f, bins=100, ax=axes[int(i / cols) * 2, i % cols])
            frame.plot.scatter(x=f, y='total cost', ylim=(0, 4000000), ax=axes[int(i / cols) * 2 + 1, i % cols]) # 4000000
        elif kind == 'box':
            frame.plot.box(y=f, ax=axes[int(i / cols), i % cols])
        elif kind == 'boxp':
            sns.boxplot(x=f, y='total cost', data=frame, ax=axes[int(i / cols), i % cols])
        i += 1
    plt.show()

# encodes a categorical feature into numerical values based on the mean target value of each category
def encode(frame, feature, targetfeature='total cost'):
    ordering = pd.DataFrame()
    # 找出指定特征的水平值，并做临时df的索引（Find the unique values of the specified feature and set them as the index of a temporary dataframe）
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    # 按各水平分组，并求每组房价的均值（Group by each level of the feature and calculate the mean value of the target feature for each group）
    ordering['price_mean'] = frame[[feature, targetfeature]].groupby(feature).mean()[targetfeature]
    # 排序并为order列赋值1、2、3、……（Sort the dataframe based on the mean target value and assign an order number to each category）
    ordering = ordering.sort_values('price_mean')
    ordering['order'] = range(1, ordering.shape[0]+1)
    ordering = ordering['order'].to_dict()
    return ordering

# performs an ANOVA (Analysis of Variance)
def anovaXY(data):
    samples = []
    X = data.columns[0]
    Y = data.columns[1]
    for level in data[X].unique():
        if (type(level) == float): # np.NaN 的特殊处理
            s = data[data[X].isnull()][Y].values
        else:
            s = data[data[X] == level][Y].values
        samples.append(s)
    f,p = stats.f_oneway(*samples) # 也能用指针
    return (f,p)

#calculates the correlation between variables and house price using Spearman's rank correlation coefficient
def spearman(frame, features):
    '''
    采用“斯皮尔曼等级相关”来计算变量与房价的相关性
    '''
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['corr'] = [frame[f].corr(frame['total cost'], 'spearman') for f in features] # 此处用的是 Series.corr()
    spr = spr.sort_values('corr')
    plt.figure(figsize=(6, 0.2*len(features)))
    sns.barplot(data=spr, y='feature', x='corr', orient='h')
    plt.show()

square_loss = gluon.loss.L2Loss()

#computes the root mean squared error (RMSE) between the logarithm of the predicted values
def get_rmse_log(net, X_train, y_train):
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train),1,float('inf'))
    return np.sqrt( 2 * nd.sum(square_loss(nd.log(clipped_preds), nd.log(y_train))).asscalar()/num_train)


#k-fold cross-validation and returns the average loss and standard deviation of the training and test sets
def k_fold_cross_valid(k, epochs, X_train, y_train, learning_rate, weight_decay, units=128, dropout=0.1, savejpg=False):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    train_loss_std_sum = 0.0
    test_loss_std_sum = 0.0

    cols = k
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 5))

    for test_i in range(k):
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)

        net = get_net(units=units, dropout=dropout)
        train_loss, test_loss = train(
            net, X_val_train, y_val_train, X_val_test, y_val_test,
            epochs, learning_rate, weight_decay)
        print("%d-fold \tTrain loss:%f \tTest loss: %f" % (test_i + 1, train_loss[-1], test_loss[-1]))

        axes[test_i % cols].plot(train_loss, label='train')
        axes[test_i % cols].plot(test_loss, label='test')

        train_loss_sum += np.mean(train_loss[-10:])
        test_loss_sum += np.mean(test_loss[-10:])

        train_loss_std_sum += np.std(train_loss[10:])
        test_loss_std_sum += np.std(test_loss[10:])

    print("%d-fold Avg: train loss: %f, Avg test loss: %f, Avg train lost std: %f, Avg test lost std: %f" %
          (k, train_loss_sum / k, test_loss_sum / k, train_loss_std_sum / k, test_loss_std_sum / k))

    if savejpg:
        # plt.savefig("~/house-prices/%d-%d-%.3f-%d-%d-%.3f.jpg" %(k,epochs,learning_rate,weight_decay,units,dropout))
        plt.close()
    else:
        plt.show()

    return train_loss_sum / k, test_loss_sum / k, train_loss_std_sum / k, test_loss_std_sum / k


