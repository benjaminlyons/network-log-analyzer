import sklearn.linear_model
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
import sklearn.tree
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

import copy
import random

def preprocess(file_location):
    df = pd.read_csv(file_location)
    df = df.replace(["default", "datacenter"], [0,1])
    df = df.replace(["normal", "outlier"], [0,1]) # 0 is normal, 1 is outlier
    df = df.replace(["E", "F"], [0, 1])
    stats = df.columns[2:-1].tolist()
    df = df[stats + ["classification"]]
    df.replace('', np.nan, inplace=True)
    df = df.dropna()
    train = df.sample(frac=0.8)
    train_x = train[stats].to_numpy()
    train_y = train["classification"].to_numpy()
    validation = df.drop(train.index).sample(frac=1)
    val_x = validation[stats].to_numpy()
    val_y = validation["classification"].to_numpy()
    return stats, (train_x, train_y), (val_x, val_y)

def remove_col_arr(arr, col):
    #print(np.shape(arr))
    return np.delete(arr, col, 1)

#delete column of stats and data one at a time
def get_list_increasing_importance(model, new_model_call, stats, train_x, train_y, val_x, val_y):
    #print(stats, "haha")
    ret = []
    dropped = 0
    model = eval(new_model_call)
    model.fit(train_x, train_y)
    pred_y = model.predict(val_x)
    f1 = f1_score(val_y, pred_y)
    acc = model.score(val_x, val_y)
    ret.append({'no_drop': (f1, acc)})
    #print(stats, "haha", ret)
    while True:
        #print(ret)
        #print(np.shape(train_x))
        max_acc = 0
        max_col_name = ''
        max_f1 = -1
        for i in range(0, len(stats)):
            if len(stats) == 1:
                break
            temp_train_x = np.delete(train_x, i, 1)
            temp_val_x = np.delete(val_x, i, 1)
            model = eval(new_model_call)
            model.fit(temp_train_x, train_y)
            pred_y = model.predict(temp_val_x)
            f1 = f1_score(val_y, pred_y)
            acc = model.score(temp_val_x, val_y)
            print(dropped, i, f1, acc, stats[i], len(stats), len(temp_train_x[0]))
            if acc > max_acc:
                max_acc = acc
                max_col_name = stats[i]
                max_f1 = f1
        print("----------------")
        if len(stats) == 1:
            ret.append({stats[0]: (-1, -1)})
            break
        elif max_acc < 0.5:
            ret.append({max_col_name: (max_f1, max_acc)})
            break
        else:
            ret.append({max_col_name: (max_f1, max_acc)})
            rm_ind = stats.index(max_col_name)
            stats.pop(rm_ind)
            train_x = np.delete(train_x, rm_ind, 1)
            val_x = np.delete(val_x, rm_ind, 1)
            dropped += 1
    return ret



def main():
    file_loc = "data/supervised_dataset.csv"

    stats, (train_x, train_y), (val_x, val_y) = preprocess(file_loc)
    print(stats, len(train_x[0]))

    print(len(train_x), len(train_y), len(val_x), len(val_y))

    gnb = GaussianNB()

    l = get_list_increasing_importance(gnb, 'GaussianNB()', copy.deepcopy(stats), copy.deepcopy(train_x), copy.deepcopy(train_y), copy.deepcopy(val_x), copy.deepcopy(val_y))
    print('---Gaussian NB---')
    for e in l:
        for k in e:
            print(k, 'f1:', e[k][0], 'acc:', e[k][1])

    knn = KNeighborsClassifier(n_neighbors=5)

    l = get_list_increasing_importance(knn, 'KNeighborsClassifier(n_neighbors=10)', copy.deepcopy(stats), copy.deepcopy(train_x), copy.deepcopy(train_y), copy.deepcopy(val_x), copy.deepcopy(val_y))
    print('---K nearest neighbors---')
    for e in l:
        for k in e:
            print(k, 'f1:', e[k][0], 'acc:', e[k][1])

    lda = LinearDiscriminantAnalysis()

    l = get_list_increasing_importance(lda, 'LinearDiscriminantAnalysis()', copy.deepcopy(stats), copy.deepcopy(train_x), copy.deepcopy(train_y), copy.deepcopy(val_x), copy.deepcopy(val_y))
    print('---Linear Discriminant Analysis---')
    for e in l:
        for k in e:
            print(k, 'f1:', e[k][0], 'acc:', e[k][1])

    pct = Perceptron()

    l = get_list_increasing_importance(pct, 'Perceptron()', copy.deepcopy(stats), copy.deepcopy(train_x), copy.deepcopy(train_y), copy.deepcopy(val_x), copy.deepcopy(val_y))
    print('---Perceptron---')
    for e in l:
        for k in e:
            print(k, 'f1:', e[k][0], 'acc:', e[k][1])



    dtc = sklearn.tree.DecisionTreeClassifier()
    l = get_list_increasing_importance(dtc, 'sklearn.tree.DecisionTreeClassifier()', copy.deepcopy(stats), copy.deepcopy(train_x), copy.deepcopy(train_y), copy.deepcopy(val_x), copy.deepcopy(val_y))
    print('-----------------DecisionTreeClassifier--------------')
    for e in l:
        for k in e:
            print(k, 'f1:', e[k][0], 'acc:', e[k][1])

    svm = SVC()
    l = get_list_increasing_importance(svm, 'SVC()', copy.deepcopy(stats), copy.deepcopy(train_x), copy.deepcopy(train_y), copy.deepcopy(val_x), copy.deepcopy(val_y))
    print('-----------------SVC--------------')
    for e in l:
        for k in e:
            print(k, 'f1:', e[k][0], 'acc:', e[k][1])

    lgr = sklearn.linear_model.LogisticRegression()
    l = get_list_increasing_importance(lgr, 'sklearn.linear_model.LogisticRegression(max_iter=1000)', copy.deepcopy(stats), copy.deepcopy(train_x), copy.deepcopy(train_y), copy.deepcopy(val_x), copy.deepcopy(val_y))
    print('-----------------LogisticRegression--------------')
    for e in l:
        for k in e:
            print(k, 'f1:', e[k][0], 'acc:', e[k][1])

    gpc = RandomForestClassifier()
    l = get_list_increasing_importance(gpc, 'RandomForestClassifier()', copy.deepcopy(stats), copy.deepcopy(train_x), copy.deepcopy(train_y), copy.deepcopy(val_x), copy.deepcopy(val_y))
    print('-----------------RandomForestClassifier--------------')
    for e in l:
        for k in e:
            print(k, 'f1:', e[k][0], 'acc:', e[k][1])


if __name__ == "__main__":
    main()
