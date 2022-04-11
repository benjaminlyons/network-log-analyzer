import sklearn.linear_model 
import sklearn.tree
import pandas as pd
import numpy as np

import random

def main():
    # read from file
    df = pd.read_csv("data/supervised_dataset.csv")

    # replace categorical data with numbers
    df = df.replace(["default", "datacenter"], [0,1])
    df = df.replace(["normal", "outlier"], [0,1]) # 0 is normal, 1 is outlier

    # get important info headers
    stats = df.columns[2:-2].tolist()
    stats.remove("api_access_uniqueness")
    # stats.remove("num_users")
    df = df[stats + ["classification"]]
    df.replace('', np.nan, inplace=True)
    df = df.dropna()

    train = df.sample(frac=0.8)
    train_x = train[stats].to_numpy()
    train_y = train["classification"].to_numpy()

    validation = df.drop(train.index).sample(frac=1)
    val_x = validation[stats].to_numpy()
    val_y = validation["classification"].to_numpy()

    logisticRegr = sklearn.linear_model.LogisticRegression()
    logisticRegr.fit(train_x, train_y)

    predictions = logisticRegr.predict(val_x)
    accuracy = logisticRegr.score(val_x, val_y)
    print(accuracy)

    dtc = sklearn.tree.DecisionTreeClassifier(max_depth=3)
    dtc.fit(train_x, train_y)

    rules = sklearn.tree.export_text(dtc, feature_names = stats)
    print(rules)
    predictions = dtc.predict(val_x)
    accuracy = dtc.score(val_x, val_y)
    print(accuracy)


if __name__ == "__main__":
    main()
