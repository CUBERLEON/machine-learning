from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common.models.custom_svm import BinaryLinearSVM
from common.plot import plot_pca_2d, plot_roc
from datasets.generated_dataset import GeneratedDataset


def model_build_and_fit(x_train: np.ndarray, y_train: np.ndarray, model_class, **kwargs):
    model = model_class(**kwargs)
    model.fit(x_train, y_train)
    return model


def main():
    # dataset = IrisDataset()
    dataset = GeneratedDataset(n_samples=600, n_classes=4, random_state=25, class_sep=2.)
    # dataset = GeneratedDataset(n_samples=200, n_classes=2, random_state=13, class_sep=1.)

    features_cnt = min(20, len(dataset.features_cols))
    targets_cnt = min(20, len(dataset.data[dataset.target_col].unique()))

    features_cols = dataset.features_cols[: features_cnt]
    target_col = dataset.target_col
    targets = sorted(dataset.data[dataset.target_col].unique())[:targets_cnt]

    data = dataset.data.loc[dataset.data[target_col].isin(targets)]
    data = data[features_cols + [target_col]]
    print(f"Target classes: {targets}")

    print(f"Dataset samples shape=({data.shape}):")
    print(data.head())

    x = data.loc[:, features_cols].values
    y = data.loc[:, target_col].values

    x = StandardScaler().fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(24, 48))

    builders = {"Naive Bayes Gaussian": partial(model_build_and_fit, model_class=GaussianNB),
                "SVM Linear": partial(model_build_and_fit, model_class=SVC, kernel="linear", probability=True, gamma="scale"),
                "SVM Poly": partial(model_build_and_fit, model_class=SVC, kernel="poly", probability=True, gamma="scale"),
                "SVM rbf": partial(model_build_and_fit, model_class=SVC, kernel="rbf", probability=True, gamma="scale"),
                "Decision Tree": partial(model_build_and_fit, model_class=DecisionTreeClassifier),
                "K Nearest Neighbors (K=1)": partial(model_build_and_fit, model_class=KNeighborsClassifier, n_neighbors=1),
                "K Nearest Neighbors (K=5)": partial(model_build_and_fit, model_class=KNeighborsClassifier, n_neighbors=5),
                "Random Forest (Estimators=1)": partial(model_build_and_fit, model_class=RandomForestClassifier, n_estimators=1),
                "Random Forest (Estimators=10)": partial(model_build_and_fit, model_class=RandomForestClassifier, n_estimators=10),
                "Random Forest (Estimators=100)": partial(model_build_and_fit, model_class=RandomForestClassifier, n_estimators=100),
                "AdaBoost (Estimators=1)": partial(model_build_and_fit, model_class=AdaBoostClassifier, n_estimators=1),
                "AdaBoost (Estimators=10)": partial(model_build_and_fit, model_class=AdaBoostClassifier, n_estimators=10),
                "AdaBoost (Estimators=50)": partial(model_build_and_fit, model_class=AdaBoostClassifier, n_estimators=50),
                "Multi-Layer Perceptron": partial(model_build_and_fit, model_class=MLPClassifier)}

    if targets_cnt == 2:
        builders["Custom SVM Linear"] = partial(model_build_and_fit, model_class=BinaryLinearSVM)
        builders["Logistic Regression"] = partial(model_build_and_fit, model_class=LogisticRegression)

    for ax, (model_name, model_builder) in zip(axes.flatten(), builders.items()):
        print(f"* {model_name}:")

        model = model_builder(x_train, y_train)

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        train_acc = metrics.accuracy_score(y_train, y_train_pred)
        test_acc = metrics.accuracy_score(y_test, y_test_pred)
        print(f"Train Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")

        plot_pca_2d(x, y, ax=ax,
                    model=model, title=f"{model_name}\nAccuracy (train={train_acc:.3f}, test={test_acc:.3f})")

        if targets_cnt == 2:
            y_score = model.predict_proba(x)[:, 1]
            bin_y = label_binarize(y, targets)
            plot_roc(axes[-1, -1], bin_y, y_score, model_name)

    plt.show()


if __name__ == "__main__":
    main()
