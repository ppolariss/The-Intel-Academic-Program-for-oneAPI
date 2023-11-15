from util.data_processingIntel import read_data, split_data
from sklearn.ensemble import RandomForestClassifier
from util.evaluate import evaluate
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

import modin.config as cfg
cfg.StorageFormat.put('hdk')
from sklearnex import patch_sklearn
patch_sklearn()



def cluster(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kmeans = KMeans(n_clusters=2, random_state=42)
    y_train_cluster = kmeans.fit_predict(X_train_scaled)
    y_test_cluster = kmeans.predict(X_test_scaled)

    return X_train_scaled, y_train_cluster, X_test_scaled, y_test_cluster


if __name__ == "__main__":
    time_start = time.time()
    file_path = "./data/creditcard.csv"
    data = read_data(file_path)

    X_train, X_test, y_train, y_test = split_data(data)

    X_train, y_train, X_test, y_test = cluster(X_train, y_train, X_test, y_test)
    param_grid = {
        "min_samples_split": range(2, 10),
        "n_estimators": [10, 50, 100, 150],
        "max_depth": [5, 10, 15, 20],
        "max_features": [5, 10, 20],
    }
    scorers = {
        "precision_score": make_scorer(precision_score),
        "recall_score": make_scorer(recall_score),
        "accuracy_score": make_scorer(accuracy_score),
    }
    classifier = RandomForestClassifier(
        criterion="entropy", oob_score=True, random_state=42
    )
    refit_score = "precision_score"
    skf = StratifiedKFold(n_splits=3)
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        refit=refit_score,
        cv=skf,
        return_train_score=True,
        scoring=scorers,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(rf.feature_importances_)
    probs = rf.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_pred)
    time_end = time.time()
    print("Time cost: ", time_end - time_start, "s")
