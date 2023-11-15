import xgboost as xgb
import util.evaluate as evaluate
import time
import util.data_processing as data_processing
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":
    time_start = time.time()
    data = data_processing.read_data("./data/creditcard.csv")
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    X_train, X_test, y_train, y_test = data_processing.split_data(data)

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate.evaluate(y_test, y_pred)
    time_end = time.time()
    print("Time cost: ", time_end - time_start, "s")

