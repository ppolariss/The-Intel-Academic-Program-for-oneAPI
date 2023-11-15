from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_test, y_pred):
    # 性能
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("准确率：", accuracy)
    print("精确率：", precision)
    print("召回率：", recall)
    print("F1分数：", f1)

