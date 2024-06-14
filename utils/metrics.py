from sklearn.metrics import confusion_matrix


def compute_pos_recall(y_true, y_pred):
    _, _, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return tp / (fn + tp)


def compute_precision(y_true, y_pred):
    _, fp, _, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return tp / (tp + fp)
