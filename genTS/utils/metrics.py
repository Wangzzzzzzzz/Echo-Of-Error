from sklearn.metrics import confusion_matrix

def calculate_fnr_fpr(y_pred, y_true):
    """Calculate False Alarm Rate and Escape Rate."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_alarm_rate = fp / (fp + tn)
    escape_rate = fn / (fn + tp)
    return false_alarm_rate, escape_rate
