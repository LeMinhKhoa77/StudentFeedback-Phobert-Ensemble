from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(y_true_dict, y_pred_dict):

    metrics = {}
    for task in y_true_dict:
        metrics[f"{task}_f1"] = f1_score(y_true_dict[task], y_pred_dict[task], average="macro")
        metrics[f"{task}_acc"] = accuracy_score(y_true_dict[task], y_pred_dict[task])
    return metrics
