import numpy as np

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix


def _onehot_encoding(num_actions, answer_actions):
    # onehot encoding for prediction_actions and answer_actions
    answer_actions_onehot = np.zeros((len(answer_actions), num_actions), dtype=np.int32)
    for i in range(len(answer_actions)):
        answer_actions_onehot[i, int(answer_actions[i])] = 1
    return answer_actions_onehot


def _roc_auc_values(num_actions, prediction_values, answer_actions_onehot):
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_actions):
        fpr[i], tpr[i], _ = roc_curve(answer_actions_onehot[:, i], prediction_values[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(answer_actions_onehot.ravel(), prediction_values.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([
        fpr[i] for i in range(num_actions)
    ]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_actions):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_actions

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def get_statistics(num_actions, prediction_values, prediction_actions, answer_actions):
    answer_actions_onehot = _onehot_encoding(
        num_actions, 
        answer_actions
    )

    # roc auc score
    fpr, tpr, roc_auc = _roc_auc_values(
        num_actions,
        prediction_values,
        answer_actions_onehot
    )

    report = classification_report(answer_actions, prediction_actions)
    cm = confusion_matrix(answer_actions, prediction_actions)


    return {
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "classification_report": report,
        "confusion_matrix": cm
    }, ["roc_auc", "fpr", "tpr", "classification_report", "confusion_matrix"]