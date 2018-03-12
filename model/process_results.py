from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def accuracy(predictions, targets):
    accuracy = accuracy_score(y_true=targets, y_pred=predictions)
    return accuracy


def roc_auc(predictions, targets):
    auc = roc_auc_score(y_true=targets, y_score=predictions)
    return auc

def compute_classification_report(predictions, targets):
    conf_matrix = confusion_matrix(targets, predictions)
    print "\nConfusion matrix:\n", conf_matrix
    print(classification_report(targets, predictions))
    df_cm = pd.DataFrame(conf_matrix)
                         # index=target_names,
    #                      columns=target_names)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


probs_file="experiments/eval_output/probabilities.txt"
classes_file="experiments/eval_output/classes.txt"

targets_file="experiments/data/dev_target.txt"

classes_preds = np.loadtxt(classes_file)
targets = np.loadtxt(targets_file)


accuracy = accuracy(classes_preds, targets)
print("Accuracy %.3f"%accuracy)

compute_classification_report(classes_preds, targets)


# auc = roc_auc(classes_preds, targets)

# print("AUC %.3f"%auc)