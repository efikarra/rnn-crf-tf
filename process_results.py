from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
# matplotlib.rcParams['font.sans-serif'] = 'SimHei'
# matplotlib.rcParams['font.serif'] = 'SimHei'
# matplotlib.rcParams['font.family'] = "sans-serif"
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


def compute_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
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

def labels_from_predictions(predictions):
    return np.argmax(predictions, axis=1)

def process_results(params):

    predictions = np.loadtxt(params.preds_file)
    print np.sum(predictions, axis=1)
    preds_labels = labels_from_predictions(predictions)
    targets = np.loadtxt(params.targets_file)
    targets = targets.astype(int)

    accuracy = compute_accuracy(y_true=targets, y_pred=preds_labels)
    print("Accuracy %.3f" % accuracy)

    compute_classification_report(preds_labels, targets)


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--preds_file", type=str, default=None)
    parser.add_argument("--targets_file", type=str, default=None)


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    params, unparsed = parser.parse_known_args()
    process_results(params)

if __name__ == '__main__':
    main()




# auc = roc_auc(classes_preds, targets)

# print("AUC %.3f"%auc)