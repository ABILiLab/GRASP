import numpy as np
from sklearn import metrics as skmetrics
import warnings
from collections import Counter
from sklearn.metrics import hamming_loss, zero_one_loss, coverage_error
from sklearn.metrics import label_ranking_loss, average_precision_score
from sklearn.metrics import matthews_corrcoef,accuracy_score

warnings.filterwarnings("ignore")


def example_accuracy(label,predict):
    epsilon = 1e-8
    ex_and = np.sum(np.logical_and(label,predict), axis=1).astype('float32')
    ex_or = np.sum(np.logical_or(label,predict), axis=1).astype('float32')
    return np.mean(ex_and / (ex_or + epsilon))


def label_quantity(label, predict):
    tp = np.sum(np.logical_and(label, predict), axis=0)
    fp = np.sum(np.logical_and(1-label,predict),axis=0)
    tn = np.sum(np.logical_and(1-label, 1-predict),axis=0)
    fn = np.sum(np.logical_and(label, 1-predict),axis=0)

    return np.stack([tp,fp,tn,fn], axis=0).astype('float')

def label_recall(label, predict):
    epsilon = 1e-8
    quantity = label_quantity(label, predict)
    tp, fn = quantity[0], quantity[3]
    # print('label_accuracy_macro',tp_tn / (tp_fp_tn_fn + epsilon))
    print("tp:",quantity[0])
    print("fp:",quantity[1])
    print("tn:",quantity[2])
    print("fn:",quantity[3])
    return tp / (tp + fn + epsilon)

def label_accuracy(label, predict):
    epsilon = 1e-8
    quantity = label_quantity(label, predict)
    tp_tn = np.add(quantity[0], quantity[2])
    tp_fp_tn_fn = np.sum(quantity, axis=0)
    return tp_tn / (tp_fp_tn_fn + epsilon)

def AvgF1(targets, predict):
    fscore = 0
    total = 0
    p_total = 0
    p, r = 0, 0
    for yt, yp in zip(targets, predict):
        ytNum = sum(yt)
        if ytNum == 0:
            continue
        rec = sum(yp[yt == 1]) / ytNum
        r += rec
        total += 1
        ypSum = sum(yp)
        if ypSum > 0:
            p_total += 1
            pre = sum(yt[yp == True]) / ypSum
            p += pre
    r /= total
    if p_total > 0:
        p /= p_total
    return 2 * r * p / (r + p)


from sklearn import metrics
import numpy as np

def evaluate_all_metrics(targets, predict, isMultiLabel, thres=0.5):

    

    if isMultiLabel:
        predict_binary = (predict >= thres).astype(int)
        ex_acc = round(example_accuracy(targets, predict_binary),3)
        ham_loss = round(hamming_loss(targets, predict_binary),3)
        one_error = round(zero_one_loss(targets, predict_binary, normalize=True),3)
        # coverage_error / label_ranking_loss / average_precision_score require continuous y_score (sklearn)
        cov = round(coverage_error(targets, predict),3)
        rank_loss = round(label_ranking_loss(targets, predict),3)
        ap = round(average_precision_score(targets, predict, average='weighted'),3)
        # single_label_acc = label_accuracy(targets, predict_binary)
        single_label_recall = label_recall(targets, predict_binary)

        mccs = []
        for i in range(targets.shape[1]):
            mccs.append(round(matthews_corrcoef(targets[:, i], predict_binary[:,i]),3))

        avgF1 = AvgF1(targets, predict_binary)
        
        miP = round(skmetrics.precision_score(targets, predict_binary, average='micro'),3)
        maP = round(skmetrics.precision_score(targets, predict_binary, average='macro'),3)
        miR = round(skmetrics.recall_score(targets, predict_binary, average='micro'),3)
        maR = round(skmetrics.recall_score(targets, predict_binary, average='macro'),3)

        AUCs = [round(skmetrics.roc_auc_score(targets[:,i],predict[:,i]),3) for i in range(targets.shape[1])]
        avgAUC = round(np.mean(AUCs),3)
    
        metrics_dict = {
            "Example Accuracy": ex_acc,
            "Hamming Loss": ham_loss,
            "Zero-One Loss": one_error,
            "Coverage Error": cov,
            "Ranking Loss": rank_loss,
            "Average Precision Score": ap,
            # "Single Label Accuracy": single_label_acc,
            # "Single Label Recall": single_label_recall,
            "Matthews Correlation Coefficient": mccs,
            "Average F1 Score": avgF1,
            "Micro Precision": miP,
            "Micro Recall": miR,
            "Macro Precision": maP,
            "Macro Recall": maR,
            "Average AUC": avgAUC,
            "AUC": AUCs
        }
    else:
        # Assign 1 to argmax, 0 to the rest
        predict_binary = np.zeros_like(predict)
        predict_binary[np.arange(predict.shape[0]), predict.argmax(axis=1)] = 1
        ex_acc = round(example_accuracy(targets, predict_binary),3)
        # ham_loss = round(hamming_loss(targets, predict_binary),3)
        one_error = round(zero_one_loss(targets, predict_binary, normalize=True),3)
        # cov = round(coverage_error(targets, predict_binary),3)
        # rank_loss = round(label_ranking_loss(targets, predict_binary),3)
        # ap = round(average_precision_score(targets, predict_binary, average='weighted'),3)
        # single_label_acc = label_accuracy(targets, predict_binary)
        # single_label_recall = label_recall(targets, predict_binary)

        mccs = []
        for i in range(targets.shape[1]):
            mccs.append(round(matthews_corrcoef(targets[:, i], predict_binary[:,i]),3))

        avgF1 = AvgF1(targets, predict_binary)
        
        miP = round(skmetrics.precision_score(targets, predict_binary, average='micro'),3)
        maP = round(skmetrics.precision_score(targets, predict_binary, average='macro'),3)
        miR = round(skmetrics.recall_score(targets, predict_binary, average='micro'),3)
        maR = round(skmetrics.recall_score(targets, predict_binary, average='macro'),3)

        AUCs = [round(skmetrics.roc_auc_score(targets[:,i],predict[:,i]),3) for i in range(targets.shape[1])]
        avgAUC = round(np.mean(AUCs),3)
    
        metrics_dict = {
            "Example Accuracy": ex_acc,
            # "Hamming Loss": ham_loss,
            "Zero-One Loss": one_error,
            # "Coverage Error": cov,
            # "Ranking Loss": rank_loss,
            # "Average Precision Score": ap,
            # "Single Label Accuracy": single_label_acc,
            # "Single Label Recall": single_label_recall,
            "Matthews Correlation Coefficient": mccs,
            "Average F1 Score": avgF1,
            "Micro Precision": miP,
            "Micro Recall": miR,
            "Macro Precision": maP,
            "Macro Recall": maR,
            "Average AUC": avgAUC,
            "AUC": AUCs
        }
        
    return metrics_dict
