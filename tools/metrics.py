import math

# calcuale F1-Score criteria
def F1_Score(TP, FP, TN, FN):
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    return 2 * Precision * Recall / (Precision + Recall + 1e-10)

# calcuale MCC criteria
def MCC(TP, FP, TN, FN):
    return (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-10)