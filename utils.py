import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import os
def ROC(args, y_test,y_pred):
    auc=roc_auc_score(y_test,y_pred)
    print('auroc', auc)
    return auc

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

class log():
    def __init__(self) -> None:
        self.roc_auc_max = 0
        self.f1_max = 0
    def print_result(self, y_test, y_pred, model, i, args):
        y_test = np.nan_to_num(y_test)
        y_pred = np.nan_to_num(y_pred)

        auc=roc_auc_score(y_test,y_pred)
      
        if not os.path.exists("./{}{}modelroc".format(args.model, args.name,i, auc)):
            os.makedirs("./{}{}modelroc/".format(args.model,args.name, i, auc))
  
        if self.roc_auc_max < auc:
            self.roc_auc_max = auc
       
          
        print('auroc:{:.4f}'.format(auc))
        

from collections import namedtuple
EventWiseMetrics = namedtuple(
    "EventWiseMetrics", "P_ew R_ew F1_ew "
)


def make_intervals(y):
    """Find intervals of consecutive 1 in input array.

    Args:
        y (array): input array of labels or predictions. Should
        be an array of bool or 0/1 values.

    Returns:
        list: list of intervals, each as a (onset, offset) tuple.
    """
    y = np.asarray(y).astype(int)
    d = np.diff(y, prepend=0, append=0)
    (onsets,) = np.where(d == 1)
    (offsets,) = np.where(d == -1)
    return list(zip(onsets, offsets))


def compute_event_wise_metrics(y_true, y_pred, gt_intervals=None):
    """Compute event-wise metrics including composite F1 score.
    Args:
        y_true (array): ground truth anomalies.
        y_pred (array): predicted anomalies.
        gt_intervals (list, optional): ground truth intervals of anomalies
            (i.e., anomalous event). Although most of the time  this can be
            automatically computed using `y_true`, in some cases you'd want
            to explicitly pass the events to avoid that two contiguous but
            independent events be considered as one single event.

    Returns:
        namedtuple: an object with the following fields containing event-wise
        metrics: TP_ew, FP_ew, FN_ew, P_ew, R_ew, F1_ew and F1_c.
    """

    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=int)
    intervals = make_intervals(y_true) if gt_intervals is None else gt_intervals
    pred_intervals = make_intervals(y_pred)

    TP = 0
    FP = 0
    FN = 0

    # Count correctly detected events
    for onset, offset in intervals:
        if y_pred[onset:offset].any():
            TP += 1
        else:
            FN += 1

    # Count segments that don't overlap with any ground truth event

    for onset, offset in pred_intervals:
        if (~y_true[onset:offset]).all():
            FP += 1

    # Compute 1 - False Alarm Rate point-wise
    nFPR = 1 - ((~y_true) & (y_pred == 1)).sum() / (~y_true).sum()

    # Compute event-wise Precision, Recall and F1 score
    R_ew = TP / (TP + FN)
    # Compute composite F1_ew score
    P_ew = (((y_true) & (y_pred == 1)).sum() / y_pred.sum()) * nFPR # point-wise precision
    F1_ew = 2 * P_ew  * R_ew / (P_ew  + R_ew)

    return EventWiseMetrics( P_ew, R_ew, F1_ew)