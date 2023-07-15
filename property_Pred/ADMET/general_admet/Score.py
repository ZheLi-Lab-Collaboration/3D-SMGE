# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 14:09
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

from tdc import Evaluator

def ADMET_score(y, pred_y, eval_type, **kwargs):
    """

    Args:
        y: real item
        pred_y: pred item from model
        eval_type: the type of evaluation
                    >> "MAE"
                    >> "Spearman"
                    >> "PR-AUC"
                    >> "ROC-AUC"
        **kwargs: the other parameters

    Returns: the score from the evaluation of model

    """
    evaluator = Evaluator(name=eval_type)  # Spearman, PR-AUC, ROC-AUC
    # y_true: [0.8, 0.7, ...]; y_pred: [0.75, 0.73, ...]
    pred = evaluator(y, pred_y)

    return pred