from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, balanced_accuracy_score
import numpy as np
import inspect
from collections import OrderedDict


def ppv(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=y_true.max())


def npv(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=y_true.min())


def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=y_true.max())


def specificity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=y_true.min())


def accuracy_class0(y_true, y_pred):
    # psych
    # noinspection PyTypeChecker
    return np.sum((y_true == 0) & (y_pred == 0))/np.sum(y_true == 0, dtype=np.float)


def accuracy_class1(y_true, y_pred):
    # neurol
    # noinspection PyTypeChecker
    return np.sum((y_true == 1) & (y_pred == 1))/np.sum(y_true == 1, dtype=np.float)


def accuracy_class2(y_true, y_pred):
    # ftd
    # noinspection PyTypeChecker
    return np.sum((y_true == 2) & (y_pred == 2))/np.sum(y_true == 2, dtype=np.float)


def multiclass_accuracy(y_true, y_pred):
    return (accuracy_class0(y_true, y_pred) + accuracy_class1(y_true, y_pred) + accuracy_class2(y_true, y_pred))/3.


class Evaluator(object):

    def __init__(self, multiclass=False):
        self.multiclass = multiclass
        self.evaluations = self.set_evaluations()
        self.results = OrderedDict()
        self.evaluation_string = ''

    def evaluate(self, **kwargs):
        for eval_label, eval_fun in self.evaluations.iteritems():
            args_to_use = set(inspect.getargspec(eval_fun).args) & set(kwargs.keys())
            args_to_use = {key: kwargs[key] for key in args_to_use}
            self.results[eval_label] = eval_fun(**args_to_use)

    def evaluate_prediction(self, **kwargs):
        self.evaluate(**kwargs)
        return self.results.values()

    def evaluate_labels(self):
        return self.evaluations.keys()

    def print_evaluation(self, logger):
        if not self.results:
            raise RuntimeError('evaluate has to be run first')

        if self.multiclass:
            self.evaluation_string = 'Accuracy: {balanced_accuracy:.2f}, Accuracy class 0: {acc_class_0:.2f} ' \
                                     'Accuracy class 1: {acc_class_1:.2f}, ' \
                                     'Accuracy class 2: {acc_class_2:.2f}'.format(**self.results)
        else:
            self.evaluation_string = 'Accuracy: {balanced_accuracy:.2f}, AUC: {AUC:.2f}, F1-score: {F1:.2f}, Recall: ' \
                                     '{recall:.2f}, Precision: {precision:.2f}, Sensitivity: {sensitivity:.2f}, ' \
                                     'Specificity: {specificity:.2f}, ' \
                                     'PPV: {positive_predictive_value:.2f}, ' \
                                     'NPV: {negative_predictive_value:.2f}'.format(**self.results)
        logger.debug(self.evaluation_string)

    def set_evaluations(self):
        if self.multiclass:
            evals = OrderedDict([('accuracy', accuracy_score),
                                 ('balanced_accuracy', multiclass_accuracy),
                                 ('acc_class_0', accuracy_class0),
                                 ('acc_class_1', accuracy_class1),
                                 ('acc_class_2', accuracy_class2)])
        else:
            evals = OrderedDict([('accuracy', accuracy_score),
                                 ('balanced_accuracy', balanced_accuracy_score),
                                 ('AUC', roc_auc_score),
                                 ('F1', f1_score),
                                 ('recall', recall_score),
                                 ('precision', precision_score),
                                 ('sensitivity', recall_score),           # recall is the same as sensitivity
                                 ('specificity', specificity),            # i.e. recall for the negative class
                                 ('positive_predictive_value', ppv),      # same as precision
                                 ('negative_predictive_value', npv)])     # i.e. precision for the negative class
        return evals
