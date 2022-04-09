import numpy as np


class FeatureSelector(object):

    def __init__(self, z_thresh=3.5):
        self.z_thresh = z_thresh
        self.chosen_ftrs = np.nan

    def fit(self, X, y):
        cls = np.unique(y)

        y_class1 = y == cls[0]
        y_class2 = y == cls[1]

        mean_group1 = X[y_class1].mean(axis=0)
        mean_group2 = X[y_class2].mean(axis=0)

        mean_diffs = mean_group1 - mean_group2
        z_scores_diff = (mean_diffs - mean_diffs.mean())/mean_diffs.std()

        self.chosen_ftrs = np.abs(z_scores_diff) >= self.z_thresh

        z_thresh = self.z_thresh

        while not self.check_ftrs():
            z_thresh -= 0.1
            print z_thresh
            self.chosen_ftrs = np.abs(z_scores_diff) >= z_thresh

    def check_ftrs(self):
        if self.chosen_ftrs.sum() > 0:
            return True
        else:
            print 'No features survived the {} criterion! Adjusting z-threshold'.format(self.z_thresh)
            return False

    def transform(self, X, y=None):
        return X[:, self.chosen_ftrs]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    def get_params(self, deep=True):
        return {'z_thresh': self.z_thresh}

    def set_params(self, **parameters):
        self.z_thresh = parameters.get('z_thresh')
        return self
