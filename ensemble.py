import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classfifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.classifier_list = []
        self.alpha = np.zeros(n_weakers_limit)

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        w = np.ones(X.shape[0])
        w = w * (1 / X.shape[0])

        for i in range(self.n_weakers_limit):
            print(i)
            basicclassfier = DecisionTreeClassifier(max_depth=1)
            basicclassfier.fit(X, y, sample_weight=w)
            self.classifier_list.append(basicclassfier)
            y_predict = basicclassfier.predict(X)
            epsilon = np.sum(w[y != y_predict])
            if epsilon > 0.5:
               break
            self.alpha[i] = 0.5*np.log((1-epsilon)/epsilon)
            temp = w * np.exp((-1) * y * y_predict )
            z = np.sum(temp)
            w = temp / z


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        h = []
        score = np.zeros(X.shape[0])
        for i in range(self.n_weakers_limit):
            h.append(self.classifier_list[i].predict(X))
            print(h[i])
            score += self.alpha[i] * h[i]
        return score


    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        score = self.predict_scores(X)
        score[score > threshold] = 1
        score[score < threshold] = -1
        return score


    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
