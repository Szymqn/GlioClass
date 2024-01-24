import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, f1_score, matthews_corrcoef, mean_squared_error


def plot_classifier_results(y_test: np.ndarray, results: dict):
    methods = list(results.keys())
    predictions = list(results.values())

    scores = [accuracy_score(y_test, prediction) for prediction in predictions]

    plt.bar(methods, scores)
    plt.ylim(0.5, 1)
    plt.yticks(np.arange(0.5, 1.01, 0.05))

    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy Score')
    plt.title('Classifiers Accuracy Scores')

    plt.xticks(rotation=90)

    plt.show()


def plot_ensemble_results(features: str, ensemble_results: list):
        methods = ['Voting', 'Bagging', 'Stacking']
        scores = ensemble_results

        plt.bar(methods, scores, color=['red', 'green', 'blue'])
        plt.ylim(0.5, 1)
        plt.yticks(np.arange(0.5, 1.01, 0.05))

        plt.xlabel('Ensemble Methods')
        plt.ylabel('Accuracy Score')
        plt.title(f'{features} Accuracy Scores')

        plt.show()


class PerformanceMetrics:
    def __init__(self, y_test, y_pred, fold=1):
        self.y_test = y_test
        self.y_pred = y_pred
        self.fold = fold

    def confusionMatrix(self):
        return "Confusion matrix:" + str(confusion_matrix(self.y_test, self.y_pred))

    def accuracyScore(self):
        acc = []

        if self.fold != 1:
            for f in range(self.fold):
                acc.append(accuracy_score(self.y_test[f], self.y_pred[f]))
        else:
            acc.append(accuracy_score(self.y_test, self.y_pred))

        return "ACC:" + str(np.mean(acc))

    def roc_auc(self):
        roc_auc = []

        for f in range(self.fold):
            fpr, tpr, thresholds = roc_curve(self.y_test[f], self.y_pred[f])
            print(auc(fpr, tpr))
            roc_auc.append(auc(fpr, tpr))

        return "AUC:" + str(np.mean(roc_auc))

    def f1_score(self):
        f1_scores = []

        if self.fold != 1:
            for f in range(self.fold):
                f1_scores.append(f1_score(self.y_test[f], self.y_pred[f]))
        else:
            f1_scores.append(f1_score(self.y_test, self.y_pred))

        return "F1 score:" + str(np.mean(f1_scores))

    def matthewsCorrcoef(self):
        mc_scores = []

        if self.fold != 1:
            for f in range(self.fold):
                mc_scores.append(matthews_corrcoef(self.y_test[f], self.y_pred[f]))
        else:
            mc_scores.append(matthews_corrcoef(self.y_test, self.y_pred))

        return "MCC:" + str(np.mean(mc_scores))

    def sd(self):
        sd_scores = []

        if self.fold != 1:
            for f in range(self.fold):
                sd_scores.append(np.std(self.y_test[f], self.y_pred[f]))
        else:
            sd_scores.append(np.std(self.y_test, self.y_pred))

        return "SD:" + str(np.mean(sd_scores))

    def mse(self):
        mse_scores = []

        if self.fold != 1:
            for f in range(self.fold):
                mse_scores.append(mean_squared_error(self.y_test[f], self.y_pred[f]))
        else:
            mse_scores.append(mean_squared_error(self.y_test, self.y_pred))

        return "MSE:" + str(np.mean(mse_scores))

    def allMetrics(self):
        return [
            # self.confusionMatrix(),
            self.accuracyScore(),
            self.roc_auc(),
            self.f1_score(),
            self.matthewsCorrcoef(),
            # self.sd(),
            self.mse()
        ]
