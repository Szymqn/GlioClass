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

        for f in range(self.fold):
            acc.append(accuracy_score(self.y_test[f], self.y_pred[f]))
        return "ACC:" + str(np.mean(acc))

    def roc_auc(self):
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred)

        # plt.figure()
        # plt.plot(fpr, tpr)
        # plt.plot([0, 1], [0, 1], 'r--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        # plt.show()

        return "AUC:" + str(auc(fpr, tpr))

    def f1_score(self):
        return "F1 score:" + str(f1_score(self.y_test, self.y_pred))

    def matthewsCorrcoef(self):
        return "MCC:" + str(matthews_corrcoef(self.y_test, self.y_pred))

    def sd(self):
        return "SD:" + str(np.std(self.y_pred))

    def mse(self):
        return "MSE:" + str(mean_squared_error(self.y_test, self.y_pred))

    def allMetrics(self):
        return [
            self.confusionMatrix(),
            self.accuracyScore(),
            self.roc_auc(),
            self.f1_score(),
            self.matthewsCorrcoef(),
            self.sd(),
            self.mse()
        ]
