import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, mean_squared_error


class PerformanceMetrics:
    def __init__(self, y_test: list, y_pred: dict, time: dict, fold=1):
        self.y_test = list(y_test)
        self.y_pred = y_pred
        self.classifiers = (self.y_pred.keys())
        self.time = time
        self.fold = fold

    def confusion_matrix(self):
        cm_dict = {}
        for classifier in self.classifiers:
            cm_list = []
            if self.fold != 1:
                for f in range(self.fold):
                    list_y_pred = list(self.y_pred[classifier][f])
                    cm_list.append(confusion_matrix(self.y_test[f], list_y_pred))

                total_cm = np.sum(cm_list, axis=0)
                mean_cm = total_cm / len(cm_list)
                cm_dict[classifier] = mean_cm
                disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm)
                disp.plot(cmap='Blues')
                plt.title(classifier)
                plt.show()
            else:
                cm_dict[classifier] = confusion_matrix(self.y_test, self.y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_dict[classifier])
                disp.plot(cmap='Blues')
                plt.title(classifier)
                plt.show()

        return "Confusion matrix:" + str(cm_dict)

    def accuracy_score(self):
        acc_dict = {}

        for classifier in self.classifiers:
            acc = []
            if self.fold != 1:
                for f in range(self.fold):
                    acc.append(accuracy_score(self.y_test[f], self.y_pred[classifier][f]))
                acc_dict[classifier] = acc
            else:
                acc_dict[classifier] = accuracy_score(self.y_test, self.y_pred)

        mean_dict = {classifier: sum(values) / len(values) for classifier, values in acc_dict.items()}
        return "ACC: " + str(mean_dict), mean_dict

    def roc_auc(self):
        roc_auc_dict = {}

        for classifier in self.classifiers:
            acc = []
            if self.fold != 1:
                for f in range(self.fold):
                    acc.append(roc_auc_score(self.y_test[f], self.y_pred[classifier][f]))
                roc_auc_dict[classifier] = acc
            else:
                roc_auc_dict[classifier] = roc_auc_score(self.y_test, self.y_pred)

        mean_dict = {classifier: sum(values) / len(values) for classifier, values in roc_auc_dict.items()}
        return "Roc Auc: " + str(mean_dict)

    def f1_score(self):
        f1_score_dict = {}

        for classifier in self.classifiers:
            acc = []
            if self.fold != 1:
                for f in range(self.fold):
                    acc.append(f1_score(self.y_test[f], self.y_pred[classifier][f]))
                f1_score_dict[classifier] = acc
            else:
                f1_score_dict[classifier] = f1_score(self.y_test, self.y_pred)

        mean_dict = {classifier: sum(values) / len(values) for classifier, values in f1_score_dict.items()}
        return "F1 score: " + str(mean_dict)

    def matthews_corrcoef(self):
        matthews_corrcoef_dict = {}

        for classifier in self.classifiers:
            acc = []
            if self.fold != 1:
                for f in range(self.fold):
                    acc.append(matthews_corrcoef(self.y_test[f], self.y_pred[classifier][f]))
                matthews_corrcoef_dict[classifier] = acc
            else:
                matthews_corrcoef_dict[classifier] = matthews_corrcoef(self.y_test, self.y_pred)

        mean_dict = {classifier: sum(values) / len(values) for classifier, values in matthews_corrcoef_dict.items()}
        return "MCC: " + str(mean_dict)

    def sd(self):
        sd_dict = {}

        for classifier in self.classifiers:
            acc = []
            if self.fold != 1:
                for f in range(self.fold):
                    acc.append(np.std(self.y_test[f], self.y_pred[classifier][f]))
                sd_dict[classifier] = acc
            else:
                sd_dict[classifier] = np.std(self.y_test, self.y_pred[classifier])

        mean_dict = {classifier: sum(values) / len(values) for classifier, values in sd_dict.items()}
        return "MCC: " + str(mean_dict)

    def mse(self):
        mse_dict = {}

        for classifier in self.classifiers:
            acc = []
            if self.fold != 1:
                for f in range(self.fold):
                    acc.append(mean_squared_error(self.y_test[f], self.y_pred[classifier][f]))
                mse_dict[classifier] = acc
            else:
                mse_dict[classifier] = mean_squared_error(self.y_test, self.y_pred)

        mean_dict = {classifier: sum(values) / len(values) for classifier, values in mse_dict.items()}
        return "MSE: " + str(mean_dict)

    def plot_classifier_acc(self):
        scores_dict = self.accuracy_score()[1]

        sorted_results = sorted(zip(scores_dict.keys(), scores_dict.values()), key=lambda x: x[1], reverse=True)

        methods, scores = zip(*sorted_results)

        plt.bar(methods, scores)
        plt.ylim(0.5, 1)
        plt.yticks(np.arange(0.5, 1.01, 0.05))

        plt.xlabel('Classifiers')
        plt.ylabel('Accuracy Score')
        plt.title('Classifiers Accuracy Scores')

        plt.xticks(rotation=90)

        plt.show()

    def plot_classifier_time(self):
        sorted_results = sorted(zip(self.time.keys(), self.time.values()), key=lambda x: x[1], reverse=False)

        methods, scores = zip(*sorted_results)
        max_score = max(scores)

        plt.bar(methods, scores)
        plt.ylim(0.01, max_score)
        plt.yticks(np.arange(0.01, max_score, max_score / 10))

        plt.xlabel('Classifiers')
        plt.ylabel('Accuracy Score')
        plt.title('Classifiers Accuracy Scores')

        plt.xticks(rotation=90)

        plt.show()

    def all_metrics(self):
        return [
            # self.confusion_matrix(),
            self.accuracy_score()[0],
            self.roc_auc(),
            self.f1_score(),
            self.matthews_corrcoef(),
            # self.sd(),
            self.mse()
        ]
