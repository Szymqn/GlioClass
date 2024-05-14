import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, mean_squared_error

from package_.classifier import Classifier


class PerformanceMetrics:
    def __init__(self, classifier: Classifier):
        self.y_test = list(classifier.y_test)
        self.y_pred = classifier.predictions
        self.classifiers = (self.y_pred.keys())
        self.time = classifier.time
        self.fold = classifier.fold
        self.fs = classifier.fs

        self.check_for_none()

    def check_for_none(self):
        if any(var is None for var in [self.y_test, self.y_pred, self.time, self.fold, self.fs]):
            raise ValueError("One or more classifier variables are invalid.")

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
        mse_dict = {}

        for classifier in self.classifiers:
            acc = []
            mse = []
            if self.fold != 1:
                for f in range(self.fold):
                    acc.append(accuracy_score(self.y_test[f], self.y_pred[classifier][f]))
                    mse.append(mean_squared_error(self.y_test[f], self.y_pred[classifier][f]))
                acc_dict[classifier] = acc
                mse_dict[classifier] = mse
            else:
                acc_dict[classifier] = accuracy_score(self.y_test, self.y_pred)
                mse_dict[classifier] = mean_squared_error(self.y_test, self.y_pred)

        mean_dict = {classifier: round(np.mean(values), 3) for classifier, values in acc_dict.items()}
        mse_dict = {classifier: round(np.mean(values), 3) for classifier, values in mse_dict.items()}
        combined_dict = {classifier: [mean_dict[classifier], mse_dict.get(classifier)] for classifier in mean_dict}

        print("Accuracy:\n", combined_dict)

        return "ACC: " + str(mean_dict), acc_dict

    def roc_auc(self):
        roc_auc_dict = {}
        mse_dict = {}

        for classifier in self.classifiers:
            acc = []
            mse = []
            if self.fold != 1:
                for f in range(self.fold):
                    acc.append(roc_auc_score(self.y_test[f], self.y_pred[classifier][f]))
                    mse.append(mean_squared_error(self.y_test[f], self.y_pred[classifier][f]))
                roc_auc_dict[classifier] = acc
                mse_dict[classifier] = mse
            else:
                roc_auc_dict[classifier] = roc_auc_score(self.y_test, self.y_pred)
                mse_dict[classifier] = mean_squared_error(self.y_test, self.y_pred)

        mean_dict = {classifier: round(np.mean(values), 3) for classifier, values in roc_auc_dict.items()}
        mse_dict = {classifier: round(np.mean(values), 3) for classifier, values in mse_dict.items()}
        combined_dict = {classifier: [mean_dict[classifier], mse_dict.get(classifier)] for classifier in mean_dict}
        print("Roc Auc:\n", combined_dict)

        return "Roc Auc: " + str(mean_dict)

    def f1_score(self):
        f1_score_dict = {}
        mse_dict = {}

        for classifier in self.classifiers:
            acc = []
            mse = []
            if self.fold != 1:
                for f in range(self.fold):
                    acc.append(f1_score(self.y_test[f], self.y_pred[classifier][f]))
                    mse.append(mean_squared_error(self.y_test[f], self.y_pred[classifier][f]))
                f1_score_dict[classifier] = acc
                mse_dict[classifier] = mse
            else:
                f1_score_dict[classifier] = f1_score(self.y_test, self.y_pred)
                mse_dict[classifier] = mean_squared_error(self.y_test, self.y_pred)

        mean_dict = {classifier: round(np.mean(values), 3) for classifier, values in f1_score_dict.items()}
        mse_dict = {classifier: round(np.mean(values), 3) for classifier, values in mse_dict.items()}
        combined_dict = {classifier: [mean_dict[classifier], mse_dict.get(classifier)] for classifier in mean_dict}
        print("F1 score:\n", combined_dict)

        return "F1 score: " + str(mean_dict)

    def matthews_corrcoef(self):
        matthews_corrcoef_dict = {}
        mse_dict = {}

        for classifier in self.classifiers:
            acc = []
            mse = []
            if self.fold != 1:
                for f in range(self.fold):
                    acc.append(matthews_corrcoef(self.y_test[f], self.y_pred[classifier][f]))
                    mse.append(mean_squared_error(self.y_test[f], self.y_pred[classifier][f]))
                matthews_corrcoef_dict[classifier] = acc
                mse_dict[classifier] = mse
            else:
                matthews_corrcoef_dict[classifier] = matthews_corrcoef(self.y_test, self.y_pred)
                mse_dict[classifier] = mean_squared_error(self.y_test, self.y_pred)

        mean_dict = {classifier: round(np.mean(values), 3) for classifier, values in matthews_corrcoef_dict.items()}
        mse_dict = {classifier: round(np.mean(values), 3) for classifier, values in mse_dict.items()}
        combined_dict = {classifier: [mean_dict[classifier], mse_dict.get(classifier)] for classifier in mean_dict}
        print("MCC:\n", combined_dict)

        return "MCC: " + str(mean_dict)

    def plot_classifier_acc(self):
        scores_dict = self.accuracy_score()[1]

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=list(scores_dict.values()), palette='hls')
        plt.xticks(ticks=range(len(scores_dict)), labels=list(scores_dict.keys()), rotation=90)
        plt.ylabel('Accuracy score')
        plt.title(f'Box plot of classifiers accuracy, FS: {self.fs}')
        plt.grid(True)

        plt.show()

    def plot_classifier_time(self):
        sorted_results = sorted(zip(self.time.keys(), self.time.values()), key=lambda x: x[1], reverse=False)

        methods, times = zip(*sorted_results)
        max_time = max(times)
        time_stamp = max_time / 10

        plt.bar(methods, times)
        plt.ylim(0.01, max_time + time_stamp)
        plt.yticks(np.arange(0.01, max_time + time_stamp, time_stamp))

        plt.xlabel('Classifiers')
        plt.ylabel('Time in seconds')
        plt.title(f'Classifiers Time Measure - {self.fs}')

        for i, (method, time) in enumerate(zip(methods, times)):
            plt.text(i, time + (time_stamp * 0.1), f'{round(time, 3)} s', ha='center', va='bottom')

        plt.xticks(rotation=90)

        plt.show()

        for method, time in zip(methods, times):
            print(f"{method}: {round(time, 3)} s.")

    def all_metrics(self):
        return [
            self.accuracy_score()[0],
            self.roc_auc(),
            self.f1_score(),
            self.matthews_corrcoef()
        ]
