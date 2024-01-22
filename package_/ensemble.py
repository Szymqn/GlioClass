import pandas as pd
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier


class Ensemble:
    def __init__(self, X_train: pd.DataFrame, X_test: pd.Series, y_train: pd.DataFrame, y_test: pd.Series):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def Voting(self, classifiers: list, voting: str):
        if voting not in ('soft', 'hard'):
            raise Exception('Voting should be soft or hard')
        Voting = VotingClassifier(estimators=classifiers, voting=voting)
        Voting.fit(self.X_train, self.y_train)
        return Voting.predict(self.X_test)

    def Bagging(self, classifiers: list):
        bagging = BaggingClassifier(estimator=classifiers[0][1])
        bagging.fit(self.X_train, self.y_train)
        return bagging.predict(self.X_test)

    def Stacking(self, classifiers: list):
        stacking = StackingClassifier(estimators=classifiers)
        stacking.fit(self.X_train, self.y_train)
        return stacking.predict(self.X_test)

    # def run_ensemble(method, classifiers, X_train, X_test, y_train, y_test):
    #     ense = ensemble.Ensemble(X_train, X_test, y_train, y_test)
    #
    #     if method == 'voting':
    #         prediction = ense.Voting(classifiers, 'soft')
    #     elif method == 'bagging':
    #         prediction = ense.Bagging(classifiers)
    #     elif method == 'stacking':
    #         prediction = ense.Stacking(classifiers)
    #     else:
    #         raise ValueError(f"Invalid ensemble method: {method}")
    #
    #     pm = performanceMetrics.PerformanceMetrics(y_test, prediction)
    #     accuracy = pm.accuracyScore()[1]
    #
    #     return accuracy

    # def evaluate(X: pd.DataFrame, y: pd.Series, features_name: str, features: list):
    #     print(f'{features_name}:')
    #     voting_score, bagging_score, stacking_score = None, None, None
    #     me = modelEvaluation.ModelEvaluation(X[features], y)
    #     n_splits = 3
    #     X_train_list, X_test_list, y_train_list, y_test_list = me.StratifiedKFold(n_splits)
    #
    #     for method in ['voting', 'bagging', 'stacking']:
    #         scores = []
    #
    #         for i in range(n_splits):
    #             clf = classifier.Classifier(X_train_list[i], X_test_list[i], y_train_list[i], y_test_list[i])
    #             classifiers = [
    #                 ('rf', clf.RandomForest()[0]),
    #                 ('svm', clf.SVM()[0]),
    #                 ('knn', clf.KNeighbors()[0]),
    #             ]
    #
    #             accuracy = run_ensemble(method, classifiers, X_train_list[i], X_test_list[i], y_train_list[i],
    #                                     y_test_list[i])
    #             scores.append(accuracy)
    #
    #         print(f"Acc list ({method}):", scores)
    #         avg_score = np.mean(scores).round(decimals=2)
    #         print(f"Average acc ({method}):", avg_score)
    #
    #         if method == 'voting':
    #             voting_score = avg_score
    #         elif method == 'bagging':
    #             bagging_score = avg_score
    #         elif method == 'stacking':
    #             stacking_score = avg_score
    #
    #     return voting_score, bagging_score, stacking_score
