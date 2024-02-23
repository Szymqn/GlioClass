import pandas as pd
import package_.modelEvaluation as modelEvaluation
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Classifier:
    def __init__(self, X: pd.DataFrame = None, y: pd.Series = None, features: pd.Series = None, classifiers: list = None,
                 cross_validation: str = 'hold_out', fold: int = 1):
        self.X = X[features] if features else X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.cross_validation = cross_validation
        self.classifiers = classifiers
        self.predictions = {}
        self.fold = fold

        me = modelEvaluation.ModelEvaluation(self.X, self.y)

        match self.cross_validation:
            case 'hold_out':
                self.X_train, self.X_test, self.y_train, self.y_test = me.hold_out(0.3)
            case 'k_fold':
                self.X_train, self.X_test, self.y_train, self.y_test = me.k_fold(self.fold)
            case 'stratified_k_fold':
                self.X_train, self.X_test, self.y_train, self.y_test = me.stratified_k_fold(self.fold)
            case 'leave_one_out':
                self.X_train, self.X_test, self.y_train, self.y_test = me.leave_one_out()
            case 'resample':
                self.X_train, self.X_test, self.y_train, self.y_test = me.resample(self.fold)
            case _:
                raise ValueError('Invalid cross_validation')

        for classifier in self.classifiers:
            match classifier:
                case 'adaboost':
                    self.predictions['adaboost'] = self.ada_boost()
                case 'gradient_boosting':
                    self.predictions['gradient_boosting'] = self.gradient_boosting()
                case 'random_forest':
                    self.predictions['random_forest'] = self.random_forest()
                case 'k_neighbors':
                    self.predictions['k_neighbors'] = self.k_neighbors()
                case 'decision_tree':
                    self.predictions['decision_tree'] = self.decision_tree()
                case 'extra_trees':
                    self.predictions['extra_trees'] = self.extra_trees()
                case 'svm':
                    self.predictions['svm'] = self.svm()
                case 'xgb':
                    self.predictions['xgb'] = self.xgb()
                case 'all':
                    self.predictions['adaboost'] = self.ada_boost()
                    self.predictions['gradient_boosting'] = self.gradient_boosting()
                    self.predictions['random_forest'] = self.random_forest()
                    self.predictions['k_neighbors'] = self.k_neighbors()
                    self.predictions['decision_tree'] = self.decision_tree()
                    self.predictions['extra_trees'] = self.extra_trees()
                    self.predictions['svm'] = self.svm()
                    self.predictions['xgb'] = self.xgb()
                case _:
                    raise ValueError('Invalid classifier name')

    def ada_boost(self):
        predict_proba = []
        for fold in range(self.fold):
            adaboostClf = AdaBoostClassifier(random_state=42)
            adaboostClf_f = adaboostClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(adaboostClf_f.predict(self.X_test[fold]))

        return predict_proba

    def gradient_boosting(self):
        predict_proba = []
        for fold in range(self.fold):
            gboostClf = GradientBoostingClassifier(random_state=42)
            gboostClf_f = gboostClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(gboostClf_f.predict(self.X_test[fold]))

        return predict_proba

    def random_forest(self):
        predict_proba = []
        for fold in range(self.fold):
            randomForestClf = RandomForestClassifier(random_state=42)
            randomForestClf_f = randomForestClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(randomForestClf_f.predict(self.X_test[fold]))

        return predict_proba

    def k_neighbors(self):
        predict_proba = []
        for fold in range(self.fold):
            kneighborsClf = KNeighborsClassifier()
            kneighborsClf_f = kneighborsClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(kneighborsClf_f.predict(self.X_test[fold]))

        return predict_proba

    def decision_tree(self):
        predict_proba = []
        for fold in range(self.fold):
            dtreeClf = DecisionTreeClassifier(random_state=42)
            dtreeClf_f = dtreeClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(dtreeClf_f.predict(self.X_test[fold]))

        return predict_proba

    def extra_trees(self):
        predict_proba = []
        for fold in range(self.fold):
            extraTreeClf = ExtraTreesClassifier(random_state=42)
            extraTreeClf_f = extraTreeClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(extraTreeClf_f.predict(self.X_test[fold]))

        return predict_proba

    def svm(self):
        predict_proba = []
        for fold in range(self.fold):
            svmClf = SVC(probability=True, gamma='auto')
            svmClf_f = svmClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(svmClf_f.predict(self.X_test[fold]))

        return predict_proba

    def xgb(self):
        predict_proba = []
        for fold in range(self.fold):
            xgbClf = XGBClassifier()
            xgbClf_f = xgbClf.fit(self.X_train[fold], self.y_train[fold])
            predict_proba.append(xgbClf_f.predict(self.X_test[fold]))

        return predict_proba
