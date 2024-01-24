import pandas as pd
import package_.modelEvaluation as modelEvaluation
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Ensemble:
    def __init__(self, X: pd.DataFrame = None, y: pd.Series = None, features: pd.Series = None, ensemble: str = None, classifiers: list = None,
                 cross_validation: str = 'hold_out', fold: int = 1, **kwargs):
        self.X = X[features] if features else X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.ensemble = ensemble
        self.cross_validation = cross_validation
        self.classifiers = classifiers
        self.model_classifiers = []
        self.predictions = []
        self.fold = fold

        me = modelEvaluation.ModelEvaluation(self.X, self.y)

        match self.cross_validation:
            case 'hold_out':
                self.X_train, self.X_test, self.y_train, self.y_test = me.holdOut(0.3)
            case 'k_fold':
                self.X_train, self.X_test, self.y_train, self.y_test = me.kFold(self.fold)
            case 'stratified_k_fold':
                self.X_train, self.X_test, self.y_train, self.y_test = me.StratifiedKFold(self.fold)
            case 'leave_one_out':
                self.X_train, self.X_test, self.y_train, self.y_test = me.leaveOneOut()
            case _:
                raise ValueError('Invalid cross_validation')

        for classifier in self.classifiers:
            match classifier:
                case 'adaboost':
                    self.model_classifiers.append(('adaboost', AdaBoostClassifier()))
                case 'gradient_boosting':
                    self.model_classifiers.append(('gradient_boosting', GradientBoostingClassifier()))
                case 'random_forest':
                    self.model_classifiers.append(('random_forest', RandomForestClassifier()))
                case 'k_neighbors':
                    self.model_classifiers.append(('k_neighbors', KNeighborsClassifier()))
                case 'decision_tree':
                    self.model_classifiers.append(('decision_tree', DecisionTreeClassifier()))
                case 'extra_trees':
                    self.model_classifiers.append(('extra_trees', ExtraTreesClassifier()))
                case 'svm':
                    self.model_classifiers.append(('svm', SVC()))
                case 'xgb':
                    self.model_classifiers.append(('xgb', XGBClassifier()))
                case 'all':
                    self.model_classifiers = [('adaboost', AdaBoostClassifier()),
                                              ('gradient_boosting', GradientBoostingClassifier()),
                                              ('random_forest', RandomForestClassifier()),
                                              ('k_neighbors', KNeighborsClassifier()),
                                              ('decision_tree', DecisionTreeClassifier()),
                                              ('extra_trees', ExtraTreesClassifier()),
                                              ('svm', SVC()),
                                              ('xgb', XGBClassifier())]
                case _:
                    raise ValueError('Invalid classifier name')

        match self.ensemble:
            case 'Voting':
                self.Voting(**kwargs)
            case 'Bagging':
                self.Bagging()
            case 'Stacking':
                self.Stacking()

    def Voting(self, **kwargs):
        voting = kwargs.get('voting', 'soft')
        if voting not in ('soft', 'hard'):
            raise Exception('Voting should be soft or hard')
        for fold in range(self.fold):
            Voting = VotingClassifier(estimators=self.model_classifiers, voting=voting)
            Voting.fit(self.X_train[fold], self.y_train[fold])
            self.predictions.append(Voting.predict(self.X_test[fold]))

    def Bagging(self):
        for fold in range(self.fold):
            bagging = BaggingClassifier(estimator=self.model_classifiers[0][1])
            bagging.fit(self.X_train[fold], self.y_train[fold])
            self.predictions.append(bagging.predict(self.X_test[fold]))

    def Stacking(self):
        for fold in range(self.fold):
            stacking = StackingClassifier(estimators=self.model_classifiers)
            stacking.fit(self.X_train[fold], self.y_train[fold])
            self.predictions.append(stacking.predict(self.X_test[fold]))
