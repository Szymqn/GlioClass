import package_.preprocessing as preprocessing
import package_.modelEvaluation as modelEvaluation
from math import floor, ceil

pr = preprocessing.DataPreprocessing()
pr.load_data('../../test_data/exampleData_TCGA_LUAD_2000.csv')
X, y = pr.set_target('class')
me = modelEvaluation.ModelEvaluation(X, y)


def test_hold_out() -> None:
    test_size = 0.3

    X_train, X_test, y_train, y_test = me.hold_out(test_size)

    assert any(var is None for var in [X_train, X_test, y_train, y_test]) is False
    assert X_train.shape[0] == floor(X.shape[0] * (1 - test_size)) & \
           X_test.shape[0] == ceil(X.shape[0] * test_size) & \
           y_train.shape[0] == floor(y.shape[0] * (1 - test_size)) & \
           y_test.shape[0] == ceil(y.shape[0] * test_size)


def test_k_fold() -> None:
    n_splits = 3

    X_train, X_test, y_train, y_test = me.k_fold(n_splits)
    assert any(var is None for var in [X_train, X_test, y_train, y_test]) is False
    assert len(X_train) == n_splits & \
           len(X_test) == n_splits & \
           len(y_train) == n_splits & \
           len(y_test) == n_splits


def test_stratified_k_fold() -> None:
    n_splits = 3

    X_train, X_test, y_train, y_test = me.stratified_k_fold(n_splits)
    assert any(var is None for var in [X_train, X_test, y_train, y_test]) is False
    assert len(X_train) == n_splits & \
           len(X_test) == n_splits & \
           len(y_train) == n_splits & \
           len(y_test) == n_splits


def test_leave_one_out() -> None:
    X_train, X_test, y_train, y_test = me.leave_one_out()

    assert any(var is None for var in [X_train, X_test, y_train, y_test]) is False
    assert len(X_train) == X.shape[0] & \
           len(X_test) == X.shape[0] & \
           len(y_train) == y.shape[0] & \
           len(y_test) == y.shape[0]
