import package_.preprocessing as preprocessing
import package_.modelEvaluation as modelEvaluation

pr = preprocessing.DataPreprocessing()
pr.load_data('test_data/exampleData_TCGA_LUAD_2000.csv')
X, y = pr.set_target('class')


def test_hold_out() -> None:
    me = modelEvaluation.ModelEvaluation(X, y)
    X_train, X_test, y_train, y_test = me.hold_out(0.3)

    assert any([X_train, X_test, y_train, y_test]) is not None
