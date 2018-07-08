from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import numpy as np


def eval_and_print(model, X_test, y_test, scores='r2', folds_nr=3):
    eval_metrics = cross_validate(model, X_test, y_test,
                                  scoring=scores,
                                  return_train_score=False,
                                  cv=folds_nr)
    for score in scores:
        score_results = eval_metrics['test_' + score]
        index = 0
        print(score + ':')
        for fold in score_results:
            index += 1
            print(str(index) + ': ' + str(fold))
        print()


def render_features_importance(model, dataset):
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, dataset.columns.values[:-1])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.show()
