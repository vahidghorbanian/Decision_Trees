from DT_utils import *


# Classification
data = load_iris()
data, data_filtered, data_statistics = load_analyze_data_iris(data, vis_feature='sepal', plot=True)
results_clf = decision_trees_classification(data_filtered, criterion='gini', max_depth=None, plot=True,
                                                          vis_tree=False)

# Regression
data = load_boston()
data, data_statistics = load_analyze_data_boston(data, plot=True)
results_reg = decision_trees_regression(data, criterion='mse', max_depth=None, n_estimator=100, plot=True)

# Show results
plt.show()