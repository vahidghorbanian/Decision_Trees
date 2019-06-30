from DT_utils import *


data = load_iris()
data, data_filtered = load_analyze_data(data, vis_feature='sepal', plot=False)
features_classes, results = decision_trees_classification(data_filtered, criterion='gini', max_depth=None, plot=True,
                                                          vis_tree=False)
plt.show()