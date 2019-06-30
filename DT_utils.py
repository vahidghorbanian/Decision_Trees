import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_boston
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from subprocess import call
import sys

# Initialization
bins = 20
alpha = 0.8
num_sample_plot = 200

# load and analyse datasets
def load_analyze_data(data, vis_feature='petal', plot=False):
    print('feature names: ', list(data['feature_names']))
    print('target names: ', list(data['target_names']))
    data = pd.DataFrame(data=np.append(data['data'], data['target'][:, np.newaxis],axis=1),
                        columns=np.append(data['feature_names'], 'target'))
    if len(data.values[:,-1]) > len(np.unique(data['target'])):
        print('This is a classification problem.')
        print('number of classes: ', len(np.unique(data['target'])))
    else:
        print('This is a regression problem.')
    if plot == True:
        data.loc[:, data.columns != 'target'].hist(bins=bins, edgecolor=[0, 0, 0], alpha=alpha)
        f = plt.gcf()
        for i in np.arange(0, len(f.axes), 1):
            ax = f.axes[i]
            ax.legend([ax.title._text])
            ax.title._text = ''
            ax.yaxis.label._text = 'number of samples'
        data.loc[:, data.columns == 'target'].hist(bins=bins, edgecolor=[0, 0, 0], alpha=alpha)
        plt.xlabel('classes')
        plt.ylabel('number of samples')
    print('features mean: ', np.mean(data.values[:, :-1], axis=0))
    print('features variance: ', np.var(data.values[:, :-1], axis=0))
    print('features covariance: \n', np.cov(data.values[:, :-1].T))
    print('\nThe function has two outputs, i.e. original dataset as well as a filtered dataset with only two features\n'
          'useful for visualization. If so, choose either sepal or petal as input.')
    data_filtered = data
    if vis_feature == 'petal':
        data_filtered = data_filtered.drop(['sepal length (cm)', 'sepal width (cm)'], axis=1)
    else:
        data_filtered = data_filtered.drop(['petal length (cm)', 'petal width (cm)'], axis=1)
    return data, data_filtered


def decision_trees_classification(data, criterion='gini', max_depth=None, plot=True,
                                                          vis_tree=False):
    X = data.values[:, :-1]
    y = np.ravel(data.values[:, -1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = [DecisionTreeClassifier(criterion=criterion, splitter='best', max_depth=max_depth, min_samples_split=2,
                                    random_state=0)]
    score_test = []
    score_train = []
    num_class = []
    Tree = []
    for i in np.arange(0, len(model), 1):
        model[i].fit(X_train, y_train)
        score_test.append(model[i].score(X_test, y_test))
        score_train.append(model[i].score(X_train, y_train))
        num_class.append(model[i].n_classes_)
        Tree.append(model[i].tree_)
    print('\ntest score:\n', score_test)
    print('training score:\n', score_train)

    if plot == True:
        if len(data.columns) > 3:
            sys.exit('There are more than 2 features. Cannot plot classification results.')
        else:
            X = data.loc[:, data.columns != 'target'].values
            max = np.ceil(np.max(X, axis=0))
            min = np.floor(np.min(X, axis=0))
            xx, yy = np.meshgrid(np.linspace(min[0], max[0], num_sample_plot),
                                 np.linspace(min[1], max[1], num_sample_plot))
            k = 1
            fig = plt.figure(figsize=(10, 3))
            features_classes = {}
            for i in np.arange(0, len(model), 1):
                Z = model[i].predict_proba(np.c_[xx.ravel(), yy.ravel()])
                for j, cls in enumerate(np.arange(0, num_class[i], 1)):
                    features_classes[cls] = []
                    features_classes[cls] = data.loc[data['target'] == cls, data.columns != 'target'].values
                    ax = fig.add_subplot(int(len(model)), int(num_class[i]), k)
                    plt.pcolormesh(xx, yy, Z[:, j].reshape(xx.shape))
                    plt.colorbar()
                    ax.title.set_text('Criterion:'+model[i].criterion+'\nclass'+str(j+1)+', max_depth='+str(Tree[i].max_depth))
                    ax.xaxis.label.set_text(data.columns[0])
                    ax.yaxis.label.set_text(data.columns[1])
                    plt.tight_layout()
                    plt.scatter(features_classes[cls][:, 0], features_classes[cls][:, 1], s=20, marker='o',
                    linewidths=1, edgecolors=[0, 0, 0], facecolor=[0.8, 0.8, 0.8], alpha=alpha)
                    k = k + 1

    # Visualize Tree
    if vis_tree == True:
        for i in np.arange(0, len(model), 1):
            try:
                tree.export_graphviz(model[i], out_file='tree.dot', rounded=True, proportion=False,
                            precision=2, filled=True, feature_names=data.columns[:-1])
                call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
                img = mpimg.imread('tree.png')
                plt.figure()
                plt.imshow(img)
            except:
                print('An error occurred while visualizing the tree!')

    return features_classes, {'models': model, 'score_test': score_test, 'score_train': score_train, 'Tree': Tree}









