import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_boston
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from subprocess import call
import sys

# Initialization
bins = 50
alpha = 1
num_sample_plot = 200

# load and analyse iris dataset (Classification)
def load_analyze_data_iris(data, vis_feature='petal', plot=False):
    print('\nClassification Problem: iris Data set')
    print('feature names: ', list(data['feature_names']))
    print('target names: ', list(data['target_names']))
    data = pd.DataFrame(data=np.append(data['data'], data['target'][:, np.newaxis], axis=1),
                        columns=np.append(data['feature_names'], 'target'))
    if plot == True:
        data.loc[:, data.columns != 'target'].hist(bins=bins, edgecolor=[0, 0, 0], alpha=alpha)
        plt.tight_layout()
        f = plt.gcf()
        for i in np.arange(0, len(f.axes), 1):
            ax = f.axes[i]
            ax.legend([ax.title._text])
            ax.title._text = ''
            ax.yaxis.label._text = 'number of samples'
            ax.yaxis.label._fontproperties._size = 6
        plt.suptitle('Classification problem: iris data set')
        data.loc[:, data.columns == 'target'].hist(bins=bins, edgecolor=[0, 0, 0], alpha=alpha)
        plt.suptitle('Classification problem: iris data set')
        plt.xlabel('classes')
        plt.ylabel('number of samples')
    print('features mean: ', np.mean(data.values[:, :-1], axis=0))
    print('features variance: ', np.var(data.values[:, :-1], axis=0))
    # print('features covariance: \n', np.cov(data.values[:, :-1].T))
    print('\nThe function has two outputs, i.e. original dataset as well as a filtered dataset with only two features\n'
          'useful for visualization. If so, choose either sepal or petal as input.')
    data_filtered = data
    if vis_feature == 'petal':
        data_filtered = data_filtered.drop(['sepal length (cm)', 'sepal width (cm)'], axis=1)
    else:
        data_filtered = data_filtered.drop(['petal length (cm)', 'petal width (cm)'], axis=1)
    return data, data_filtered, {'features_mean': np.mean(data.values[:, :-1], axis=0),
                  'features_variance': np.var(data.values[:, :-1], axis=0),
                  'features_covariance': np.cov(data.values[:, :-1].T)}


# Load and analyze boston dataset (Regression)
def load_analyze_data_boston(data, plot=False):
    print('\nRegression Problem: Boston Data set')
    print('feature names: ', list(data['feature_names']))
    data = pd.DataFrame(data=np.append(data['data'], data['target'][:, np.newaxis], axis=1),
                        columns=np.append(data['feature_names'], 'target'))
    if plot == True:
        data.loc[:, data.columns != 'target'].hist(bins=bins, edgecolor=[0, 0, 0], alpha=alpha)
        plt.tight_layout()
        f = plt.gcf()
        for i in np.arange(0, len(f.axes), 1):
            ax = f.axes[i]
            ax.legend([ax.title._text], fontsize=6)
            ax.title._text = ''
            ax.yaxis.label._text = 'number of samples'
            ax.yaxis.label._fontproperties._size = 8
        plt.suptitle('Regression problem: boston dataset')
        data.loc[:, data.columns == 'target'].hist(bins=bins, edgecolor=[0, 0, 0], alpha=alpha)
        plt.suptitle('Regression problem: boston dataset')
        plt.xlabel('classes')
        plt.ylabel('number of samples')
    print('features mean: ', np.mean(data.values[:, :-1], axis=0))
    print('features variance: ', np.var(data.values[:, :-1], axis=0))
    # print('features covariance: \n', np.cov(data.values[:, :-1].T))
    return data, {'features_mean': np.mean(data.values[:, :-1], axis=0),
                  'features_variance': np.var(data.values[:, :-1], axis=0),
                  'features_covariance': np.cov(data.values[:, :-1].T)}


# Classification problem
def decision_trees_classification(data, criterion='gini', max_depth=None, plot=True,
                                                          vis_tree=False):
    X = data.values[:, :-1]
    y = np.ravel(data.values[:, -1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = [DecisionTreeClassifier(criterion=criterion, splitter='best', max_depth=max_depth, min_samples_split=2,
                                    random_state=0, presort=False)]
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
                    ax.title.set_text('Criterion:'+model[i].criterion+'\nclass '+str(j+1)+', depth='+str(Tree[i].max_depth))
                    ax.title._fontproperties._size = 10
                    ax.xaxis.label.set_text(data.columns[0])
                    ax.yaxis.label.set_text(data.columns[1])
                    plt.tight_layout()
                    plt.scatter(features_classes[cls][:, 0], features_classes[cls][:, 1], s=20, marker='o',
                    linewidths=1, edgecolors=[0, 0, 0], facecolor=[0.8, 0.8, 0.8], alpha=alpha)
                    k = k + 1
            plt.suptitle('Classification problem: iris data set')
            plt.tight_layout()

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

    return {'models': model, 'score_test': score_test, 'score_train': score_train, 'Tree': Tree}


# Regression problem
def decision_trees_regression(data, criterion='mse', max_depth=None, n_estimator=100, plot=True):
    X = data.values[:, :-1]
    y = np.ravel(data.values[:, -1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = {'model': [DecisionTreeRegressor(criterion=criterion, splitter='best', max_depth=max_depth,
                                             min_samples_split=2,random_state=0, presort=False),
             RandomForestRegressor(n_estimators=n_estimator, criterion=criterion, random_state=0, max_depth=max_depth,
                                   min_samples_split=2)], 'model_name': list(['Tree',
                                                                              'Forest(n_estimator='+str(n_estimator)+')'])}
    score_test  = []
    score_train = []
    Tree        = []
    feature_importance = []

    for i in np.arange(0, len(model['model']), 1):
        model['model'][i].fit(X_train, y_train)
        score_test.append(model['model'][i].score(X_test, y_test))
        score_train.append(model['model'][i].score(X_train, y_train))
        feature_importance.append(model['model'][i].feature_importances_)
        # Tree.append(model[i]['model'].tree_)
    print('\ntest score:\n', score_test)
    print('training score:\n', score_train)

    if plot == True:
        k = 1
        fig = plt.figure(figsize=(10, 3*len(model)))
        for i in np.arange(0, len(model), 1):
            y_test_predict  = model['model'][i].predict(X_test)
            y_train_predict = model['model'][i].predict(X_train)
            fig.add_subplot(int(len(model['model'])), 2, k)
            plt.scatter(y_train,y_train_predict, s=20, marker='o',
                    linewidths=1, edgecolors=[0, 0, 0], facecolor=[0.8, 0.8, 0], alpha=alpha)
            plt.xlabel('target')
            plt.ylabel('Prediction')
            plt.title(model['model_name'][i]+': Training set')
            plt.tight_layout()
            fig.add_subplot(int(len(model['model'])), 2, k+1)
            plt.scatter(y_test, y_test_predict, s=20, marker='o',
                    linewidths=1, edgecolors=[0, 0, 0], facecolor=[0.8, 0.8, 0], alpha=alpha)
            plt.xlabel('target')
            plt.ylabel('Prediction')
            plt.title(model['model_name'][i]+': test set')
            plt.tight_layout()
            k = k + 2
        # plt.suptitle('Regression problem: boston dataset\n'
        #              'criterion: '+criterion+' Depth: '+str(Tree[i].max_depth))

    return {'models': model, 'score_test': score_test, 'score_train': score_train,
            'feature_importance': feature_importance, 'Tree': Tree}





