import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.pipeline import Pipeline

def chooseModel(name):
        if name == 'kNN':
            return Pipeline([('scaler', MinMaxScaler()), ('classifier', KNeighborsClassifier())]), {'classifier__n_neighbors': np.arange(1, 21, 2)}
        elif name == 'RandomForest':
            return RandomForestClassifier(random_state=0), {'n_estimators': [200, 300, 500, 1000], 'max_depth': [3, 5, 7, 9]}
        elif name == 'DecisionTree':
            return DecisionTreeClassifier(random_state=0), {'max_features': ["sqrt", "log2"], 'criterion': ["gini", "entropy", "log_loss"]} 
        elif name == 'LogisticRegression':
            return Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=0, solver="saga", penalty="elasticnet"))]), {'classifier__l1_ratio': np.linspace(0, 1, 11)}
        elif name == 'NeuralNetworkMLP':
            return Pipeline([('scaler', StandardScaler()), ('classifier', MLPClassifier(random_state=0, max_iter=300))]), {'classifier__hidden_layer_sizes': [(100, 20), (50, 20), (20, 20, 10)]}
        elif name == 'Gradient Boosting':
            return HistGradientBoostingClassifier(random_state=0), {'min_samples_leaf': [4, 7, 10, 13], 'max_iter': [75, 100, 125]}
        else:
            raise ValueError("Invalid ML method. Please choose one of ['kNN', 'RandomForest', 'DecisionTree', 'LogisticRegression', 'NeuralNetworkMLP', 'Gradient Boosting']")