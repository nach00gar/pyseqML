import pandas as pd
import numpy as np
from mrmr import mrmr_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from skrebate import ReliefF, MultiSURFstar, SURFstar, MultiSURF
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

def featureSelectmrmr(df, target, n_selected=None):
    if n_selected == None:
        n_selected = df.shape[1] // 2
    
    selected_features = mrmr_classif(X=df, y=target.to_numpy(), K=n_selected, return_scores=True)
    sorted_features = selected_features[1].sort_values(ascending=False)
    sorted_features = pd.DataFrame(sorted_features)
    sorted_features.columns= ["Importance"]
    return sorted_features

def featureSelectRandomForest(df, target, n_selected=None):
    sel = RandomForestClassifier(n_estimators=300, random_state=42)
    sel.fit(df, target)
    feature_importances = sel.feature_importances_
    df_feature_importances = pd.DataFrame({'Importance': feature_importances}, index=df.columns)
    sorted_features = df_feature_importances.sort_values(by='Importance', ascending=False)
    return sorted_features

def featureSelectSVM(df, target, n_selected=None):
    rf = SVC(kernel="linear", C=3)
    rf.fit(df, target)
    df_feature_importances = pd.DataFrame({'Importance': np.sum(abs(rf.coef_), axis=0)}, index=df.columns)
    sorted_features = df_feature_importances.sort_values(by='Importance', ascending=False)
    return sorted_features

def featureSelectChi2(df, target, n_selected=None):
    chi2_filter = SelectKBest(chi2, k="all")
    chi2_filter.fit(df, target)
    feature_importances = chi2_filter.scores_
    df_feature_importances = pd.DataFrame({'Importance': feature_importances}, index=df.columns)
    sorted_features = df_feature_importances.sort_values(by='Importance', ascending=False)
    return sorted_features

def featureSelectReliefF(df, target, n_selected=None):
    sel = ReliefF(n_jobs=-1)
    sel.fit(df.to_numpy(), LabelEncoder().fit_transform(target))
    feature_importances = sel.feature_importances_
    df_feature_importances = pd.DataFrame({'Importance': feature_importances}, index=df.columns)
    sorted_features = df_feature_importances.sort_values(by='Importance', ascending=False)
    return sorted_features

def featureSelectMultisurf(df, target, n_selected=None):
    sel = MultiSURF(n_jobs=-1)
    sel.fit(df.to_numpy(), LabelEncoder().fit_transform(target))
    feature_importances = sel.feature_importances_
    df_feature_importances = pd.DataFrame({'Importance': feature_importances}, index=df.columns)
    sorted_features = df_feature_importances.sort_values(by='Importance', ascending=False)
    return sorted_features


def featureSelect(df, target, n_selected, method):
    if method == 'mRMR':
        return featureSelectmrmr(df, target, n_selected)
    elif method == 'RandomForest':
        return featureSelectRandomForest(df, target, n_selected)
    elif method == 'SVM':
        return featureSelectSVM(df, target, n_selected)
    elif method == 'Chi2':
        return featureSelectChi2(df, target, n_selected)
    elif method == 'Relief':
        return featureSelectReliefF(df, target, n_selected)
    elif method == 'MultiSurf':
        return featureSelectMultisurf(df, target, n_selected)
    else:
        raise ValueError("Invalid FS method. Please choose one of ['mRMR', 'RandomForest', 'SVM', 'Chi2', 'Relief', 'MultiSurf', 'MIFS']")