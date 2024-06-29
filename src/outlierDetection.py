import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def RNAseqQA(expression_matrix, methods):
    if not methods:
        return expression_matrix.T.iloc[np.array([])].index, np.array([])
    else:
        print("Performing samples quality analysis...\n")

        unique_rows = np.unique(expression_matrix, axis=0)
        expression_matrix = pd.DataFrame(unique_rows, columns=expression_matrix.columns)

        res_outliers = pd.DataFrame()

        def outlier_bar_plot(data, title, limit, xlab):
            yticks = data['y'].tolist()
            outlier_bar_plot = px.scatter(data, x='x', y='y', labels={'x': xlab, 'y': 'Samples'},
                                        title=title, template='plotly', range_x=[0, limit])
            outlier_bar_plot.add_shape(
                go.layout.Shape(
                    type="line",
                    x0=limit,
                    x1=limit,
                    y0=data['y'].min(),
                    y1=data['y'].max(),
                    line=dict(color="red", width=2),
                )
            )
            outlier_bar_plot.update_yaxes(tickvals=yticks[::3], ticktext=yticks[::3])
            return outlier_bar_plot

        num_samples = expression_matrix.shape[1]

        if "Distances" in methods:
            print("Running Distances Outliers Detection test...")
            expression_matrix_transposed = np.transpose(expression_matrix)
            distance_matrix = cdist(expression_matrix_transposed, expression_matrix_transposed, metric='cityblock')
            distance_matrix /= expression_matrix.shape[0]
            distance_sum = np.sum(distance_matrix, axis=0)
            dist_data = pd.DataFrame({'x': distance_sum, 'y': list(expression_matrix.columns)})
            
            q3 = dist_data['x'].quantile(0.75)
            iqr = dist_data['x'].quantile(0.75) - dist_data['x'].quantile(0.25)
            
            dist_limit = q3 + 1.5 * iqr
            outliers= dist_data["y"][dist_data["x"] > dist_limit]
            res_outliers["Distances"] = [-1 if s in outliers.index else 1 for s in np.arange(expression_matrix.shape[1])]

            print("Done!\n")

        if "IForest" in methods:
            print("Running Isolation Forest test...")
            clf = IsolationForest(random_state=0)
            outs = clf.fit_predict(expression_matrix.T)
            res_outliers["IForest"] = outs.copy()

            print("Done!\n")
            
        if "LOF" in methods:
            print("Running Local Outlier Factor test...")
            clf = LocalOutlierFactor(n_neighbors=20)
            outs = clf.fit_predict(expression_matrix.T)
            res_outliers["LOF"] = outs.copy()
        
            print("Done!\n")
        if "MAD" in methods:
            print("Running MAD Outliers Detection test...")
            outliersMA = []
            
            rowExpression = np.mean(expression_matrix, axis=0)
            
            for i in range(len(rowExpression)):
                exprMatrix = np.delete(rowExpression, i)
                upperBound = np.median(exprMatrix) + 3 * np.median(np.abs(exprMatrix - np.median(exprMatrix)))
                lowerBound = np.median(exprMatrix) - 3 * np.median(np.abs(exprMatrix - np.median(exprMatrix)))
                
                outlier = (rowExpression[i] < lowerBound) | (rowExpression[i] > upperBound)
                
                outliersMA.append(outlier)
            res_outliers["MAD"] = [-1 if s else 1 for s in outliersMA]
        
            print("Done!\n")

        majority = np.floor(res_outliers.shape[1]/2 + 1)
        top = np.sum(res_outliers == -1, axis=1)
        indices_mayor_igual_majority = np.where(top >= majority)[0]

        return expression_matrix.T.iloc[indices_mayor_igual_majority].index, indices_mayor_igual_majority