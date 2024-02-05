import dash
from dash import dcc
from dash import html
from datetime import date
import plotly.express as px
import plotly.io as pio
import numpy as np
import pandas as pd
import time
import copy
import rpy2.robjects.numpy2ri
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from itertools import combinations

pio.templates.default = "plotly_dark"


app = dash.Dash(__name__)
server = app.server

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#0c0d0d',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}


# Definir la estructura de la aplicación
app.layout = html.Div([
    html.H1(children='pyseqML'),
    dcc.Tabs([
        dcc.Tab(label='Carga de datos y cálculo de expresión', style=tab_style, selected_style=tab_selected_style, children=[
        html.P(children='En esta sección puedes cargar los datos de 2 formas distintas. Puedes cargar directamente los counts, para ello debes seleccionar el fichero data_info y debes colocar la carpeta que contiene los counts en el directorio data del programa. También puedes cargar directamente el fichero con la matriz de expresión procedente de R.'),
        dcc.RadioItems(options = ['Carga de Counts', 'Carga de matriz de expresión'], value='Carga de Counts', id='loadMode', className='dash-radioitems'),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Selecciona el fichero data_info con los datos de los counts',
                html.A('Select Files')
        ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
        dcc.Dropdown(
            id="dropdown",
            style={
                'visibility': 'hidden' 
            },
            placeholder="Selecciona la carpeta con los counts",
            options=[{"label": x, "value": x} for x in obtener_subdirectorios("data")]
        ),
        dcc.Upload(
            id='upload-data-labels',
            children=html.Div([
                'Selecciona el fichero con los labels procedente de R'
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'visibility': 'hidden' 
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
        dcc.Dropdown(
            id="filename",
            style={
                'visibility': 'hidden' 
            },
            placeholder="Selecciona la columna con los nombres de los ficheros",
        ),
        html.Button('CARGAR LOS DATOS', id='showMatrix', n_clicks=None, className='buttons'),
        html.Div([], id='output-data-upload'),
        ]),
        dcc.Tab(label='Análisis de calidad y Eliminación de Efecto Batch', style=tab_style, selected_style=tab_selected_style, children=[
            html.P(children='Selecciona los test de detección de outliers que quieres realizar, se efectuara una votación de outlier para cada muestra, eliminándose aquellas seleccionadas por la mayoría de los métodos. A continuación se eliminará el efecto batch con SVA.'),
            dcc.Checklist(options =[{'label': 'Distances Outliers Detection', 'value': 'Distances'}, {'label': 'MAD Outliers Detection', 'value': 'MAD'}, {'label': 'Local Outlier Factor', 'value': 'LOF'}, {'label': 'Isolation Forest', 'value': 'IForest'}], id='outliers', className='dash-radioitems', inline=True),
            html.Button('EJECUTA EL ANÁLISIS Y LA ELIMINACIÓN DEL EFECTO BATCH', id='QA', n_clicks=None, className='buttons'),
            html.Div([], id='output-qa'),
        ]),
        dcc.Tab(label='Análisis de Expresión Diferencial', style=tab_style, selected_style=tab_selected_style, children=[
            html.P(children='Selecciona los parámetros para la extracción de Genes Diferencialmente Expresados:'),
            html.Label('P-Value'),
            dcc.Input(id='p_value', type='number', value=0.05),
            html.Label('Log Fold Change (LFC)'),
            dcc.Input(id='lfc', type='number', value=1.0),
            html.Label('Coverage'),
            dcc.Input(id='coverage', type='number', value=1),
            html.Button('EXTRACCIÓN DE DEGs', id='DEGs', n_clicks=None, className='buttons'),
            html.Div([], id='output-degs'),
        ]),
        dcc.Tab(label='Feature Selection y Clasificación', style=tab_style, selected_style=tab_selected_style, children=[
            html.P(children='Escoge un algoritmo de Feature Selection y un algoritmo de clasificación:'),
            dcc.RadioItems(['mRMR', 'RandomForest', 'Chi2', 'Relief', 'MultiSurf', 'MIFS'], id='features', className='dash-radioitems', inline=True),
            dcc.RadioItems(['kNN', 'RandomForest', 'DecisionTree', 'LogisticRegression', 'NeuralNetworkMLP', 'Gradient Boosting'], id='model', className='dash-radioitems', inline=True),
            html.Button('CLASIFICACIÓN', id='clas', n_clicks=None, className='buttons'),
            html.Div([], id='output-clas'),
        ]),
    ], style=tabs_styles),


    dcc.Store(id='matrix'),
    dcc.Store(id='labels'),
    dcc.Store(id='degsmat'),
    dcc.Store(id='stats')
    #html.Div(id='selected-range')
])

@app.callback(
    dash.dependencies.Output('upload-data', 'children'),
    dash.dependencies.Output('upload-data-labels', 'style', allow_duplicate=True),
    dash.dependencies.Output('dropdown', 'style'),
    dash.dependencies.Output('filename', 'style'),
    dash.dependencies.Input('loadMode', 'value'),
    dash.dependencies.State('upload-data-labels', 'style'),
    dash.dependencies.State('dropdown', 'style'),
    dash.dependencies.State('filename', 'style'),
    prevent_initial_call=True
)
def changeInputMode(value, style1, style2, style3):
    if value == "Carga de Counts":
        style1['visibility'] = 'hidden'
        style2['visibility'] = 'visible'
        style3['visibility'] = 'hidden'
        return html.Div(['Selecciona el fichero data_info con los metadatos de los counts']), style1, style2, style3
    else:
        style1['visibility'] = 'visible'
        style2['visibility'] = 'hidden'
        style3['visibility'] = 'hidden'
        return html.Div(['Selecciona los ficheros con la matriz de counts procedentes de R']), style1, style2, style3

@app.callback(dash.dependencies.Output('filename', 'options', allow_duplicate=True),
              dash.dependencies.Output('filename', 'style', allow_duplicate=True),
              dash.dependencies.Output('dropdown', 'style', allow_duplicate=True),
              dash.dependencies.Input("upload-data", "contents"),
              dash.dependencies.State('upload-data', 'filename'),
              dash.dependencies.State('upload-data', 'last_modified'),
              dash.dependencies.State('loadMode', 'value'),
              dash.dependencies.State('filename', 'style'),
              dash.dependencies.State('dropdown', 'style'),
              prevent_initial_call=True
             )
def update_drop(list_of_contents, list_of_names, list_of_dates, mode, style, style2):
        if mode=="Carga de Counts":
            if list_of_contents is not None:
                style['visibility'] = 'visible'
                style2['visibility'] = 'visible'
                file = parse_file(list_of_contents, list_of_names, list_of_dates)
                opts = pd.read_csv(file).columns[1:]
                return opts, style, style2
        else:
            raise dash.exceptions.PreventUpdate
                

@app.callback(dash.dependencies.Output('output-data-upload', 'children', allow_duplicate=True),
              dash.dependencies.Output("matrix", "data", allow_duplicate=True),
              dash.dependencies.Output("labels", "data", allow_duplicate=True),
              dash.dependencies.Input("showMatrix", "n_clicks"),
              dash.dependencies.State('upload-data', 'contents'),
              dash.dependencies.State('upload-data', 'filename'),
              dash.dependencies.State('upload-data', 'last_modified'),
              dash.dependencies.State('loadMode', 'value'),
              dash.dependencies.State('dropdown', 'value'),
              dash.dependencies.State('output-data-upload', 'children'),
              dash.dependencies.State('filename', 'value'),
              dash.dependencies.State('upload-data-labels', 'contents'),
              dash.dependencies.State('upload-data-labels', 'filename'),
              dash.dependencies.State('upload-data-labels', 'last_modified'),
              prevent_initial_call=True
             )
def update_output_counts(n_clicks, list_of_contents, list_of_names, list_of_dates, mode, dir, children, col, c, n, d):
    if n_clicks is not None:
        if mode=="Carga de Counts":
            if list_of_contents is not None:
                file = parse_file(list_of_contents, list_of_names, list_of_dates)
                nm, dt = extractDataInfo(file, dir, id_column=col, label_column='Sample.Type')
                counts = counts_to_matrix(nm)
                listado = [a for a in counts["countsMatrix"].index]
                my_annotation = get_genes_annotation(listado)
                class_counts = dt.Class.value_counts()                
                results = calculateGeneExpression(counts["countsMatrix"], my_annotation)
                labels = pd.DataFrame(dt.Class)
                fig = px.bar(x=class_counts.index, y=class_counts.values, labels={'x': 'Clase', 'y': 'Nº de Muestras'}, title='Se ha calculado la expresión de '+ str(results.shape[0])+ ' genes para las '+str(results.shape[1])+' muestras.').update_layout(width=600, height=400)

        else:
            if list_of_contents is not None:
                results, dt = readFromR(list_of_names, n)
                class_counts = dt.value_counts()
                class_counts = class_counts.reset_index()
                labels = dt
                fig = px.bar(x=class_counts.values[:, 0], y=class_counts.values[:, 1], labels={'x': 'Clase', 'y': 'Nº de Muestras'})
                
        
        children.append(dcc.Graph(figure=fig, id='full-graph'))
        return children, results.to_json(date_format='iso', orient='split'), labels.to_json(date_format='iso', orient='split')









@app.callback(dash.dependencies.Output('output-qa', 'children', allow_duplicate=True),
              dash.dependencies.Output("matrix", "data", allow_duplicate=True),
              dash.dependencies.Output("labels", "data", allow_duplicate=True),
              dash.dependencies.Input("QA", "n_clicks"),
              dash.dependencies.State("matrix", "data"),
              dash.dependencies.State("labels", "data"),
              dash.dependencies.State('outliers', 'value'),
              dash.dependencies.State('output-qa', 'children'),
              prevent_initial_call=True
             )
def performQA_batchRemoval(n_clicks, ge, lab, methods_qa, children):
    if n_clicks is not None:
            matrix = pd.read_json(ge, orient='split' ,precise_float=True)
            labels = pd.read_json(lab, orient='split')
            name, pos = RNAseqQA(matrix, methods=methods_qa)
            qualityMatrix = matrix.drop(name, axis="columns")
            qualityLabels = labels.drop(pos, axis="index")
            batchMatrix = batchEffectRemoval(qualityMatrix, qualityLabels.to_numpy().flatten())
            batchMatrixdf = pd.DataFrame(batchMatrix)
            batchMatrixdf.columns = qualityMatrix.columns
            batchMatrixdf.index = qualityMatrix.index
            cp = PCA(2)
            cp.fit(batchMatrix)
            pcantes = PCA(2).fit(qualityMatrix)
            df = pd.DataFrame(np.vstack((cp.components_, pcantes.components_))).T
            df.columns = ["PC1A", "PC2A", "PC1B", "PC2B"]
            df["Sample"] = qualityMatrix.columns
            df["Class"] = qualityLabels
        
            fig = make_subplots(rows=1, cols=1, subplot_titles=["Sample Distribution before/after Batch Effect Removal"])
            classes = df["Class"].unique()
            for class_name in classes:
                class_df = df[df["Class"] == class_name]
                trace = go.Scatter(x=class_df["PC1A"], y=class_df["PC2A"],
                                   name=f"Batch Corrected - {class_name}", mode='markers', text=class_df["Sample"])
                fig.add_trace(trace)
                trace = go.Scatter(x=class_df["PC1B"], y=class_df["PC2B"],
                                   name=f"Raw - {class_name}", mode='markers', text=class_df["Sample"])
                fig.add_trace(trace)
            
            fig.update_layout(height=600, width=800, xaxis=dict(title='PC1'), yaxis=dict(title='PC2'), showlegend=True, legend=dict(x=1.02, y=0.55))

            children.append(dcc.Graph(figure=fig, id='pca'))
            
                
            return children, batchMatrixdf.to_json(date_format='iso', orient='split'), pd.DataFrame(qualityLabels).to_json(date_format='iso', orient='split')

import dash_bio as dashbio

@app.callback(dash.dependencies.Output('output-degs', 'children', allow_duplicate=True),
              dash.dependencies.Output("degsmat", "data", allow_duplicate=True),
              dash.dependencies.Output("stats", "data", allow_duplicate=True),
              dash.dependencies.Input("DEGs", "n_clicks"),
              dash.dependencies.State("matrix", "data"),
              dash.dependencies.State("labels", "data"),
              dash.dependencies.State('output-degs', 'children'),
              dash.dependencies.State("p_value", "value"),
              dash.dependencies.State("lfc", "value"),
              dash.dependencies.State("coverage", "value"),
              prevent_initial_call=True
             )
def performDEG_extraction(n_clicks, ge, lab, children, p, lfc, cov):
    if n_clicks is not None:
            matrix = pd.read_json(ge, orient='split' ,precise_float=True)
            labels = pd.read_json(lab, orient='split')
            results, stats = DEGsExtraction(matrix, pd.DataFrame(labels), False, p, lfc, cov)
            if np.unique(labels).shape[0] == 2:
                stats = pd.DataFrame(stats["DEGs_Table"])
                stats["GENE"] = stats.index
                stats.reset_index(inplace=True)
                
    
                new = html.Div([
                'LFC Limits',
                dcc.RangeSlider(
                    id='volcanoplot-input',
                    min=-lfc-6,
                    max=lfc+6,
                    step=0.05,
                    marks={i: {'label': str(i)} for i in range(int(-lfc-6), int(lfc+6))},
                    value=[-lfc, lfc]
                ),
                html.Br(),
                html.Div(
                    dcc.Graph(
                        id='volcanoplot',
                        figure = dashbio.VolcanoPlot(
                                    dataframe=stats,
                                    snp=None,
                                    effect_size="logFC",
                                    p="P.Value",
                                    point_size=5,
                                    effect_size_line_width=4,
                                    genomewideline_width=2,
                                    genomewideline_value=10**(-p) ,
                                    effect_size_line=[-lfc, lfc],
                                    xlabel="Log Fold Change"
                                )
                    )
                )
                ])
                children.append(new)
                return children, results.to_json(date_format='iso', orient='split'), stats.to_json(date_format='iso', orient='split')
            else:
                classes = np.unique(labels)
                num_class = np.unique(labels).shape[0]
                combinaciones = list(combinations(classes, 2))
                num_combinaciones = len(combinaciones)
                ls = ["LFC "+str(c[0])+" - "+str(c[1]) for c in combinaciones]
                ps = ["p-Value "+str(c[0])+" - "+str(c[1]) for c in combinaciones]
                columnas = ls + ps
        
                serie2 = pd.Series(matrix.index)
                serie1 = pd.Series(results.columns)
                i = np.array(serie2[serie2.isin(serie1)].index)
                
                pvals = pd.DataFrame(stats["DEGs_Table"]['p.value']).iloc[i]
                lfcs = pd.DataFrame(stats["MulticlassLFC"])
                mstats = pd.DataFrame(np.hstack((np.array(lfcs), np.array(pvals))))
                mstats.index = results.columns
                mstats.columns = columnas


                for j in range(num_combinaciones):
                    if j==0:
                        mstats["GENE"] = stats.index
                        mstats.reset_index(inplace=True)
                        
                    new = html.Div([
                    'LFC Limits',
                    dcc.RangeSlider(
                        id='volcanoplot-input',
                        min=-lfc-6,
                        max=lfc+6,
                        step=0.05,
                        marks={i: {'label': str(i)} for i in range(int(-lfc-6), int(lfc+6))},
                        value=[-lfc, lfc]
                    ),
                    html.Br(),
                    html.Div(
                        dcc.Graph(
                            id='volcanoplot',
                            figure = dashbio.VolcanoPlot(
                                        dataframe=mstats.loc[:, ["index", "LFC "+str(combinaciones[j][0])+" - "+str(combinaciones[j][1]), "p-Value "+str(combinaciones[j][0])+" - "+str(combinaciones[j][1])]],
                                        snp=None,
                                        gene="index",
                                        effect_size="LFC "+str(combinaciones[j][0])+" - "+str(combinaciones[j][1]),
                                        p="p-Value "+str(combinaciones[j][0])+" - "+str(combinaciones[j][1]),
                                        point_size=5,
                                        effect_size_line_width=4,
                                        genomewideline_width=2,
                                        genomewideline_value=10**(-p) ,
                                        effect_size_line=[-lfc, lfc],
                                        xlabel="Log Fold Change"
                                    ).update_layout(title="Contraste "+str(combinaciones[j][0])+" - "+str(combinaciones[j][1]))
                        )
                    )
                    ])
                    children.append(new)

                return children, results.to_json(date_format='iso', orient='split'), mstats.to_json(date_format='iso', orient='split')


@app.callback(dash.dependencies.Output('output-clas', 'children', allow_duplicate=True),
              dash.dependencies.Input("clas", "n_clicks"),
              dash.dependencies.State("matrix", "data"),
              dash.dependencies.State("labels", "data"),
              dash.dependencies.State('output-clas', 'children'),
              dash.dependencies.State("features", "value"),
              dash.dependencies.State("model", "value"),
              dash.dependencies.State("p_value", "value"),
              dash.dependencies.State("lfc", "value"),
              dash.dependencies.State("coverage", "value"),
              prevent_initial_call=True
             )
def selectClassify(n_clicks, degs, lab, children, features, model, p, lfc, cov):
    if n_clicks is not None:
        matrix = pd.read_json(degs, orient='split' ,precise_float=True)
        labels = pd.read_json(lab, orient='split')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        features_set=[]
        confusions=[]
        accs=[]
        pres=[]
        recs=[]
        foldOneAcc=[]
        foldOnePre=[]
        foldOneRec=[]
        firstfold=True
        
        for train_index, test_index in skf.split(matrix.T, labels):
                    matrix_train, matrix_test = matrix.T.iloc[train_index].T, matrix.T.iloc[test_index]
                    labels_train, labels_test = labels.iloc[train_index], labels.iloc[test_index]
        
                    matrix_train, stats = DEGsExtraction(matrix_train, labels_train, False, p, lfc, cov)
                    matrix_test = matrix_test.loc[:, matrix_train.columns]
            
                    # Realizar selección de características
                    selected = featureSelect(matrix_train, labels_train, 10, method=features)
                    top_10_features = selected.iloc[:10]
                    features_set.append(top_10_features)
                    matrix_train = matrix_train.loc[:, top_10_features.index]
                    matrix_test = matrix_test.loc[:, top_10_features.index]
        
                    # Ajustar el modelo y realizar predicciones
                    model_instance, param_grid = chooseModel(model)
                    grid_search = GridSearchCV(model_instance, param_grid, cv=5, n_jobs=-1)  # Ajusta cv según sea necesario
                    grid_search.fit(matrix_train, labels_train)
                    predictions = grid_search.predict(matrix_test)
        
                    # Generar informe con matriz de confusión y métricas
                    confusion = confusion_matrix(labels_test, predictions)
                    accs.append(accuracy_score(labels_test, predictions))
                    pres.append(precision_score(labels_test, predictions, average="macro"))
                    recs.append(recall_score(labels_test, predictions, average="macro"))
                    confusions.append(confusion)

                    if firstfold:
                        for i in range(10):
                            newmodel = model_instance.set_params(**grid_search.best_params_)
                            newmodel.fit(matrix_train.loc[:, top_10_features.iloc[:i+1].index], labels_train)
                            predictions = newmodel.predict(matrix_test.loc[:, top_10_features.iloc[:i+1].index])
                            confusion = confusion_matrix(labels_test, predictions)
                            foldOneAcc.append(accuracy_score(labels_test, predictions))
                            foldOnePre.append(precision_score(labels_test, predictions, average="macro"))
                            foldOneRec.append(recall_score(labels_test, predictions, average="macro"))
                        print(foldOneAcc)
                        print(foldOnePre)
                        print(foldOneRec)
                        df = pd.DataFrame({
                            'Number of genes': np.arange(1, 11),
                            'Accuracy': foldOneAcc,
                            'Specificity': foldOnePre,
                            'Sensitivity': foldOneRec
                        })
                        children.append(html.H3("Resultados obtenidos:"))
                        children.append(html.P("En la primera partición, observamos como la inclusión de los genes de uno en uno en orden de importancia, va influyendo en los resultados en test."))
                        children.append(dcc.Graph(id="comp", figure=px.line(df, x='Number of genes', y=['Accuracy', 'Specificity', 'Sensitivity'], title="Classification Results: " + model)))
                        
                        firstfold=False

        listado = [str(i) for j in features_set for i in j.index]
        common = pd.DataFrame(Counter(listado).most_common(), columns=["GENE", "COUNT"])
        confusion_sum = np.sum(confusions, axis=0)

        fig = px.bar(common, x="GENE", y="COUNT", color="COUNT",
        labels={'Number of folds': 'Number of folds'},
        title='Most Robust Genes by '+ features,
        template='plotly_dark')  

        fig.update_layout(xaxis=dict(tickangle=-45, tickmode='array', tickvals=top_10_features.index),
                              yaxis_title='Number of folds',
                              xaxis_title='Features')
        children.append(html.P("Ahora se muestran los genes más robustos, entendidos como aquellos que se seleccionan en más particiones."))
        children.append(dcc.Graph(id="importances", figure=fig))

        dfr = pd.DataFrame({
                            'Fold Number': [np.arange(1, 6), "Mean"],
                            'Accuracy': [accs, np.mean(accs)],
                            'Specificity': [pres, np.mean(pres)],
                            'Sensitivity': [recs, np.mean(recs)]
                        })

        fm = plot_confusion_matrix(confusion_sum, np.unique(labels), "Confusion Matrix 5-Fold - " + features + " " + model)
        children.append(html.P("A continuación se muestran los resultados de la validación cruzada:"))
        #children.append(dash_table.DataTable(dfr.to_dict('records'), [{"name": i, "id": i} for i in dfr.columns]))
        children.append(dcc.Graph(id="cm", figure=fm))

        cfs=[]
        accsF=[]
        presF=[]
        recsF=[]
        
        for train_index, test_index in skf.split(matrix.T, labels):
                    matrix_train, matrix_test = matrix.T.iloc[train_index].T, matrix.T.iloc[test_index]
                    labels_train, labels_test = labels.iloc[train_index], labels.iloc[test_index]
                    print(common[common["COUNT"]>=4].shape)
                    if(common[common["COUNT"]>=4].shape[1] > 5):
                        matrix_train = matrix_train.T.loc[:, common[common["COUNT"]>=4]["GENE"]]
                        matrix_test = matrix_test.loc[:, common[common["COUNT"]>=4]["GENE"]]
                    else:
                        matrix_train = matrix_train.T.loc[:, common[common["COUNT"]>=2]["GENE"]]
                        matrix_test = matrix_test.loc[:, common[common["COUNT"]>=2]["GENE"]]                        
                    model_instance, param_grid = chooseModel(model)
                    grid_search = GridSearchCV(model_instance, param_grid, cv=5, n_jobs=-1)  # Ajusta cv según sea necesario
                    grid_search.fit(matrix_train, labels_train)
                    predictions = grid_search.predict(matrix_test)
        
                    # Generar informe con matriz de confusión y métricas
                    confusion = confusion_matrix(labels_test, predictions)
                    accsF.append(accuracy_score(labels_test, predictions))
                    presF.append(precision_score(labels_test, predictions, average="macro"))
                    recsF.append(recall_score(labels_test, predictions, average="macro"))
                    cfs.append(confusion)
        confusion_sum2 = np.sum(cfs, axis=0)
        fm = plot_confusion_matrix(confusion_sum2, np.unique(labels), "Confusion Matrix 5-Fold Robust - " + features + " " + model)
        dff = pd.DataFrame({
                            'Fold Number': [np.arange(1, 6), "Mean"],
                            'Accuracy': [accsF, np.mean(accsF)],
                            'Specificity': [presF, np.mean(presF)],
                            'Sensitivity': [recsF, np.mean(recsF)]
                        })
        children.append(html.P("Repitiendo la validación cruzada, sólo con los genes más robustos."))
        #children.append(dash_table.DataTable(dff.to_dict('records'), [{"name": i, "id": i} for i in dff.columns]))
        children.append(dcc.Graph(id="cm2", figure=fm))
                    
                
        return children

@app.callback(
    dash.dependencies.Output('volcanoplot', 'figure'),
    dash.dependencies.Input('volcanoplot-input', 'value'),
    dash.dependencies.State('stats', 'data'),
    dash.dependencies.State("p_value", "value"),
    dash.dependencies.State("lfc", "value"),
    suppress_callback_exceptions=True, prevent_initial_call=True
)
def update_volcanoplot(effects, degs, p, lfc):
    matrix = pd.read_json(degs, orient='split' ,precise_float=True)
    return dashbio.VolcanoPlot(
                                dataframe=matrix,
                                snp=None,
                                effect_size="logFC",
                                p="P.Value",
                                point_size=5,
                                effect_size_line_width=4,
                                genomewideline_width=2,
                                genomewideline_value=10**(-p) ,
                                effect_size_line=effects
                            )

if __name__ == '__main__':
    app.run_server(debug=True)