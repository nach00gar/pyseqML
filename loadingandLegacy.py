import pandas as pd
import os
import conorm
import requests
from xml.etree import ElementTree
import os, shutil
import pathlib
import rpy2.rinterface
from rpy2.robjects.packages import importr, data
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects import conversion, default_converter
import base64
import datetime
import io
from dash import dash_table
import pyreadr


utils = importr('utils')
base = importr('base')
limma = importr('limma')
arrow = importr('arrow')
cqn = importr('cqn')
sva = importr('sva')

import rpy2.robjects as ro
r = ro.r
r['source']('calculateGeneExpressionValues.R')
r['source']('batchEffectRemoval.R')
r['source']('DEGsExtraction.R')
calculateGEVs = ro.globalenv['calculateGeneExpressionValues']
removeBatchEffect = ro.globalenv['batchEffectRemoval']
deg_extraction = ro.globalenv['DEGsExtraction']


def extractDataInfo(samples_info, dir_name, id_column='Internal.ID', label_column='Sample.Type'):
    samples = pd.read_csv(samples_info)
    path_name = "data/" + dir_name

    if id_column == 'Internal.ID':
        Run = samples[id_column].astype(str) + ".count"
    if id_column == 'File.Name':
        Run = samples[id_column].astype(str)
    Path = [path_name] * samples.shape[0]
    Class = samples[label_column]
    print(Class.value_counts())

    data_info = pd.DataFrame({'Run': Run, 'Path': Path, 'Class': Class})
    name = "data_info_" + dir_name + ".csv"
    data_info.to_csv(name, index=False)
    return name, data_info

def counts_to_matrix(csv_file, sep=',', extension=''):
    if not os.path.exists(csv_file):
        raise FileNotFoundError("Unable to find the CSV file. Please check the path to the file.")

    counts_data = pd.read_csv(csv_file, sep=sep)
    required_columns = ["Run", "Path", "Class"]
    if not all(col in counts_data.columns for col in required_columns):
        raise ValueError("The CSV file must have the following three columns: Run, Path, Class.")

    count_files = []
    for i in range(len(counts_data)):
        directory = f"{counts_data['Path'][i]}/{counts_data['Run'][i]}"
        if extension:
            count_files.append(f"{directory}.{extension}")
        else:
            count_files.append(directory)

    print(f"\nMerging {len(counts_data)} counts files...\n")
    counts_matrix = pd.DataFrame()
    for count_file in count_files:
        single_counts = pd.read_csv(count_file, sep='\t', header=None, names=['EnsemblID', 'Count'], index_col='EnsemblID')
        counts_matrix = pd.concat([counts_matrix, single_counts], axis=1)

    #counts_matrix = counts_matrix[~counts_matrix.index.isin(["__no_feature", "__ambiguous", "__too_low_aQual", "__not_aligned", "__alignment_not_unique", "no_feature", "alignment_not_unique", "ambiguous", "too_low_aQual", "not_aligned"])]
    cpms = conorm.cpm(counts_matrix)
    keep = (cpms > 1).sum(axis=1) >= 1
    counts_matrix = counts_matrix.loc[keep]
    counts_matrix = counts_matrix[~counts_matrix.index.isin(["__no_feature", "__ambiguous", "__too_low_aQual", "__not_aligned", "__alignment_not_unique", "no_feature", "alignment_not_unique", "ambiguous", "too_low_aQual", "not_aligned"])]

    ensembl_ids = [entry.split(".")[0] for entry in counts_matrix.index]
    counts_matrix.index = ensembl_ids
    counts_matrix.columns = count_files
    results = {"countsMatrix": counts_matrix, "labels": counts_data["Class"]}
    return results

def get_genes_annotation(values, attributes=["ensembl_gene_id", "external_gene_name", "percentage_gene_gc_content", "entrezgene_id"], filter="ensembl_gene_id", notHSapiens=False, notHumandataset="", referenceGenome=38):
    if not isinstance(attributes, list) or not all(isinstance(attr, str) for attr in attributes):
        raise ValueError("The parameter 'attributes' must be a list of strings containing the wanted Ensembl attributes.")
    if not isinstance(values, list) or not all(isinstance(val, str) for val in values):
        raise ValueError("The parameter 'values' must be a list of strings containing the gene IDs.")
    if not isinstance(filter, str):
        raise ValueError("The parameter 'filter' must be a string containing the attribute used as a filter.")
    if not isinstance(notHSapiens, bool):
        raise ValueError("The 'notHSapiens' parameter can only take the values True or False.")
    if referenceGenome not in [37, 38]:
        raise ValueError("Introduced referenceGenome is not available; it must be 37 or 38.")

    base = 'http://www.ensembl.org/biomart/martservice'
    if not notHSapiens:
        print("Getting annotation of the Homo Sapiens...")
        if referenceGenome == 38:
            print("Using reference genome 38.")
            my_annotation = pd.read_csv('GRCh38Annotation.csv')
            if filter in my_annotation.columns and set(attributes).issubset(my_annotation.columns):
                my_annotation = my_annotation[my_annotation[filter].isin(values)][attributes]
                return my_annotation
            else:
                dataset_name = 'hsapiens_gene_ensembl'
                filename = f"{dataset_name}.csv"
        else:
            print("Using reference genome 37.")
            base = 'https://grch37.ensembl.org/biomart/martservice/'
            dataset_name = 'hsapiens_gene_ensembl'
            filename = f"{dataset_name}.csv"
    else:
        if notHumandataset == "":
            raise ValueError("The notHumandataset is empty! Please provide a valid notHumandataset.")
        dataset_name = notHumandataset
        filename = f"{notHumandataset}.csv"
    '''
    print(f"Downloading annotation {dataset_name}...")
    
    act_values = values
    max_values = min(len(values), 900)
    my_annotation = pd.DataFrame()

    while len(act_values) > 0:
        query = f'<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query><Query virtualSchemaName="default" formatter="CSV" header="0" uniqueRows="0" count="" datasetConfigVersion="0.6"><Dataset name="{dataset_name}" interface="default">'
        
        if len(values) > 1 or values != 'allGenome':
            query += f'<Filter name="{filter}" value="{",".join(act_values[:max_values])}"/>'

        for attribute in attributes:
            query += f'<Attribute name="{attribute}" />'
        if filter not in attributes:
            query += f'<Attribute name="{filter}" />'
        query += '</Dataset></Query>'

        # Download annotation file
        try:
            response = requests.get(f"{base}?query={query}")
            act_my_annotation = pd.read_csv(pd.compat.StringIO(response.text), sep=',', header=None)
            act_my_annotation.columns = attributes + [filter]
        except Exception as e:
            print('\nConnection error, trying again...')
            act_my_annotation = pd.DataFrame(columns=attributes)
        
        if 'ERROR' in act_my_annotation.iloc[0, 0]:
            raise ValueError('Error in query, please check attributes and filter')

        if my_annotation.shape[0] == 0:
            my_annotation = act_my_annotation
        else:
            my_annotation = pd.concat([my_annotation, act_my_annotation], ignore_index=True)

        if len(act_values) <= max_values:
            act_values = []
        else:
            act_values = act_values[max_values+1:]

        max_values = min(900, len(act_values))

    if len(values) > 1 or values != 'allGenome':
        my_annotation = my_annotation[my_annotation[filter].isin(values)]
    '''
    return my_annotation


def calculateGeneExpression(counts, annot):
    with conversion.localconverter(default_converter):
        counts.to_parquet("counts.parquet")
        r_counts = arrow.read_parquet("counts.parquet")
        annot.to_parquet("my_annotation.parquet")
        r_annot = arrow.read_parquet("my_annotation.parquet")

        r_counts.rownames = r_counts[0]
        r_counts = base.subset(r_counts, select=-1)    
        m = base.as_matrix(r_counts)
    
        res = calculateGEVs(m, r_annot)
        print(res)
    with localconverter(ro.default_converter + pandas2ri.converter):
        p = ro.conversion.rpy2py(res)

        results = pd.DataFrame(p)
        results.index = [str(n) for n in res.rownames]
        results.columns = [str(s) for s in res.colnames]

    return results


def batchEffectRemoval(qm, ql, method="sva", batches=[]):
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        qm.to_parquet("quality2.parquet")
        quality = arrow.read_parquet("quality2.parquet")
        quality = base.subset(quality, select=-1)    
        m = base.as_matrix(quality)
        removed = removeBatchEffect(m, ro.StrVector(ql), method="sva")
        p = ro.conversion.rpy2py(removed)
        results = pd.DataFrame(p)
    return results


def parse_file(contents, filename, date):
    print("Subiendo fichero...")
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        fichero = io.StringIO(decoded.decode('utf-8'))
    except Exception as e:
        print(e)
        return html.Div([
            'Error leyendo fichero '+filename
        ])
    return fichero


def readFromR(matrixFile, labelsFile, sep=False):
    df = pyreadr.read_r(matrixFile).popitem()[1]
    target = pyreadr.read_r(labelsFile).popitem()[1]
    return df, target

def obtener_subdirectorios(ruta):
    return [d for d in os.listdir(ruta) if os.path.isdir(os.path.join(ruta, d))]



def DEGsExtraction(rdf, labels, inR, p, lfcs, covs):
    if not inR:
        with localconverter(ro.default_converter):
            hora = str(datetime.datetime.now())[-9:]
            rdf.to_parquet("temporal"+hora+".parquet")
            rdf = arrow.read_parquet("temporal"+hora+".parquet")
            labels.to_parquet("temporalt"+hora+".parquet")
            labels = arrow.read_parquet("temporalt"+hora+".parquet")

    with localconverter(ro.default_converter):
        rdf.rownames = rdf[0]
        ppr = base.subset(rdf, select=-1)
    with localconverter(ro.default_converter): 
        print(base.length(ro.vectors.FactorVector(labels[1])))
        print(base.as_matrix(ppr, rownames=True).nrow)
        print(base.as_matrix(ppr, rownames=True).ncol)
        degs = deg_extraction(base.as_matrix(ppr, rownames=True), ro.vectors.FactorVector(labels[1]), pvalue=p, lfc=lfcs, cov=covs)
    with localconverter(ro.default_converter + pandas2ri.converter):
        p = ro.conversion.rpy2py(degs)
        df = pd.DataFrame(p['DEG_Results']['DEGs_Matrix'])
    with localconverter(ro.default_converter): 
        df.index = degs[0][1].rownames
    return df.T, p['DEG_Results']