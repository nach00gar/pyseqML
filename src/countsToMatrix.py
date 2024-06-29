import pandas as pd
import numpy as np
import conorm
import os


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
        print(count_file)
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
        print("Using reference genome 38.")
        my_annotation = pd.read_csv('GRCh38Annotation.csv')
        if filter in my_annotation.columns and set(attributes).issubset(my_annotation.columns):
            my_annotation = my_annotation[my_annotation[filter].isin(values)][attributes]
            return my_annotation
        else:
            dataset_name = 'hsapiens_gene_ensembl'
            filename = f"{dataset_name}.csv"
    else:
        if notHumandataset == "":
            raise ValueError("Sorry. NotHumandatasets are currently not supported.")

        return my_annotation
    