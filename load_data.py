#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ast import Num
from tkinter import N
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def Change_part_name(names,string_to_replace, string_new):
    """
    This fucntion is desinged to replace one string for another string

    :parm names: a list of string names 
    :parm string_to_replace: a string that should be replaced 
    :parm string_to_replace: a string that should be replaced it with 

    :Return correct_names: a list containing the replace string 

    """
    correct_names = []
    for n in names:
        if string_to_replace in n:
            n = n.replace(string_to_replace, string_new)
        correct_names.append(n)
    return correct_names


def load_multi_omics(data_path="Active vs remission omics datasets"):
    """
    This function loads and preprocesses multi-omics data across six data domains for use in a stacked model.

    Parameters:
    - data_path: Path to the datasets directory.

    Returns:
    - dataset: A combined DataFrame containing features across all six domains.
    - y: DataFrame containing the labels (1 for high, 0 for low).
    - datasets_names: List of names for each data domain.
    - data_positions: List of index arrays for each data domain.
    - feat_names_by_group: List of feature names by data group.
    """

    # Load each dataset as previously done
    Samples_overview = pd.read_excel(f"{data_path}/Sample_overview.xlsx", index_col=0, header=1)
    Samples_overview = Samples_overview.loc[Samples_overview['Diagnosis'] == 0]

    Proteomics = pd.read_excel(f"{data_path}/Proteomics_feces.xlsx", index_col='Protein IDs')
    Metabolomics_Fecal = pd.read_excel(f"{data_path}/MS_Fecal_Water_Urine_Plasma.xlsx", index_col=0, sheet_name="Fecal Water")
    Metabolomics_Fecal.columns = Change_part_name(Metabolomics_Fecal.columns, "P", "p")

    Metabolomics_Urine = pd.read_excel(f"{data_path}/MS_Fecal_Water_Urine_Plasma.xlsx", index_col=0, sheet_name="Urine Creatinine Corrected")
    Metabolomics_Urine.columns = Change_part_name(Metabolomics_Urine.columns, "P", "p")

    Metabolomics_Plasma = pd.read_excel(f"{data_path}/MS_Fecal_Water_Urine_Plasma.xlsx", index_col=0, sheet_name="Plasma")
    Metabolomics_Plasma.columns = Change_part_name(Metabolomics_Plasma.columns, "P", "p")

    Microbial = pd.read_excel(f"{data_path}/16S_Feces.xlsx", index_col=1, sheet_name="normalized")
    ITS = pd.read_excel(f"{data_path}/ITS_Feces.xlsx", index_col=0)

    datasets_names = ["Proteomics", "Metabolomics_Fecal", "Metabolomics_Urine", "Metabolomics_Plasma", "Microbial", "ITS"]
    data_positions = []
    feat_names_by_group = []

    for i, name in enumerate(datasets_names):
        exec(f"feat_names_by_group.append(np.array(list({name}.index)))")
        if i == 0:
            data_positions.append(np.array(range(0, feat_names_by_group[i].shape[0])))
        else:
            begin = max(data_positions[i - 1]) + 1
            end = begin + feat_names_by_group[i].shape[0]
            data_positions.append(np.array(range(begin, end)))

    # Combine all datasets
    dataset = pd.concat([Proteomics, Metabolomics_Fecal, Metabolomics_Urine, Metabolomics_Plasma, Microbial, ITS], axis=0, join="outer")
    dataset = pd.DataFrame(dataset.T)
    dataset = dataset.loc[Samples_overview.index]

    # Apply median imputation
    imputer = SimpleImputer(strategy='median')
    dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns, index=dataset.index)

    # Skip sparsity filtering; directly set data positions and feature names by group
    data_positions = []
    feat_names_by_group = []
    for i, name in enumerate(datasets_names):
        feat_names_by_group.append(dataset.columns[data_positions[i]].to_numpy())
        if i == 0:
            data_positions.append(np.array(range(0, feat_names_by_group[i].shape[0])))
        else:
            begin = max(data_positions[i - 1]) + 1
            end = begin + feat_names_by_group[i].shape[0]
            data_positions.append(np.array(range(begin, end)))

    # Create label DataFrame
    y = Samples_overview.iloc[:, -1].replace({"high": 1, "low": 0}).astype(int)
    y = pd.DataFrame(y, columns=['y'])

    return dataset, y, datasets_names, data_positions, feat_names_by_group