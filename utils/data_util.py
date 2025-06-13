from collections import OrderedDict
import numpy as np
import torch
import random
from scipy import spatial
import os
from .tool_utils import *
import scanpy as sc, anndata as ad
import pandas as pd
def read_folder_sc(folder_path):
    adata = sc.read_mtx(os.path.join(folder_path, 'count_matrix_sparse.mtx'))
    adata_bc=pd.read_csv(os.path.join(folder_path, 'count_matrix_barcodes.tsv'),header=None)
    adata_features=pd.read_csv(os.path.join(folder_path, 'count_matrix_genes.tsv'),header=None)
    adata_meta=pd.read_csv(os.path.join(folder_path, 'metadata.csv'),header=0, index_col=0)
    adata= adata.T
    adata.obs_names = adata_bc[0].tolist()
    adata.var['gene_name']= adata_features[0].tolist()
    adata.var.index= adata.var['gene_name']
    adata.obs = adata_meta
    return adata
def read_kidney_folder(folder_path, sample):
    adata = sc.read_mtx(os.path.join(folder_path, sample, 'matrix.mtx'))
    adata_bc=pd.read_csv(os.path.join(folder_path, sample, 'barcodes.tsv'), header=None)
    adata_features=pd.read_csv(os.path.join(folder_path, sample, 'features.tsv'),header=None, sep="\t").iloc[:,1].values
    #adata_feature_select = np.load(os.path.join(folder_path, 'hvg_union.npy'))
    #adata_features = adata_features[adata_feature_select]
    adata= adata.T
    adata.obs_names= adata_bc[0].tolist()
    adata.var['gene_name']= adata_features
    adata.var.index= adata.var['gene_name']
    return adata
def read_kidney_folder_harmony(folder_path, sample):
    adata_mtx = np.load(os.path.join(folder_path, sample, 'harmony_matrix.npy')).T
    adata_bc=pd.read_csv(os.path.join(folder_path, sample, 'barcodes.tsv'), header=None)
    adata_features=pd.read_csv(os.path.join(folder_path, sample, 'features.tsv'),header=None, sep="\t").iloc[:,1].values
    adata_feature_select = np.load(os.path.join(folder_path, 'hvg_union.npy'))
    adata_features = adata_features[adata_feature_select]
    adata= ad.AnnData(X=adata_mtx)
    adata.obs_names= adata_bc[0].tolist()
    adata.var['gene_name']= adata_features
    adata.var.index= adata.var['gene_name']
    return adata


def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True