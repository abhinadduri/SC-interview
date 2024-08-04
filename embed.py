import argparse
import json
import sys
import torch

import numpy as np
import pandas as pd
import scanpy as sc
import scgpt as scg

from pathlib import Path
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed
from scgpt.preprocess import Preprocessor

parser = argparse.ArgumentParser(description="Use scGPT foundation model to embed gene expression")
parser.add_argument("--model-dir", default="scgpt", dest="model_dir")
parser.add_argument("-seed", type=int, default=42, dest="seed")
parser.add_argument("--featurizer", default="scgpt", choices=["scgpt", "pca", "umap"], dest="featurizer")
parser.add_argument("--num-pcs", type=int, default=25, dest="num_pcs")
parser.add_argument("--out-file", default="embeddings.h5ad", dest="out_file")

def main(args):
    """
    Embed the gene expression data according to different featurizers.

    scgpt: This uses the scGPT foundation model to featurize the data
    pca: This projects the dimensionality of the data down to the number of principal components. Defaults to 25 but may be specified by user.
    """

    try:
        data_file = 'cells.npy'
        cells = np.load(data_file, allow_pickle=True).item()
    except:
        print("Please download the datafile into this folder: https://drive.google.com/file/d/1nsmQHdWek4YzIfKs9xUnLBHxKBWu8hUJ/view?usp=drive_link")

    adata = sc.AnnData(X=cells['UMI'].toarray(), obs={'celltype': cells['classes']}, var={'gene_id': cells['gene_ids']})
    assert args.out_file.endswith('h5ad'), "please specify an out file that ends in h5ad"

    # preprocess, in the way recommended by scGPT
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if args.featurizer == "scgpt":
        # compute highly variable genes which are the only ones that will be featurized
        N_HVG = 1800
        sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)
        adata = adata[:, adata.var['highly_variable']]
        print("Computed highly variable genes")

        # these params were taken from the scGPT zero shot tutorial. 
        # for more information, see: https://github.com/bowang-lab/scGPT/blob/main/tutorials/zero-shot/Tutorial_ZeroShot_Integration_Continual_Pretraining.ipynb
        sys.path.insert(0, "../")
        set_seed(args.seed)
        pad_token = "<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        n_bins = 51
        mask_value = -1
        pad_value = -2
        n_input_bins = n_bins
        gene_col = "Gene Symbol"
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Our goal is to lineraly probe the model. We will generate embeddings and then use those as inputs to our network (essentially freezing the foundation model weights)
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        for file in [model_config_file, model_file, vocab_file]:
            assert file.exists(), f"Required file {file} not found in {model_dir}, please download the model files (found on scGPT repo): https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
        
        # The gene id info provided in the cells.npy file are ENSEMBL gene ids, but we need to convert these to the vocab that scGPT expects.
        # scGPT provides a helper csv for this conversion task, also linked in their repo.
        ensembl_gene_info = model_dir / "gene_info.csv"
        assert ensembl_gene_info.exists(), f"gene_info.csv not found in {model_dir}, please download the conversion file here: https://github.com/bowang-lab/scGPT/files/13243634/gene_info.csv"

        ensembl_df = pd.read_csv(ensembl_gene_info, sep=',', index_col=2) # column 2 is ensembl name
        adata.var[gene_col] = adata.var['gene_id'].apply(lambda id: ensembl_df.loc[id, 'feature_name'] if id in ensembl_df.index else id)
        print("Converted ENSEMBL IDs to gene symbols")

        embed_adata = scg.tasks.embed_data(
            adata,
            args.model_dir,
            gene_col=gene_col,
            batch_size=64,
            device=device,
        )

    elif args.featurizer == 'pca':
        print("Computing the PCA for our genes...")
        sc.pp.pca(adata, n_comps=args.num_pcs)
        embed_adata = sc.AnnData(X=adata.obsm['X_pca'], dtype="float32")

    embed_adata.write_h5ad(args.out_file, compression='gzip')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
