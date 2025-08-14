import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd

"""
This linear method comes from https://www.nature.com/articles/s41592-025-02772-6
The method produced an overall score of 0.00 on the VCC leaderboard
  * Differential expression score: 0.00
  * Perturbation discrimination score: 0.51
  * Mean absolute error: 2.37
"""

adata = ad.read_h5ad('vcc_data/adata_Training.h5ad')

df = pd.DataFrame.sparse.from_spmatrix(adata.X, index=adata.obs['target_gene'])
group_means = df.groupby('target_gene', observed=True).mean()
group_means = group_means.drop(index='non-targeting')


mean_unperturbed = np.asarray(adata[adata.obs['target_gene'] == 'non-targeting'].X.mean(axis=0))
mean_perturbed = np.asarray(adata[adata.obs['target_gene'] != 'non-targeting'].X.mean(axis=0))

perturbed_genes = set(adata.obs['target_gene'])
perturbed_genes.remove('non-targeting')
perturbed_mask = adata.var_names.isin(perturbed_genes)

top_k = 50
sc.pp.pca(adata, n_comps=top_k, svd_solver='arpack')

G = adata.varm['PCs'] # top k principal components
P = G[perturbed_mask, :]

lmbda = 0.1 # ridge penalty
I = np.eye(top_k)

# Train
W = (
    np.linalg.inv(G.T @ G + lmbda * I) @ G.T @
    (group_means.values - mean_perturbed).T @
    P @ np.linalg.inv(P.T @ P + lmbda * I)
)

# Predict
val_adata = pd.read_csv('vcc_data/pert_counts_Validation.csv')
val_genes = val_adata['target_gene'].values
test_mask = adata.var_names.isin(val_genes)
P_val = G[test_mask, :]

Yhat = G @ W @ P_val.T + mean_perturbed.T

Yhat = np.concatenate((mean_unperturbed, Yhat.T), axis=0)
observations = ['non-targeting'] + val_genes.tolist()
obs_df = pd.DataFrame({'target_gene': observations})
adata_pred = ad.AnnData(X=Yhat, obs=obs_df, var=adata.var)

adata_pred.write('/home/ngalioto/vcc_2025/linear_baseline.h5ad') # save predictions
pd.Series(adata.var_names).to_csv("genes.csv", index=False, header=False) # save gene list