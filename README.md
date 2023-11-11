# MMCL-CDR:
#### Enhancing Cancer Drug Response Prediction with Multi-Omics and Morphology Images  Contrastive Representation Learning

### Quick start

- Run MMCL-CDR:
```
$ python main.py 
```

### Code and data

- `model.py`: MMCL-CDR model
- `main.py`: use the dataset to run MMCL-CDR

#### `data/` directory.  

- `resistant.npy`: cancer cell lines and drug resistance.
- `sensitive.npy`: cancer cell lines and drug sensitivity.
- `pic.npy`: cell line morphological picture features
- `cnv.npy`: cell line copy number variationn features
- `tpm.npy`: cell line gene expression features
- `drug_fea.npy`: drug feature matrix
- `dict.npy`: drug adjacency matrix 
