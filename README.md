# GCN_SCZ_Classification

This repository provides core code and toolboxes for analysis in the paper entitled "Graph convolutional networks reveal network-level functional dysconnectivity in schizophrenia" by Lei et al. Please check the paper for the latest description of data analysis.

# Overview
Content includes demo data and source code for the implementation of graph convolutional network (GCN), linear support vector machine (SVM) and non-linear SVM with radial basis function (RBF) kernel on a large multi-site schizophrenia fMRI dataset. The site effects are eliminated by using ComBat Harmonization. The Codes for ComBat harmonization methods is supported by [NeuroComBat-sklearn](https://github.com/Warvito/neurocombat_sklearn).

# Requirements
- Python (>= 3.5)
- Scikit-Learn
- NeuroComBat-sklearn
- Pytorch
- Pytorch-geometric
- Scipy
- Numpy
- Pandas

# Toolboxes

All other toolboxes and codes used in our study for image preprocessing, ancillary analysis and visualization are shown below:

- [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
- [GraphSaliencyMap](https://github.com/sarslancs/graph_saliency_maps)
- [GRETNA](https://www.nitrc.org/projects/gretna/)
- [BrainNetViewer](https://www.nitrc.org/projects/bnv/)
