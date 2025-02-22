# CancerQML

This is the repository related to a project of QubiTO (a team of Politecnico di Torino) involving the training of a quantum machine learning model.


### Dataset
The dataset used was obtained from [here](https://www.kaggle.com/datasets/reihanenamdari/breast-cancer). It concerned the correct prediction of breast cancer in various patients.

### Feature importance analysis
We firstly analyzed the relative importance of the features since we had limited power to simulate quantum hardware. We chose to use a range of 8 to 12 qubits, so 8 to 12 features.
We obtained the following:

![Figure 1](./results/feature_importance_transparent.png)

We then produced different datasets.

### Downsampling and PCA
We tried to perform random downsampling to make the classes balanced, obtaining a 1:1 dataset. We also tried exploring the possibility of reducing degrees of freedom applying principal component analysis, and cutting the number of features according to that. We obtained, acting on the downsampled dataset:

![Figure 2](./results/variance_transparent.png)

We so discarded all features but the first 10, retaining a good amount of variance.

### Training the models


Script: https://docs.google.com/document/d/1Ddh6a-vay6XPDfGoNZPHq0MFdjb54wiGrWTzQMZWKMY/edit?tab=t.0

Pres: https://docs.google.com/presentation/d/1cAAfEJD8zmuGvs3vDbFRBj16YBLdosQKMw3aOA1Qc-w/edit#slide=id.g31ecbd86e14_0_0