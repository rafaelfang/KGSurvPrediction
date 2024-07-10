# KGSurvPrediction

## Integrating Knowledge Graphs into Machine Learning Models for Survival Prediction and Biomarker Discovery in Patients with Non–Small-Cell Lung Cancer 

## Description

Survival prediction is a critical aspect of clinical study design and biomarker discovery. It is a
highly complex task, given the large number of “omics” and clinical features, as well as the high
degrees of freedom that drive patient survival. Prior knowledge can play a critical role in
uncovering the complexity of a disease and understanding the driving factors affecting a
patient’s survival. We introduce a methodology for incorporating prior knowledge into machine
learning–based models for prediction of patient survival through knowledge graphs,
demonstrating the advantage of such an approach for patients with non–small-cell lung cancer.
Using data from patients treated with immuno-oncologic therapies in the POPLAR
(NCT01903993) and OAK (NCT02008227) clinical trials, we found that the use of knowledge
graphs yielded significantly improved hazard ratios, including in the POPLAR cohort, for
models based on biomarker tumor mutation burden compared with those based on knowledge
graphs. Use of a model-defined mutational 10-gene signature led to significant overall survival
differentiation for both trials. We provide parameterized code for incorporating knowledge
graphs into survival analyses for use by the wider scientific community.


## Jupyter Notebook explanation

- `RobustTestUsingMSKDataset_PrepareEmbedding.ipynb`: to generate patient embedding representation for both training and test.

- `RobustTestUsingMSKDataset_RunSurvival.ipynb`: to perform survival prediction using the patient emebedding. Compare results to other approaches such as TMB, model trained using panel genes. Also some other analysis for identify biomarkers and understand model predictions.

This applies to other notebooks but using different datasets: OAK and POPLAR

## The following prerequisites are needed:

1. Installing and activating the environment:
```
conda env create -f environment.yml
conda activate graph_surv_analysis
```
