# FLORES+ Dataset for Debiasing Maltese LMs

This dataset leverages the Maltese translations from the open-source FLORES+ machine translation evaluation benchmark. It was utilized in our research to analyze and mitigate gender bias in Maltese Language Models (LMs), including BERTu and mBERTu.

## Dataset Source

FLORES+ is a multilingual machine translation benchmark. The Maltese portions of the dataset were used.
Original repo: 
- [Github] (https://github.com/openlanguagedata/flores?tab=readme-ov-file)

## Purpose

Similar to the Korpus Malti v4.2 subset, the FLORES+ data was selected to provide sentences **unseen** by the BERTu and mBERTu models during their training. This ensures a clean, independent evaluation of the debiasing techniques applied.

## Data Volume

We combined both the `dev` split (997 sentences) and the `devtest` split (1,012 sentences) of the Maltese translation from FLORES+. This resulted in a total of **2,009 Maltese sentences** from FLORES+ used for our data augmentation purposes.

The FLORES+ sentences played a crucial role in two key debiasing strategies: **Counterfactual Data Augmentation (CDA)** and **Dropout Regularization**.


