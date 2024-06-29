
# Stellar Association Recognition System (STARS) 
---
- [Stellar Association Recognition System (STARS)](#stellar-association-recognition-system-stars)
  - [Description](#description)
  - [What to Expect](#what-to-expect)
  - [Deployment](#deployment)
- [STARS Repository](#stars-repository)

---



Understanding stellar structures is crucial for gaining insights into the evolution of the universe, dark matter, and stellar evolution, among other things.

However, detecting these structures and efficiently determining which stars genuinely belong to them and which appear to do so owing to perspective effects or data defects is a complicated task. Currently, this work is conducted manually with only the help of some basic statistical tools, which are entirely insufficient. As a consequence of this approach, erroneous conclusions often arise in scientific publications due to the inability to accurately determine the structures under study. STARS was created to address this problem.

## Description

STARS (Stellar Association Recognition System) is a tool for detecting stellar associations based on an unsupervised clustering model on cross-validation with a supervised classification model.

- Initially, an HDBSCAN clustering model is employed to identify structures and detect noise.
- Subsequently, the clustering results are validated using a Random Forest classification model.

Additionally, Bayesian optimization is performed for HDBSCAN, where the objective function depends on the quality and separation of the identified clusters and the consistency between the results of both models.

## What to Expect

STARS can perform the following:

1. Efficiently find stellar associations in datasets where it is initially unknown whether any stellar structure exists.
2. Discard stars that do not belong to any cluster and, thus, whose apparent membership in the cluster is a perspective effect.

## Deployment

Given the complexity of the problem and the high dimensionality that STARS can handle, the model execution is performed in batch mode. However, a Streamlit interface has been deployed for users to input their data and receive keys to download their results once they are obtained.

# STARS Repository

Because some of the necessary files are too large to be hosted on GitHub, the repository is currently incomplete.

Before running, you will need to:

1. Fetch the repository.
2. Complete your local repository with the files that you can find at the following location: 
   > [STARS heavy files](https://drive.google.com/drive/u/0/folders/1ac8yV0D-KNRFOH3jnlYUnNfeU6bkZc8S)
   
   1. Download and place the `Models` folder in the root directory of the repository.
   2. Download and place the `Samples` folder in the root directory of the repository.

Now your repository should works properly.
