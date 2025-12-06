# 3D GNN for Antibody–Antigen ΔΔG Prediction

The accurate prediction of changes in binding free energy ($\Delta\Delta G$) upon point mutation is a cornerstone of computational antibody engineering and affinity maturation. While structural Deep Learning has gained prominence, distinguishing the inductive bias provided by 3D geometry from signal inherent in sequence and physicochemical properties remains a challenge.This repository implements a rigorous comparative framework to evaluate the predictive performance of 3D Graph Neural Networks (GNNs) against robust linear and sequence-based baselines. The primary objective is to quantify the marginal gain of modeling the antibody–antigen interface as a graph, focusing on the AB-Bind dataset and extending to SKEMPI 2.0 for generalizability.

Given a wild-type antibody–antigen complex with a resolved 3D structure (PDB) and a specific single-point mutation (defined by chain, residue index, wild-type amino acid $\to$ mutant amino acid). The objective is to predict the scalar change in binding free energy:
$$\Delta\Delta G = \Delta G_{\text{mutant}} - \Delta G_{\text{wild-type}}$$

A negative $\Delta\Delta G$ typically indicates improved affinity (stabilization), while a positive value indicates destabilization.

## Methodology: A Hierarchical Modeling Approach
To ensure that performance gains are attributable to structural reasoning rather than data leakage or simple residue propensities, this project employs a two-stage modeling strategy.

### The Linear and Sequence-Based Baseline
Before implementing geometric deep learning, we establish a "lower bound" of performance using interpretable linear models (Linear/Logistic Regression) and tree-based ensembles (Random Forest/XGBoost).
Rationale:
- Leakage Detection: Historical benchmarks on datasets like SKEMPI have frequently suffered from data leakage, where models memorize complex-specific biases rather than learning biophysical rules1.High performance by a linear model often indicates improper train/test splits (e.g., random splitting rather than complex-level splitting).
- Signal Quantification: Recent studies suggest that sequence-only and simple statistical features can achieve significant correlations in $\Delta\Delta G$ prediction tasks2. This baseline quantifies how much variance in the AB-Bind dataset can be explained by residue identity and physicochemical descriptors alone, without explicit 3D coordinates.
- Interpretability: Linear weights provide immediate sanity checks regarding physicochemical intuition (e.g., penalties for burying hydrophilic residues or introducing steric clashes via volume changes).Feature Engineering:Wild-type and Mutant amino acid identities (One-Hot).Physicochemical property shifts (Volume, Hydrophobicity, Charge, Polarity).Interface vs. Non-interface positioning flags.

### Feature Engineering:

- Wild-type and Mutant amino acid identities (One-Hot).
- Physicochemical property shifts (Volume, Hydrophobicity, Charge, Polarity).
- Interface vs. Non-interface positioning flags.

### 3D Graph Neural Network (GNN) on the Interface

Upon validation of the dataset splits and baselines, we implement a geometric deep learning architecture.
Graph Construction:Following the philosophy of frameworks such as DeepRank-GNN3, the protein interface is transformed into a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
- Nodes ($\mathcal{V}$): Interface residues (defined by a distance cutoff, typically 8-10 Å) from both the Antibody and Antigen. Features include amino acid type, chain logic (Ab/Ag), and atomic coordinates.
- Edges ($\mathcal{E}$): Spatial neighbors within the defined cutoff. Edges are annotated with Euclidean distances and categorical flags for intra-chain vs. inter-chain interactions.

  Architecture:
  - A message-passing GNN (e.g., Graph Convolutional Networks or Graph Attention Networks) propagates features across the interface graph.
  - The architecture specifically encodes the mutation site, allowing the network to learn the localized perturbation in the structural environment.
  - The readout layer pools node representations to regress the scalar $\Delta\Delta G$.
 
## Datasets and Curation

AB-Bind
- Source: Sirin et al., AB-Bind: Antibody binding mutational database for computational affinity predictions
- Composition: 1,101 mutants across 32 unique antibody–antigen complexes with experimentally determined $\Delta\Delta G$ values.
- Processing: Raw data is sourced via submodule from data/external/AB-Bind-Database. Processed data (data/processed/ab_bind_with_labels.csv) includes PDB IDs, chain mappings, normalized mutation strings, and discrete labels (Improved/Neutral/Worsened) for classification tasks.

SKEMPI 2.0
- Source: Jankauskaite et al., SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy
- Composition: 7,085 mutants covering a diverse range of general protein–protein interactions (PPIs).
- Utility: Used to assess the transferability of features learned on antibody interfaces to general PPIs.

  ## References

1. Geng, C., et al. (2019). ISPRED4: interaction sites PREDiction in protein structures with a refinement strategy. Bioinformatics. (Context: Evaluation of leakage in PPI datasets).
2. Dehghanpoor, R., et al. (2018). ProAffiMuSeq: sequence-based prediction of protein-protein binding affinity change upon mutation. Bioinformatics.
3. Réau, M., et al. (2023). DeepRank-GNN: a graph neural network framework to learn patterns in protein–protein interfaces. Bioinformatics.
4. Sirin, S., et al. (2016). AB-Bind: Antibody binding mutational database for computational affinity predictions. Protein Science.
5. Jankauskaite, J., et al. (2019). SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy, kinetics and thermodynamics upon mutation. Bioinformatics.****
