# 3D GNN for Antibody–Antigen ΔΔG Prediction

Predicting how point mutations change antibody–antigen binding affinity (ΔΔG) is central to affinity maturation and developability. This repo explores how far you can go with **simple sequence/linear models** and what extra signal you actually gain from a **3D graph neural network over the interface**.

## 1. Problem

Given:

- A **wild-type antibody–antigen complex** with a known 3D structure (PDB).
- A **mutation** (chain, residue index, WT AA → mutant AA).

Predict the **change in binding free energy**: We train and evaluate primarily on the **AB-Bind** database (1101 mutants across 32 antibody–antigen complexes) and optionally extend to SKEMPI 2.0 (7085 mutants on general PPIs).

## 2. Why two stages: linear baseline → 3D GNN?

### Stage 1 – Linear / sequence-level baseline

Before touching 3D structure, we train simple models (linear regression / logistic regression / tree-based) on **sequence- and mutation-level features**:

- WT and mutant amino acids, position, chain, interface flag, simple physicochemical descriptors.
- Optionally per-complex intercepts.

Why bother?

1. **Debugging & leakage check**  
   If a linear model with these features already gets suspiciously high performance on ΔΔG, the problem is probably in the **splits or features**, not in the architecture. This is exactly what has happened historically on SKEMPI-style benchmarks when complex-level leakage wasn’t controlled. :contentReference[oaicite:1]{index=1}  

2. **Quantify the “easy” signal**  
   Sequence-only and simple statistical features can already reach decent correlations on ΔΔG prediction.:contentReference[oaicite:2]{index=2}  
   The baseline tells us **how much of AB-Bind can be explained without any 3D geometry**.

3. **Interpretability**  
   Linear weights over amino-acid changes and positions give direct sanity checks:  
   “does mutating buried hydrophobics to charged residues look strongly destabilizing?” etc.

If the linear/sequence model already explains most of the variance, then a fancy 3D GNN is probably overkill or mis-specified; if it hits a clear ceiling, that’s the justification for stage 2.

### Stage 2 – 3D Graph Neural Network on the interface

Once the dataset, features and **complex-level train/val/test splits** are trustworthy, we move to a **3D GNN**:

- Build **residue-level graphs** from PDBs:
  - Nodes = interface residues (Ab + Ag) with AA type, chain/role, coordinates, basic physicochemical features.
  - Edges = spatial neighbors within a cutoff, annotated with distances and intra- vs inter-chain flags.
- For each mutation, mark the mutated residue node and encode WT/MT identity.
- Run a message-passing GNN (GraphConv / GAT-style) over the interface graph, then pool to predict ΔΔG.

This follows the same philosophy as frameworks like **DeepRank-GNN**, which convert protein–protein interfaces into graphs and train GNNs to learn interaction patterns. :contentReference[oaicite:3]{index=3}  

## 3. Data

### AB-Bind

- Source: Sirin et al., *AB-Bind: Antibody binding mutational database for computational affinity predictions*. :contentReference[oaicite:4]{index=4}  
- 1101 mutants across 32 antibody–antigen complexes with experimental ΔΔG.  
- Repo: `data/external/AB-Bind-Database/AB-Bind_experimental_data.csv` (added as git submodule).

We provide:

- `data/processed/ab_bind_with_labels.csv` – cleaned version with:
  - PDB ID, chain partners, mutation string, ΔΔG in kcal/mol.
  - Discrete labels (improved / neutral / worsened) for classification / ranking.

### SKEMPI 2.0 (optional extension)

- Source: Jankauskaite et al., *SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy…*. :contentReference[oaicite:5]{index=5}  
- 7085 mutants on diverse PPIs, with affinities and ΔΔG.  
- Downloaded into `data/raw/skempi2/` via a small script.

SKEMPI is mainly used to check whether patterns learned on antibody–antigen interfaces transfer to more general protein–protein interfaces.

## 4. Experiments
### EDA & dataset curation

- Inspect ΔΔG distribution, class imbalance, per-complex heterogeneity.
- Define complex_id and create complex-level train/val/test splits.
- Sequence / linear baselines
- Feature engineering at mutation + sequence level.

### Linear regression / logistic regression / tree-based models.
- Metrics: Pearson/Spearman, MAE, ROC-AUC / PR-AUC for improved vs others, within-complex ranking.

### 3D GNN

- Build and cache interface graphs from PDBs.
- Implement message-passing GNN over mutation-annotated graphs.
- Same metrics and splits as baselines.

### Analysis

- Compare GNN vs baseline under strict splits.
- Case studies on specific complexes (e.g. 1T83, 1MHP) to inspect where 3D geometry helps or fails.
- Ablations (remove 3D features, use sequence embeddings only, etc.).

## 5. References

1. Sirin et al., AB-Bind: Antibody binding mutational database for computational affinity predictions, Protein Sci 2016. 
2. Jankauskaite et al., SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy, kinetics and thermodynamics upon mutation, Bioinformatics 2019.
3. Réau et al., DeepRank-GNN: a graph neural network framework to learn patterns in protein–protein interfaces, Bioinformatics 2023.
4. Huang et al., SSIPe: accurately estimating protein–protein binding affinity change upon mutation, Bioinformatics 2020 (example of structure-based ΔΔG prediction).
5. ProAffiMuSeq: sequence-based prediction of protein-protein binding affinity change upon mutation (example of strong sequence-only baseline).

