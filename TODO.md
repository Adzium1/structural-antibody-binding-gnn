## TODO (priorités)

- Mettre `3D-GNN-over-antibody-antigen/reports/gnn_metrics.csv` à jour avec le run GPU propre (InterfaceGNN 128d / 3 couches, lr=1e-3, standardize, dist_weighted) et régénérer `reports/compare_gnn_baselines.png` via `python -m src.analysis.compare_gnn_vs_baselines`.
- Documenter la config exacte du run GPU (hidden_dim, layers, lr, loss, standardisation) dans le README ou un log dédié.
- Enrichir les graphes : charger `data/processed/ab_bind_features.csv` dans `src/data/build_interface_graphs.py` et concaténer les features physico-chimiques/structurales dans `node_features` (et éventuellement `edge_attr`) par `sample_id`; ajuster `InterfaceGNN` si la dimension change.
- Ajouter un script/notebook de suivi des runs GPU dans `notebooks/` (voir `notebooks/README.md`).
- Nettoyer les caractères corrompus dans le README initial (encodage).
- Ajouter un test simple (ex. chargement de `ab_bind_graphs.pkl`, shape assertions sur `node_features`/`edge_index`) pour détecter les régressions.
- Optionnel : déplacer `InterfaceGNN` dans un module dédié (`src/gnn/model.py`) pour clarifier l’API.
