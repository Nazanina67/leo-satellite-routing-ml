# Traffic-Aware Routing in LEO Satellite Networks

> Supervised ML system for congestion-aware next-hop routing across a 648-satellite constellation with 11,404 inter-satellite links.

---

## What this project does

Standard shortest-path routing is blind to network state. When a satellite link becomes congested, traffic keeps flowing into it — causing queue buildup and delay propagation across the constellation.

This project replaces the distance heuristic with a supervised ML classifier that reads live network state (queue lengths, traffic rates, neighbor conditions) and selects the optimal next hop in real time.

**Key result:** DelayAware achieves >90% reduction in routing delay vs. the shortest-path baseline, with stable performance under high-load stress scenarios.

---

## System architecture

```
Data layer       Topology · Traffic simulation · Queue capture · ISL delays
      ↓
Feature layer    Local state · Neighbor state · Temporal lag (ρ = 0.99)
      ↓
Training layer   Labeled dataset → Scikit-learn classifier → Cross-validation
      ↓
Inference layer  DelayAware engine — distributed, real-time, congestion-aware
```

---

## Simulation pipeline

| Stage | What it does | Output artifact |
|---|---|---|
| 1 · Constellation setup | Walker orbit, 648 sats, 11,404 ISLs | `topology.json`, adjacency matrix |
| 2 · Traffic injection | Normal load + stress scenarios | `traffic_rates.csv`, `scenario_config.yaml` |
| 3 · Feature extraction | Queue · delay · rate · neighbor state | `features_train.parquet`, `labels.npy` |
| 4 · Model training | Scikit-learn, cross-validation, tuning | `model.pkl`, `cv_scores.json` |
| 5 · Routing evaluation | DelayAware vs shortest-path baseline | `results_normal.csv`, `results_stress.csv` |

Each stage is git-tagged for full reproducibility.

---

## Feature engineering

### Local state features
- Queue length at current node
- Current transmission rate
- Buffer occupancy ratio

### Neighbor state features
- Queue lengths at all adjacent nodes (1-hop)
- Available bandwidth per link

### Temporal lag features
- Propagation delay delta (t vs t-1)
- Queue trend over t-1, t-2 timesteps

> **Key insight:** Traffic generation rate correlates with queue buildup at ρ = 0.99. This makes delay-lag features the strongest congestion predictor in the dataset.

---

## Before / After

| Metric | Shortest-path baseline | DelayAware (ML) |
|---|---|---|
| Avg. routing delay | Baseline | **>90% lower** |
| Congestion avoidance | None — blind to state | Active — reads queue signals |
| Stress-load behavior | Degrades sharply | Stable |
| Decision scope | Geometric distance only | Local + neighbor + temporal |

---

## Tech stack

- **Language:** Python 3.x
- **ML:** Scikit-learn (supervised classification)
- **Simulation:** Custom network simulation environment
- **Data:** Parquet, NumPy, CSV — all versioned
- **Tooling:** Git, Jupyter Notebook, LaTeX

---

## Reproducibility

All experiments are fully reproducible:
- Version-controlled code and configuration
- Structured dataset with documented checkpoints
- Separate evaluation protocols for normal and stress traffic conditions
- Statistical analysis scripts included

---

## Results summary

```
Correlation (traffic generation ↔ queue buildup):  ρ = 0.99
Performance improvement over baseline:             >90%
Constellation size:                                648 satellites
Inter-satellite links modeled:                     11,404
Traffic scenarios evaluated:                       Normal + stress
```

---

## Contact

Nazanin Ansaripour · nazanin.ansaripour@students.uniroma2.eu · [LinkedIn](https://linkedin.com/in/nazanin-ansaripour)
