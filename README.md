# LEO Satellite Routing — Teaching a Satellite to Think Locally

Most routing algorithms treat satellites like they're omniscient.
They're not.

In real LEO constellations, a satellite knows its own queue state, its neighbors, and where it's headed. That's it. No global map. No real-time updates from 600 other nodes. Sharing queue states across the whole network every second would cost more in signaling than it saves in routing efficiency.

So the question this project asks is a practical one: **can a small classifier — trained offline, running locally — make routing decisions as good as a centralized oracle that sees everything?**

It turns out: mostly yes.

---

## What I built

A teacher-student pipeline for congestion-aware routing on a OneWeb-like LEO constellation.

**The teacher** is a centralized Dijkstra algorithm that has full visibility of every queue in the network. It picks the best next hop at every timestep using both propagation delay and queue state. It's too expensive to run in production — but it's an excellent label generator.

**The student** is a Random Forest classifier. It learns to copy the teacher's decisions using only what a real satellite can observe: its own queue, its neighbors' queues, and the direction to the destination. At runtime, no inter-satellite queue exchange happens. Each satellite scores its neighbors and picks the best one locally, in microseconds.

---

## The setup

- **Topology:** 90-node connected subgraph of a 648-satellite OneWeb-like constellation, 574 inter-satellite links
- **Traffic:** Diurnal Poisson load, region-skewed toward European peak (λ ≈ 50 pkt/s), designed to create realistic congestion hotspots
- **Evaluation:** Three policies compared on the same source-destination pairs — DistanceOnly (geometric baseline), DelayAware (the oracle), and the ML agent

---

## What the numbers look like

| | ML agent | Oracle (DelayAware) | Baseline (DistanceOnly) |
|---|---|---|---|
| Median end-to-end delay | 1.26 s | 1.23 s | 8.09 s |
| Path drop rate | 0% | 0% | ~6.95% |
| Normalized throughput | 1.24× baseline | 1.66× baseline | 1.00× |

The agent recovers about 99.7% of the oracle's delay improvement over the baseline — using only local information.

The honest part: under a stricter held-out time test (train on early snapshots, test on peak-hour snapshots the model never saw), offline classification accuracy drops from 0.81 to 0.41. The end-to-end routing performance stays usable, but this is a real limitation of static-snapshot training. Periodic retraining would be needed in any real deployment.

---

## How the features work

Each routing decision expands into one row per candidate neighbor. Four feature classes per row:

- **Link propagation** — geometric distance divided by speed of light. Fast to compute, already in the edge list.
- **Neighbor queue proxy** — Q(t)/µ at the candidate node. The single most important feature. Removing it pushes the agent back toward distance-only behavior.
- **Congestion flag** — binary: is this neighbor above the 90th-percentile queue threshold? Catches transient bursts that the continuous ratio smooths over.
- **Geometric progress** — how much closer to the destination does this hop get us? Acts as a tiebreaker when queue states are similar.

The ablation results match this intuition exactly: queue proxy > congestion flag > geometric progress > propagation alone.

---

## Why Random Forest and not something deeper

Three reasons:

1. No GPU needed. Inference runs in tens of microseconds on a commodity CPU. A satellite payload doesn't have a V100.
2. Tabular features with mixed scales work well with tree splits — no normalization required.
3. The out-of-bag score (0.944) and Y-randomization control (accuracy drops from 0.81 to 0.16 when labels are shuffled, Wilcoxon p = 0.002, Cliff's δ = 1.0) give honest, built-in validation without a separate held-out set.

DRL and GNN alternatives are in the future work list — but the point here was to see how far a lightweight model gets under real operational constraints.

---

## Reproducibility

The pipeline runs in 11 phases, from raw constellation data to paired comparison logs. Everything is seeded and checksummed. The key artifacts:

```
satellite_link_matrix_cleaned.csv   — cleaned topology
teacher_paths.json                  — oracle labels
ml_candidate_labels.csv             — feature matrix (3,145 rows × 11 features)
rf_classifier_model.joblib          — trained model (~few MB)
final_routing_comparison_results_v2.csv — paired KPI logs
```

Ten random seeds, BCa bootstrap confidence intervals on every reported number, Holm-Bonferroni correction across the three KPIs.

---

## Project structure

```
data/          raw and cleaned constellation data
features/      feature extraction pipeline
models/        trained classifier and validation artifacts
evaluation/    routing comparison logs and statistical tests
notebooks/     step-by-step walkthrough
```

---

## Stack

Python 3.10 · NetworkX · scikit-learn · NumPy · pandas

---

*Part of MSc thesis work at University of Rome Tor Vergata, with Erasmus exchange at University of Göttingen. Supervisor: Prof. Ernestina Cianca.*
