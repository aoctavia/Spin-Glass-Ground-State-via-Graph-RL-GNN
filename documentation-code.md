# Summary

---

### Cell 1 — Setup & Installs

* **Content:** Install `numpy/scipy/matplotlib/seaborn/networkx/tqdm/einops/ortools` + PyTorch Geometric (PyG) via wheels.
* **Purpose:** Provide all dependencies for graph simulation, baselines (ILP/SA), and GNN models.

---

### Cell 2 — Imports, Config, Reproducibility

* **Content:** Import libraries, set random seed, define `DEVICE`, and central config dictionary `CFG`.
* **Purpose:** Ensure experiments are **reproducible** with centralized parameters (graph size, epochs, learning rate, etc.).

---

### Cell 3 — Graph Generators (ER, Grid, 3-Regular) + Couplings

* **Content:** Functions to generate **Erdős–Rényi**, **grid** \$L \times L\$, and **k-regular** graphs, with **spin-glass couplings** \$J \in {-1,+1}\$.
* **Purpose:** Build diverse problem instances (graph + couplings) for training and evaluation.

---

### Cell 4 — PyG Data & Energy

* **Content:** Convert graph → `torch_geometric.data.Data` (node features + `edge_weight=J`), define energy and \$\Delta\$energy functions.
* **Purpose:** Provide a representation compatible with **GNNs** and the physical objective (Ising Hamiltonian).

---

### Cell 5 — EDA: Topology & Couplings

* **Content:** Plot degree histogram, \$J\$ distribution, and visualize topologies.
* **Purpose:** **Explore data** to understand the characteristics of graphs used for training.

---

### Cell 6 — Baseline: Simulated Annealing (SA)

* **Content:** Implement SA with geometric cooling and quick evaluation.
* **Purpose:** A strong **heuristic baseline** for comparison against RL+GNN.

---

### Cell 7 — Baseline: Exact ILP (Max-Cut Mapping)

* **Content:** Formulate **CP-SAT** OR-Tools for Max-Cut (mapped from spin-glass energy), return optimal energy.
* **Purpose:** Provide the **true optimum** (for small graphs) to measure the **gap** between RL and optimal solutions.

---

### Cell 8 — RL Environment (Sequential Assignment)

* **Content:** Environment for sequential spin assignment with **reward shaping** \$r\_t = -\Delta H\$.
* **Purpose:** Cast ground-state search as an **episodic RL** problem with stable local rewards.

---

### Cell 9 — Policy–Value GNN (A2C Style)

* **Content:** GCN backbone with `edge_weight=J`, plus **policy head** (logits for −1/+1) and **value head** (\$V(s)\$).
* **Purpose:** Actor–critic architecture that encodes graph structure and spin couplings.

---

### Cell 10 — Instance Generator & Permutation Augmentation

* **Content:** Utility to generate graph instances + augmentation by **node permutation**.
* **Purpose:** Improve **generalization** and **invariance** to node indexing.

---

### Cell 11 — A2C Training Loop (Actor–Critic)

* **Content:** Simple A2C training (policy loss + value loss + entropy reg + gradient clipping) with **curriculum sizes**.
* **Purpose:** Train GNN policies to **minimize energy** stably without collapse.

---

### Cell 12 — Training Curves

* **Content:** Plot **mean energy** and **mean return** per epoch.
* **Purpose:** Track training progress and detect stagnation/overfitting.

---

### Cell 13 — Greedy Inference (RL Policy)

* **Content:** Evaluate policy using **greedy actions**; compare with SA on a single instance.
* **Purpose:** Assess deterministic performance of the RL policy at inference time.

---

### Cell 14 — Batch Evaluation (+ILP for Small Graphs)

* **Content:** Batch evaluation: **RL vs SA**; for small graphs, include **ILP** and compute **relative gap**.
* **Purpose:** Measure average performance and how close RL is to the optimum.

---

### Cell 15 — Visualization of Evaluation Results

* **Content:** Boxplots **RL vs SA** (small & large graphs), histogram of **RL–ILP gaps**.
* **Purpose:** Provide **comparative analysis** that is presentation-ready.

---

### Cell 16 — Save Artifacts

* **Content:** Save model `state_dict`, `train_history.json`, and evaluation stats (`.npz`).
* **Purpose:** Ensure **reproducibility** and facilitate sharing in repos/portfolios.

---

### Cell 17 — (Optional) Replace Backbone with GIN

* **Content:** Implementation of **PolicyValueGIN** (more expressive than GCN).
* **Purpose:** **Architecture ablation** to study GNN capacity.

---

### Cell 18 — Notes & Next Steps (Markdown)

* **Content:** Ideas for extensions: full A2C, mixed curriculum/topologies, QAOA bridge, etc.
* **Purpose:** Provide a **roadmap for future research**, making the project appear extensible.

---

### Cell 19 — Checkpointing & Utilities

* **Content:** `Timer` class, `save_ckpt`/`load_ckpt` functions, `checkpoints/` & `artifacts/` dirs.
* **Purpose:** **Experiment management** (resume runs, save best model).

---

### Cell 20 — Validation Set & Early Stopping

* **Content:** `VALSET`, `mean_val_energy`, and A2C training with **early stopping**.
* **Purpose:** Prevent **overfitting**, select best model by validation energy.

---

### Cell 21 — Plot Loss Terms & Validation Energy

* **Content:** Curves of **policy/value/entropy losses** and **train vs val energy**.
* **Purpose:** Diagnose **training stability** and regularization trade-offs.

---

### Cell 22 — Local Refinement (Zero-T Hill-Climb)

* **Content:** Apply greedy **1-opt refinement** on RL output.
* **Purpose:** Cheap **post-processing** that often improves energy.

---

### Cell 23 — Cross-Topology Generalization

* **Content:** Evaluate on **Grid** & **3-regular** graphs (RL, RL+refine, SA) + boxplots.
* **Purpose:** Test **transferability** across topologies (robustness indicator).

---

### Cell 24 — Size Scaling Study

* **Content:** Plot **energy per edge vs \$n\$** for RL, RL+refine, and SA.
* **Purpose:** Analyze **scaling behavior** as graph size grows.

---

### Cell 25 — Ablation Tests

* **Content:** Evaluate without \$J\$ weights or without degree features.
* **Purpose:** Assess which **input components** truly matter.

---

### Cell 26 — Mini Hyper-Parameter Sweep

* **Content:** Small sweep (hidden size, entropy coefficient) with early stopping.
* **Purpose:** **Lightweight tuning** for structured performance improvement.

---

### Cell 27 — Save Best Model & CSV Summary

* **Content:** Save **final checkpoint**, `train_history2.json`, and summary CSV of key metrics.
* **Purpose:** Ready for **README/poster** inclusion and cross-run comparison.

---

### Cell 28 — (Optional) QAOA Baseline (p=1)

* **Content:** If `qiskit` is available, compute **QAOA p=1** for small graphs; compare with RL/SA.
* **Purpose:** Provide a **bridge to quantum algorithms**, relevant to combinatorial optimization.

---

### Cell 29 — Quick Report (Markdown)

* **Content:** Auto-generate short report from evaluation results.
* **Purpose:** Facilitate **mini-paper/poster writing** with consistent numbers.

---

### Cell 30 — Export Figures (Poster-Ready)

* **Content:** Save key plots (energy curves, scaling) to `figs/`.
* **Purpose:** **Presentation-ready** results without manual re-plotting.

---

## Project Overview

* **Physics → Optimization:** Solve the spin glass ground state (Ising Hamiltonian) as a **Max-Cut problem**.
* **Physics-aware ML method:** **GNN + RL (A2C)** exploits graph structure and couplings \$J\$.
* **Experimental rigor:** Includes **SA baseline**, **ILP optimum**, **local refinement**, and **generalization tests** (across topology and size).
* **Portfolio readiness:** Artifacts, checkpoints, CSVs, **reports and figures** → easy packaging into **GitHub** and **poster** form.
