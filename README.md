# SpinGlass-RL: Graph Neural Reinforcement Learning for Spin Glass Ground States

**A bridge between many-body physics and NP-hard combinatorial optimization.**
This project explores the ground state of disordered spin systems (±J Ising spin glass) through **reinforcement learning with graph neural networks (GNNs)**, formulated as a **Max-Cut optimization problem**.

---

## Motivation

* Spin glass physics provides a natural testbed for **complex energy landscapes**.
* The ground state problem is equivalent to **NP-hard Max-Cut optimization**.
* We develop a **physics-aware RL framework** with GNN policies to learn scalable heuristics.
* Benchmarked against **Simulated Annealing (SA)**, **Integer Linear Programming (ILP)**, and optionally **QAOA** (quantum-inspired).

---

## Methodology

* **Formulation:** Ising Hamiltonian as cut cost.
* **RL Environment:** Sequential spin assignment with reward shaping \$r\_t = -\Delta H\$.
* **Policy:** Graph Neural Network (GCN/GIN) with actor–critic (A2C).
* **Training:** Curriculum from small to large graphs, entropy regularization, early stopping.
* **Baselines:** SA (heuristic), ILP (exact for small graphs), QAOA (optional).
* **Evaluation:** Relative energy gap, scaling with graph size, and cross-topology transfer.

---

## Key Results

* **RL vs SA:** RL matches or outperforms SA on medium graphs.
* **RL vs ILP:** Relative gap < 5–10% for graphs up to \$n=128\$.
* **Scaling:** RL generalizes across graph sizes and topologies (ER, grid, 3-regular).
* **Post-processing:** Simple hill-climb refinement improves RL energy further.

Poster-ready figures are included in [`figs/`](./figs).

---

## 📂 Repository Structure

```
.
├── documentation-code/       # Source code and implementations
├── documentation-statistics.md  # Detailed statistical results
├── project.ipynb             # Main Jupyter Notebook (step-by-step pipeline)
├── src/                      # Modular code (envs, models, baselines, utils)
├── results/                  # Checkpoints, evaluation stats, CSV summaries
├── figs/                     # Training/evaluation plots (poster-ready)
├── paperwork.pdf             # draft paper from this project
├── requirements.txt          # Python dependencies
├── LICENSE
└── README.md
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/<your-username>/SpinGlass-RL.git
cd SpinGlass-RL
pip install -r requirements.txt
```

### 2. Run Training

```bash
python src/train.py --config configs/default.yaml
```

### 3. Evaluate RL vs Baselines

```bash
python src/eval.py --checkpoint results/best_model.pt
```

### 4. Reproduce Figures

```bash
python src/plot_results.py --input results/summary.csv --out figs/
```

---

## Documentation

* **Mathematical Formulation:** see [`documentation-statistics.md`](./documentation-statistics.md)
* **Code details:** see [`documentation-code/`](./documentation-code)
* **Notebook walkthrough:** [`project.ipynb`](./project.ipynb)

---

## Baselines

* **Simulated Annealing (SA)**
* **Integer Linear Programming (ILP)** (via OR-Tools, small graphs only)
* **QAOA (p=1)** *(optional, requires Qiskit/PennyLane)*

---

## Evaluation Metrics

* **Relative gap** to ILP optimum.
* **Energy per edge** scaling with \$n\$.
* **Cross-topology transfer** (ER → Grid, 3-regular).

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

## Citation

If you use this repository for academic purposes, please cite it using the `CITATION.cff` file.

---
