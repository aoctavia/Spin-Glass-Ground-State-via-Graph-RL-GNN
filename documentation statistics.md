## Statistical Formulation & Equations

### 1. Spin Glass Hamiltonian

$$
H(\mathbf{s}) \;=\; - \sum_{(i,j)\in E} J_{ij} \, s_i s_j, 
\quad s_i \in \{-1, +1\}
$$

---

### 2. Max-Cut Formulation

$$
\max_{x \in \{0,1\}^n} 
\; C(x) \;=\; \sum_{(i,j)\in E} J_{ij} \; \mathbf{1}[x_i \neq x_j]
$$

with the mapping \$s\_i = 2x\_i - 1\$.

---

### 3. Reward Shaping (RL Environment)

$$
r_t = -\Delta H_t \;=\; - \Big( H(\mathbf{s}_{1:t}) - H(\mathbf{s}_{1:t-1}) \Big)
$$

---

### 4. Policy Gradient Loss

$$
\mathcal{L}_{\pi} = - \mathbb{E}_{\pi_\theta} 
\left[ \log \pi_\theta(a_t|s_t) \, A_t \right]
$$

with

$$
A_t = R_t - V_\phi(s_t)
$$

---

### 5. Value Loss (Critic)

$$
\mathcal{L}_{V} = \frac{1}{2} \, \big(R_t - V_\phi(s_t)\big)^2
$$

---

### 6. Entropy Regularization

$$
\mathcal{L}_{\text{ent}} = - \beta \, 
\sum_{a} \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)
$$

---

### 7. Total Actor–Critic Loss

$$
\mathcal{L} = \mathcal{L}_{\pi} + c_v \mathcal{L}_{V} + c_e \mathcal{L}_{\text{ent}}
$$

---

### 8. Relative Gap (Evaluation)

$$
\text{Gap} = 
\frac{E_{\text{RL}} - E_{\text{opt}}}{|E_{\text{opt}}|}
$$

---

### 9. Energy per Edge (Scaling Analysis)

$$
\epsilon(n) = \frac{1}{|E|} \, \mathbb{E}\big[H(\mathbf{s})\big]
$$

---

## List of Symbols

### Graph & Physics

* \$G = (V,E)\$ : graph with nodes \$V\$ and edges \$E\$.
* \$n = |V|\$ : number of nodes.
* \$|E|\$ : number of edges.
* \$s\_i \in {-1,+1}\$ : spin variable at node \$i\$.
* \$J\_{ij} \in {-1,+1}\$ : coupling (interaction) between nodes \$i\$ and \$j\$.
* \$H(\mathbf{s})\$ : Hamiltonian energy of configuration \$\mathbf{s}\$.

---

### Optimization

* \$x\_i \in {0,1}\$ : binary variable for the Max-Cut formulation.
* \$C(x)\$ : cut value (number of edges crossing the cut).
* \$\mathbf{1}\[x\_i \neq x\_j]\$ : indicator function that edge \$(i,j)\$ belongs to the cut.

---

### Reinforcement Learning

* \$a\_t\$ : action at time step \$t\$ (spin assignment).
* \$s\_t\$ : state at time step \$t\$.
* \$r\_t\$ : reward at time step \$t\$.
* \$R\_t\$ : return (cumulative reward from step \$t\$ to the end of the episode).
* \$\pi\_\theta(a|s)\$ : policy parameterized by \$\theta\$ (GNN-based actor).
* \$V\_\phi(s)\$ : value function parameterized by \$\phi\$ (critic).
* \$A\_t = R\_t - V\_\phi(s\_t)\$ : advantage.
* \$\beta\$ : entropy regularization coefficient.
* \$c\_v, c\_e\$ : weights for value loss and entropy loss.

---

### Evaluation & Statistics

* \$E\_{\text{RL}}\$ : energy obtained by the RL policy.
* \$E\_{\text{opt}}\$ : optimal energy (from ILP or exact solver).
* \$\text{Gap}\$ : relative gap between RL and the optimum.
* \$\epsilon(n)\$ : average energy per edge for a graph of size \$n\$.

---
