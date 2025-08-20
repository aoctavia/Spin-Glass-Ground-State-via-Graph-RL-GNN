## Statistical Formulation & Equations

### 1. Spin Glass Hamiltonian
$$
H(\mathbf{s}) = - \sum_{(i,j)\in E} J_{ij}\, s_i s_j,
\quad s_i \in \{-1,+1\}
$$

---

### 2. Max-Cut Formulation
$$
\max_{x \in \{0,1\}^n} \; C(x) = \sum_{(i,j)\in E} J_{ij}\, \mathbb{1}\!\left[x_i \neq x_j\right]
$$
with the mapping $s_i = 2x_i - 1$.

---

### 3. Reward Shaping (RL Environment)
$$
r_t \;=\; -\Delta H_t \;=\; -\Big(H(\mathbf{s}_{1:t}) - H(\mathbf{s}_{1:t-1})\Big)
$$

---

### 4. Policy Gradient Loss
$$
\mathcal{L}_\pi \;=\; -\,\mathbb{E}_{\pi_\theta}\!\left[ \log \pi_\theta(a_t \mid s_t)\; A_t \right],
\qquad
A_t \;=\; R_t - V_\phi(s_t)
$$

---

### 5. Value Loss (Critic)
$$
\mathcal{L}_V \;=\; \tfrac12\,\Big(R_t - V_\phi(s_t)\Big)^2
$$

---

### 6. Entropy Regularization
$$
\mathcal{L}_{\mathrm{ent}} \;=\; -\,\beta \sum_a \pi_\theta(a \mid s_t)\,\log \pi_\theta(a \mid s_t)
$$

---

### 7. Total Actor–Critic Loss
$$
\mathcal{L} \;=\; \mathcal{L}_\pi \;+\; c_v\,\mathcal{L}_V \;+\; c_e\,\mathcal{L}_{\mathrm{ent}}
$$

---

### 8. Relative Gap (Evaluation)
$$
\mathrm{Gap} \;=\; \frac{E_{\mathrm{RL}} - E_{\mathrm{opt}}}{\lvert E_{\mathrm{opt}} \rvert}
$$

---

### 9. Energy per Edge (Scaling Analysis)
$$
\epsilon(n) \;=\; \frac{1}{\lvert E \rvert}\;\mathbb{E}\!\big[H(\mathbf{s})\big]
$$

---

## List of Symbols

### Graph & Physics
- $G=(V,E)$ : graph with nodes $V$ and edges $E$  
- $n=\lvert V\rvert$ : number of nodes  
- $\lvert E\rvert$ : number of edges  
- $s_i\in\{-1,+1\}$ : spin at node $i$  
- $J_{ij}\in\{-1,+1\}$ : coupling between $i$ and $j$  
- $H(\mathbf{s})$ : Hamiltonian energy  

### Optimization
- $x_i\in\{0,1\}$ : Max‑Cut variable  
- $C(x)$ : cut value  
- $\mathbb{1}[x_i\neq x_j]$ : indicator for edge $(i,j)$ crossing the cut  

### Reinforcement Learning
- $a_t$ : action; $s_t$ : state; $r_t$ : reward; $R_t$ : return  
- $\pi_\theta(a\mid s)$ : policy (actor)  
- $V_\phi(s)$ : value function (critic)  
- $A_t=R_t - V_\phi(s_t)$ : advantage  
- $\beta$ : entropy weight; $c_v,c_e$ : loss weights  

### Evaluation & Statistics
- $E_{\mathrm{RL}}$ : energy from RL policy  
- $E_{\mathrm{opt}}$ : optimal energy (ILP/exact)  
- $\mathrm{Gap}$ : relative gap  
- $\epsilon(n)$ : avg. energy per edge for size $n$
