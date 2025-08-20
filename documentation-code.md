# Summary
---

### Cell 1 — Setup & Installs

* **Isi:** Install `numpy/scipy/matplotlib/seaborn/networkx/tqdm/einops/ortools` + PyTorch Geometric (PyG) via wheels.
* **Tujuan:** Menyediakan semua dependensi untuk simulasi graf, baseline (ILP/SA), dan model GNN.

### Cell 2 — Imports, Config, Reproducibility

* **Isi:** Import library, set seed, definisi `DEVICE`, dan kamus konfigurasi `CFG`.
* **Tujuan:** Menjamin eksperimen **reproducible** dan parameter terpusat (ukuran graf, epoch, lr, dsb).

### Cell 3 — Graph Generators (ER, Grid, 3‑Regular) + Couplings

* **Isi:** Fungsi pembuat graf **Erdős–Rényi**, **grid** L×L, dan **k‑regular**, dengan **coupling spin-glass** $J∈{−1,+1}$.
* **Tujuan:** Membangun instansi masalah (graf + couplings) yang beragam untuk training & evaluasi.

### Cell 4 — PyG Data & Energi

* **Isi:** Konversi graf → `torch_geometric.data.Data` (fitur node + `edge_weight=J`), fungsi energi dan Δenergi.
* **Tujuan:** Representasi data yang kompatibel dengan **GNN** dan fungsi objektif fisika (Hamiltonian Ising).

### Cell 5 — EDA Topologi & Couplings

* **Isi:** Plot histogram derajat, distribusi J, dan visualisasi topologi.
* **Tujuan:** **Eksplorasi data** untuk memahami karakter graf yang akan dipakai model.

### Cell 6 — Baseline: Simulated Annealing (SA)

* **Isi:** Implementasi SA dengan pendinginan geometrik dan evaluasi cepat.
* **Tujuan:** Baseline **heuristik** yang kuat untuk dibandingkan dengan RL+GNN.

### Cell 7 — Baseline: Exact ILP (Max‑Cut Mapping)

* **Isi:** Formulasi **CP‑SAT** OR‑Tools untuk Max‑Cut (mapping dari energi spin-glass), kembalikan energi optimal.
* **Tujuan:** Pembanding **optimal** (untuk graf kecil) agar bisa mengukur **gap** RL terhadap optimum.

### Cell 8 — RL Environment (Sequential Assignment)

* **Isi:** Environment konstruksi spin: assign spin per node, **reward shaping** $r_t = −ΔH$.
* **Tujuan:** Menjadikan ground-state search sebagai **episodic RL** dengan reward lokal yang stabil.

### Cell 9 — Policy‑Value GNN (A2C‑style)

* **Isi:** GCN backbone dengan `edge_weight=J`, **policy head** (logits untuk −1/+1) dan **value head** (V(s)).
* **Tujuan:** Arsitektur **actor‑critic** yang memahami struktur graf & bobot couplings.

### Cell 10 — Instance Generator & Permutation Augmentation

* **Isi:** Utilitas membuat instansi graf sesuai tipe + augmentasi **permute nodes**.
* **Tujuan:** **Generalization** dan **invariance** terhadap pelabelan node (tidak overfit ke index).

### Cell 11 — A2C Training Loop (Actor‑Critic)

* **Isi:** Pelatihan A2C sederhana (policy loss + value loss + entropy reg + grad clip), **curriculum sizes**.
* **Tujuan:** Melatih kebijakan GNN untuk **meminimalkan energi** secara stabil dan tidak collapse.

### Cell 12 — Training Curves

* **Isi:** Plot **mean energy** & **mean return** per epoch.
* **Tujuan:** Memantau kemajuan training dan mendeteksi stagnasi/overfitting.

### Cell 13 — Greedy Inference (RL Policy)

* **Isi:** Evaluasi kebijakan dengan aksi **greedy**; bandingkan dengan SA pada satu instansi.
* **Tujuan:** Lihat kualitas kebijakan RL secara deterministik saat inference.

### Cell 14 — Batch Evaluation (+ILP untuk kecil)

* **Isi:** Evaluasi batch: **RL vs SA**; untuk graf kecil, tambahkan **ILP** dan hitung **relative gap**.
* **Tujuan:** Menilai performa rata‑rata dan **seberapa dekat RL ke optimum**.

### Cell 15 — Visualisasi Hasil Evaluasi

* **Isi:** Boxplot **RL vs SA** (small & large), histogram **gap RL–ILP**.
* **Tujuan:** **Analisis komparatif** yang mudah disampaikan di laporan/poster.

### Cell 16 — Save Artifacts

* **Isi:** Simpan `state_dict` model, `train_history.json`, dan statistik evaluasi (`.npz`).
* **Tujuan:** **Reproducibility** dan kemudahan publikasi ke repo/portofolio.

### Cell 17 — (Optional) Ganti Backbone ke GIN

* **Isi:** Implementasi **PolicyValueGIN** (lebih ekspresif dari GCN).
* **Tujuan:** **Ablasi arsitektur** untuk melihat pengaruh kapasitas GNN.

### Cell 18 — Notes & Next Steps (Markdown)

* **Isi:** Daftar ide peningkatan: A2C penuh, curriculum/topologi campur, QAOA bridge, dsb.
* **Tujuan:** Peta **riset lanjutan** agar proyek tampak berkembang dan visioner.

### Cell 19 — Checkpointing & Utils

* **Isi:** Kelas `Timer`, fungsi `save_ckpt`/`load_ckpt`, direktori `checkpoints/` & `artifacts/`.
* **Tujuan:** **Manajemen eksperimen** (resume, simpan best model).

### Cell 20 — Validasi Set & Early‑Stopping

* **Isi:** `VALSET`, fungsi `mean_val_energy`, dan **training A2C dengan early‑stopping**.
* **Tujuan:** Mencegah **overfitting**, memilih model terbaik berdasar **val energy**.

### Cell 21 — Plot Loss Terms & Val Energy

* **Isi:** Kurva **policy/value/entropy** dan **train vs val energy**.
* **Tujuan:** Diagnostik **stabilitas pelatihan** dan trade‑off regularisasi.

### Cell 22 — Local Refinement (Zero‑T Hill‑Climb)

* **Isi:** Refinement **greedy 1‑opt** pada output RL.
* **Tujuan:** **Post‑processing** murah yang sering memberi penurunan energi ekstra.

### Cell 23 — Cross‑Topology Generalization

* **Isi:** Evaluasi **Grid** & **3‑regular** (RL, RL+refine, SA) + boxplot.
* **Tujuan:** Uji **transfer** antar topologi (indikator robust/general).

### Cell 24 — Size Scaling Study

* **Isi:** Kurva **energy per edge vs n** untuk RL, RL+refine, SA.
* **Tujuan:** Analisis **skaling** performa seiring ukuran graf membesar.

### Cell 25 — Ablation Tests

* **Isi:** Evaluasi tanpa bobot $J$ atau tanpa fitur derajat.
* **Tujuan:** Menilai **komponen penting** pada representasi (apa yang benar‑benar membantu).

### Cell 26 — Mini Hyper‑Parameter Sweep

* **Isi:** Sweep kecil (hidden size, entropy coef) + early‑stopping per konfigurasi.
* **Tujuan:** **Tuning ringan** yang terstruktur untuk peningkatan performa.

### Cell 27 — Save Best Model & CSV Summary

* **Isi:** Simpan **checkpoint final**, `train_history2.json`, dan **ringkasan CSV** metrik inti.
* **Tujuan:** Siap dipakai untuk **README/poster** dan perbandingan antar run.

### Cell 28 — (Opsional) QAOA Baseline (p=1)

* **Isi:** Jika `qiskit` tersedia, hitung **QAOA p=1** untuk graf kecil; bandingkan dengan RL/SA.
* **Tujuan:** **Jembatan ke quantum algorithms**, relevan dengan call (combinatorial + quantum).

### Cell 29 — Quick Report (Markdown)

* **Isi:** Generate laporan singkat otomatis dari hasil evaluasi.
* **Tujuan:** Memudahkan **penulisan mini‑paper/poster** dengan angka yang konsisten.

### Cell 30 — Export Figures (Poster‑Ready)

* **Isi:** Simpan ulang figur kunci (kurva energi, scaling) ke folder `figs/`.
* **Tujuan:** **Siap presentasi** (paper/poster) tanpa perlu re‑plot manual.

---

## Gambaran besar (tujuan proyek)

* **Formulasi fisika → optimisasi:** Menyelesaikan ground state spin glass (Hamiltonian Ising) sebagai problem **Max‑Cut**.
* **Metode ML yang fisika-aware:** **GNN + RL (A2C)** memanfaatkan struktur graf dan couplings $J$.
* **Rigor eksperimen:** Baseline **SA**, pembanding **ILP (optimal)**, **refinement lokal**, dan **uji generalisasi** (topologi, ukuran).
* **Kesiapan portofolio:** Artifacts, checkpoint, CSV, **report** dan **figures** → mudah dipaketkan ke **GitHub** dan **poster**.
