# LDA From Scratch (CS315)

Linear Discriminant Analysis (LDA) implemented from scratch in Python for dimensionality reduction.

This project follows the CS315 convention:
- **features on rows**
- **observations on columns**

So the core data matrix is:
- `X.shape == (d, N)` where `d` is number of features and `N` is number of observations.

## 1. Problem Setup

Given:
- observations $x_n \in \mathbb{R}^d$, $n = 1, \dots, N$
- class labels $t_n \in \{1, \dots, k\}$

Define:
- class $C_j$
- class size $N_j = |\{n : t_n = j\}|$
- class proportion $P_j = N_j / N$

### Means

Overall mean:
$$
m = \frac{1}{N}\sum_{n=1}^N x_n
$$

Class mean:
$$
m_j = \frac{1}{N_j}\sum_{n:t_n=j} x_n
$$

Weighted relationship:
$$
m = \sum_{j=1}^k P_j m_j
$$

## 2. Scatter Matrices (Unnormalized)

This implementation uses **unnormalized scatter matrices**.

Total scatter:
$$
S_T = \sum_{n=1}^N (x_n - m)(x_n - m)^\top
$$

Within-class scatter:
$$
S_W = \sum_{j=1}^k \sum_{n:t_n=j}(x_n - m_j)(x_n - m_j)^\top
$$

Between-class scatter:
$$
S_B = \sum_{j=1}^k N_j(m_j - m)(m_j - m)^\top
$$

Identity:
$$
S_T = S_W + S_B
$$

## 3. Objective (Fisher Criterion)

Project to lower dimension with:
$$
y_n = W^\top x_n,\quad W \in \mathbb{R}^{d \times d'},\ d' < d
$$

Projected scatters:
$$
S_W^{(y)} = W^\top S_W W,\quad S_B^{(y)} = W^\top S_B W
$$

LDA maximizes between-class spread relative to within-class spread.

For one axis:
$$
\max_w \frac{w^\top S_B w}{w^\top S_W w}
\Rightarrow
S_B w = \lambda S_W w
$$

So we solve a generalized eigenvalue problem.

## 4. Structural Limit

$$
\text{rank}(S_B) \le k-1
$$

Therefore:
$$
d' \le \min(d,\ k-1)
$$

This limit is enforced in the implementation (`n_components` validation).

## 5. Handling Singular $S_W$: Whitening Flow Used Here

Instead of directly forming $S_W^{-1}$, this project uses whitening:

1. Eigendecompose $S_W = Q\Lambda Q^\top$
2. Drop near-zero eigenvalues (numerical rank filtering)
3. Build whitening transform $P = \Lambda_r^{-1/2}Q_r^\top$
4. Whiten between-class scatter:
   $$
   S_B^{(\text{white})} = P S_B P^\top
   $$
5. Eigendecompose $S_B^{(\text{white})}$, keep top `n_components`
6. Map back to original feature space:
   $$
   W = P^\top V_k
   $$
7. Transform centered data:
   $$
   Z = W^\top (X - m\mathbf{1}^\top)
   $$

Geometric interpretation:
- whiten within-class structure
- then do PCA-like decomposition on between-class structure

## 6. Repository Structure

- `tiaan_lda/lda.py`  
  Main LDA implementation (fit, transform, fit_transform + internal math helpers).

- `tiaan_lda/lda_utils.py`  
  Utility helpers:
  - class means / scatter matrix helpers
  - normalized scatter variants for sklearn comparisons
  - plotting before and after LDA

- `tiaan_lda/test_lda.py`  
  Simple runnable example on Iris using the CS315 matrix orientation.

- `comparison/LDA_comparison.ipynb`  
  Scratch-vs-sklearn validation notebook (`solver='eigen'`).

- `data/iris.csv`  
  Dataset used in examples/comparisons.


## 7. Scratch vs sklearn Notes

When comparing with `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`:

- Use `solver='eigen'` for closest formulation match.
- Expect potential differences in:
  - component sign (eigenvector sign ambiguity)
  - projection scale/offset (normalization and transform conventions)
- Expect strong agreement in:
  - explained variance ratios
  - class-separation geometry
  - class means / $S_B$ consistency

The comparison notebook includes checks for all of the above.

## 8. Practical Notes

- This project is **dimensionality reduction only** (not classifier training).
- Max useful LDA dimensions is at most `k - 1`.
- Reconstruction to original feature space is generally not meaningful for LDA.
