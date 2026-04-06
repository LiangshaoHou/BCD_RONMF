# RONMF: Block Coordinate Descent for Rescaled Orthogonal NMF

## Overview
This repository provides an implementation of a **Block Coordinate Descent (BCD)** algorithm for solving the **Rescaled Orthogonal Nonnegative Matrix Factorization (RONMF)** problem.

The goal is to factorize a nonnegative matrix 
$A \in \mathbb{R}^{n \times m}$ into two matrices $U \in \mathbb{R}^{n \times r}$ 
and $V \in \mathbb{R}^{r \times m}$ such that:
- $U\geq 0$, with unit-norm columns
- $V \ge 0$, with at most one positive entry per column
- $A \approx U V $

This structure is particularly useful for **clustering and representation learning**.
Please refer to our paper for more details:  
> **L. Hou, D. Chu, and L.-Z. Liao.**  
> *A fast block coordinate descent method for orthogonal nonnegative matrix factorization.*  
> *(To appear in SIAM Journal on Matrix Analysis and Applications, SIMAX).*
---

## Dataset

- **ORL_31x32.mat**  
  The ORL face dataset preprocessed into \(32 \times 32\) grayscale images.  
  Each column corresponds to a vectorized face image.

---

## Initialization Methods

The following initialization strategies are provided:

### 1. Random Initialization
- Generates nonnegative random matrices for \( U \) and \( V \)
- Simple baseline method


### 2. `NNDSVD.m`
- Nonnegative Double Singular Value Decomposition initialization
- Provides structured and deterministic initialization
- Improves convergence speed and stability

### 3. `Jiang_Initial.m`
- Initialization method proposed by Jiang et al.
- Designed to better satisfy RONMF structural constraints

---

## Algorithm

### `BCD-RONMF.m`

Implements the Block Coordinate Descent (BCD) algorithm:

- Alternates between updating \( U \) and \( V \)
- Enforces:
  - Nonnegativity constraints
  - Column normalization on \( U \)
  - 1-sparsity (winner-takes-all) constraint on \( V \)

**Stopping criteria:**
- Stabilization of objective value
- Reduction of optimality gap

---

## Demo

### `Demo.m`

A demonstration script that:
- Loads the dataset
- Initializes \( U \) and \( V \)
- Runs the BCD-RONMF algorithm
- Displays convergence behavior

Run the demo using:
```matlab
Demo
