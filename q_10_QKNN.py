"""
QKNN - Quantum K-Nearest Neighbors
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/Users/4c/Desktop/GHQ/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
K_NEIGHBORS = 5

def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def compute_quantum_kernel():
    n_states = 1 << NUM_QUBITS
    fmap = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)

    statevectors = []
    for v in range(n_states):
        feat = value_to_features(v)
        circ = fmap.assign_parameters(feat)
        sv = Statevector.from_instruction(circ)
        statevectors.append(sv)

    K = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i, n_states):
            fid = abs(statevectors[i].inner(statevectors[j])) ** 2
            K[i, j] = fid
            K[j, i] = fid

    return K


def qknn_predict(K, y, k=K_NEIGHBORS):
    n = K.shape[0]
    pred = np.zeros(n)

    for i in range(n):
        sims = K[i].copy()
        sims[i] = -1.0
        top_k = np.argsort(sims)[::-1][:k]
        weights = sims[top_k]
        w_sum = weights.sum()
        if w_sum > 0:
            pred[i] = np.dot(weights, y[top_k]) / w_sum
        else:
            pred[i] = y[top_k].mean()

    return pred


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- Kvantni kernel (ZZFeatureMap, {NUM_QUBITS}q, reps=1) ---")
    K = compute_quantum_kernel()
    print(f"  Kernel matrica: {K.shape}, rang: {np.linalg.matrix_rank(K)}")

    print(f"\n--- QKNN po pozicijama (k={K_NEIGHBORS}) ---")
    dists = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        pred = qknn_predict(K, y)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"  Poz {pos+1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QKNN, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()



"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Kvantni kernel (ZZFeatureMap, 5q, reps=1) ---
  Kernel matrica: (32, 32), rang: 32

--- QKNN po pozicijama (k=5) ---
  Poz 1 [1-33]: 4:0.069 | 26:0.067 | 25:0.066
  Poz 2 [2-34]: 7:0.064 | 5:0.057 | 2:0.054
  Poz 3 [3-35]: 26:0.058 | 10:0.054 | 3:0.050
  Poz 4 [4-36]: 24:0.063 | 27:0.061 | 11:0.060
  Poz 5 [5-37]: 18:0.071 | 25:0.067 | 24:0.054
  Poz 6 [6-38]: 32:0.071 | 25:0.053 | 27:0.052
  Poz 7 [7-39]: 8:0.072 | 17:0.070 | 33:0.069

==================================================
Predikcija (QKNN, deterministicki, seed=39):
[4, 7, 26, 27, 31, 32, 33]
==================================================
"""



"""
QKNN - Quantum K-Nearest Neighbors

Isti kvantni kernel (ZZFeatureMap, fidelity, 5 qubita)
KNN u kvantnom prostoru: za svaku vrednost nalazi 5 najslicnijih suseda po kvantnoj fidelity metrici
Predikcija = tezinski prosek frekvencija suseda (tezine = fidelity)
Bez treniranja, bez optimizacije - cist lazy learning u kvantnom feature prostoru
Najjednostavniji od svih modela, deterministicki, brz
"""
