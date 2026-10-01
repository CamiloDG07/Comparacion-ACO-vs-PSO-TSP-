"""Longitud media por arista (L/n) de los tours exportados de la ronda 2, y comparacion con un tour
aleatorio de las mismas ciudades (media de 20 permutaciones, semilla 0). Escribe resultados/aristas_tours.csv."""
import csv, os
import numpy as np
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(RAIZ, "resultados", "tours_igualdad_m")
filas = []
for n in (20, 2000, 200000):
    for alg in ("aco", "pso"):
        xy = np.loadtxt(os.path.join(D, f"{alg}_n{n}_seed1.txt"))[:, 1:3]
        L = np.hypot(*(np.roll(xy, -1, 0) - xy).T)
        rng = np.random.default_rng(0)
        n_perm = 20 if n < 200000 else 5
        aleat = np.mean([np.hypot(*(np.roll(xy[q], -1, 0) - xy[q]).T).sum() for q in [rng.permutation(len(xy)) for _ in range(n_perm)]])
        filas.append(dict(n=n, algoritmo=alg.upper(), L=L.sum(), L_sobre_n=L.mean(), arista_mediana=np.median(L),
                          arista_p99=np.percentile(L, 99), arista_max=L.max(), L_aleatorio=aleat,
                          pct_bajo_aleatorio=100 * (aleat - L.sum()) / aleat))
with open(os.path.join(RAIZ, "resultados", "aristas_tours.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, list(filas[0])); w.writeheader(); w.writerows(filas)
for r in filas:
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
