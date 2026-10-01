import math


def dist(xs, ys, a, b):
    return math.hypot(xs[a] - xs[b], ys[a] - ys[b])


def held_karp_con_ruta(xs, ys, n):
    m = n - 1
    dp = {}
    parent = {}
    for j in range(m):
        dp[(1 << j, j)] = dist(xs, ys, 0, j + 1)
        parent[(1 << j, j)] = -1
    for mask in range(1, 1 << m):
        for j in range(m):
            if not (mask >> j) & 1:
                continue
            if (mask, j) not in dp:
                continue
            cur = dp[(mask, j)]
            for t in range(m):
                if (mask >> t) & 1:
                    continue
                nmask = mask | (1 << t)
                v = cur + dist(xs, ys, j + 1, t + 1)
                key = (nmask, t)
                if key not in dp or v < dp[key]:
                    dp[key] = v
                    parent[key] = j
    best = None
    bestj = -1
    full = (1 << m) - 1
    for j in range(m):
        v = dp[(full, j)] + dist(xs, ys, j + 1, 0)
        if best is None or v < best:
            best = v
            bestj = j
    mask = full
    j = bestj
    seq = []
    while j != -1:
        seq.append(j + 1)
        pj = parent[(mask, j)]
        mask = mask ^ (1 << j)
        j = pj
    seq.reverse()
    ruta = [0] + seq
    return best, ruta


if __name__ == "__main__":
    # Lee las coordenadas reales exportadas por ACO (mismo id de ciudad que
    # usa el proyecto, incluida la renumeracion por celda) en vez de
    # reconstruir el generador aleatorio de aco_tsp.cpp.
    coords = {}
    with open("resultados/tours_igualdad_m/aco_n20_seed1.txt") as f:
        for line in f:
            partes = line.split()
            coords[int(partes[0])] = (float(partes[1]), float(partes[2]))
    n = 20
    xs = [coords[i][0] for i in range(n)]
    ys = [coords[i][1] for i in range(n)]
    L, ruta = held_karp_con_ruta(xs, ys, n)
    print(f"optimo_exacto={L:.5f}")
    print("ruta:", ruta)
    with open("resultados/tours_igualdad_m/held_karp_n20_seed1.txt", "w") as f:
        for c in ruta:
            f.write(f"{c} {xs[c]:.6f} {ys[c]:.6f}\n")
