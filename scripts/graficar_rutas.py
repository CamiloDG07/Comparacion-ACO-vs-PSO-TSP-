import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

RESULTADOS = "resultados/tours_igualdad_m"
FIGURAS = "figuras"
os.makedirs(FIGURAS, exist_ok=True)


def leer_tour(ruta):
    xs, ys = [], []
    with open(ruta) as f:
        for line in f:
            partes = line.split()
            if len(partes) < 3:
                continue
            xs.append(float(partes[1]))
            ys.append(float(partes[2]))
    return np.array(xs), np.array(ys)


def dibujar_panel_simple(ax, xs, ys, titulo, color):
    xc = np.append(xs, xs[0])
    yc = np.append(ys, ys[0])
    ax.plot(xc, yc, "-", color=color, linewidth=1.0, marker="o", markersize=3)
    ax.set_title(titulo, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def dibujar_panel_denso(ax, xs, ys, titulo, color):
    xc = np.append(xs, xs[0])
    yc = np.append(ys, ys[0])
    segs = np.stack([np.column_stack([xc[:-1], yc[:-1]]), np.column_stack([xc[1:], yc[1:]])], axis=1)
    lc = LineCollection(segs, colors=color, linewidths=0.15, rasterized=True)
    ax.add_collection(lc)
    ax.set_title(titulo, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(xc.min(), xc.max())
    ax.set_ylim(yc.min(), yc.max())
    ax.set_xticks([])
    ax.set_yticks([])


def figura_n20():
    xa, ya = leer_tour(f"{RESULTADOS}/aco_n20_seed1.txt")
    xp, yp = leer_tour(f"{RESULTADOS}/pso_n20_seed1.txt")
    xh, yh = leer_tour(f"{RESULTADOS}/held_karp_n20_seed1.txt")
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
    dibujar_panel_simple(axs[0], xh, yh, "Held-Karp (óptimo)\nL=4,106", "#2ca02c")
    dibujar_panel_simple(axs[1], xa, ya, "ACO (m=20, K=19)\nL=4,260", "#1f77b4")
    dibujar_panel_simple(axs[2], xp, yp, "PSO (P=20)\nL=4,804", "#d62728")
    fig.tight_layout(rect=[0,0,1,0.92])
    fig.savefig(f"{FIGURAS}/rutas_n20.png", dpi=200)
    plt.close(fig)
    print("figuras/rutas_n20.png guardada")


def figura_n2000():
    xa, ya = leer_tour(f"{RESULTADOS}/aco_n2000_seed1.txt")
    xp, yp = leer_tour(f"{RESULTADOS}/pso_n2000_seed1.txt")
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.2))
    dibujar_panel_simple(axs[0], xa, ya, "ACO (m=2 000)\nL=37,437", "#1f77b4")
    dibujar_panel_simple(axs[1], xp, yp, "PSO (P=2 000)\nL=997,442", "#d62728")
    fig.tight_layout(rect=[0,0,1,0.92])
    fig.savefig(f"{FIGURAS}/rutas_n2000.png", dpi=200)
    plt.close(fig)
    print("figuras/rutas_n2000.png guardada")


def figura_n200000(l_aco, l_pso):
    xa, ya = leer_tour(f"{RESULTADOS}/aco_n200000_seed1.txt")
    xp, yp = leer_tour(f"{RESULTADOS}/pso_n200000_seed1.txt")
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
    dibujar_panel_denso(axs[0], xa, ya, f"ACO (m=2 048, 108 it.)\nL={l_aco}", "#1f77b4")
    dibujar_panel_denso(axs[1], xp, yp, f"PSO\nL={l_pso}", "#d62728")
    fig.tight_layout(rect=[0,0,1,0.92])
    fig.savefig(f"{FIGURAS}/rutas_n200000.png", dpi=200)
    plt.close(fig)
    print("figuras/rutas_n200000.png guardada")


if __name__ == "__main__":
    import sys
    if "n20" in sys.argv or len(sys.argv) == 1:
        figura_n20()
    if "n2000" in sys.argv or len(sys.argv) == 1:
        figura_n2000()
    if "n200000" in sys.argv:
        l_aco = sys.argv[sys.argv.index("n200000") + 1]
        l_pso = sys.argv[sys.argv.index("n200000") + 2]
        figura_n200000(l_aco, l_pso)
