"""Agrega a barrido.csv las filas de n=200000 con 108 iteraciones (corridas limpias ya medidas),
leidas de sus logs, y anade la columna 'origen'. Idempotente."""
import csv, os, re
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "resultados", "barrido.csv")
D = os.path.join(RAIZ, "resultados", "tours_igualdad_m")


def leer(ruta):
    with open(ruta, "rb") as f:
        b = f.read()
    return b.decode("utf-16" if b[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8", errors="replace")


def fila_aco(txt):
    c = [l for l in txt.splitlines() if l.strip().startswith("CSV,")][-1].strip().split(",")
    return float(c[6]), float(c[8]) + float(c[9]), float(c[10]), "SI" if "tour_valido=SI" in txt else "NO"


def fila_pso(txt):
    c = [l for l in txt.splitlines() if l.strip().startswith("CSV,")][-1].strip().split(",")
    return float(c[4]), float(c[5]) + float(c[6]), float(c[7]), "SI" if "tour_valido=SI" in txt else "NO"


with open(CSV, newline="") as f:
    filas = list(csv.DictReader(f))
for r in filas:
    r.setdefault("origen", "barrido")
if not any(r["origen"] != "barrido" for r in filas):
    for alg, fn, arch in (("ACO", fila_aco, "log_aco_n200000_seed1.txt"), ("PSO", fila_pso, "log_pso_n200000_seed1.txt")):
        L, t, m, ok = fn(leer(os.path.join(D, arch)))
        filas.append(dict(algoritmo=alg, n=200000, agentes=2048, iteraciones=108, semilla=1, L_mejor=f"{L:.6f}",
                          tiempo_s=f"{t:.3f}", memoria_MB=f"{m:.1f}", tour_valido=ok,
                          origen=f"ronda2 resultados/tours_igualdad_m/{arch} (instancia interna, sin --input)"))
cols = ["algoritmo", "n", "agentes", "iteraciones", "semilla", "L_mejor", "tiempo_s", "memoria_MB", "tour_valido", "origen"]
with open(CSV, "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(filas)
print(len(filas), "filas;", sum(r["tour_valido"] != "SI" for r in filas), "con tour no valido")
