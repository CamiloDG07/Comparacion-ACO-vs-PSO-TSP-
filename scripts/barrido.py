"""Barrido PSO vs ACO: agentes (particulas = hormigas) x iteraciones, 3 semillas.
Ambos algoritmos leen la MISMA instancia (archivo x y por (n, semilla)).
Reanudable: omite las combinaciones ya presentes en resultados/barrido.csv."""
import csv, ctypes, os, re, subprocess, sys, time
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSO = os.path.join(RAIZ, "src", "pso_tsp.exe")
ACO = r"D:\camilo\Documentos\ACO_IA\src\aco_tsp.exe"
CSV = os.path.join(RAIZ, "resultados", "barrido.csv")
CURVAS = os.path.join(RAIZ, "resultados", "curvas")
INST = os.path.join(RAIZ, "resultados", "barrido_instancias")
COLS = ["algoritmo", "n", "agentes", "iteraciones", "semilla", "L_mejor", "tiempo_s", "memoria_MB", "tour_valido"]
TIMEOUT = 900
SEMILLAS = (1, 2, 3)
AGS = (50, 500, 2048)
ITS = (5, 20, 100)
GRID = [(20, AGS, ITS), (200, AGS, ITS), (2000, AGS, ITS), (20000, AGS, ITS), (200000, (2048,), (5, 20))]
ACO_BASE = ["--alpha", "1.5", "--beta", "5", "--rho", "0.1", "--qfac", "3", "--two", "1", "--alpha2", "1.0"]


def instancia(n, s):
    os.makedirs(INST, exist_ok=True)
    ruta = os.path.join(INST, f"n{n}_s{s}.txt")
    if not os.path.exists(ruta):
        p = np.random.default_rng(1000 * n + s).random((n, 2))
        np.savetxt(ruta, p, fmt="%.9f")
    return ruta


def comando(alg, n, a, it, s, ruta):
    if alg == "ACO":
        k = "19" if n == 20 else "8"
        return [ACO, "--input", ruta, "--ants", str(a), "--iters", str(it), "--K", k, *ACO_BASE, "--seed", str(s)]
    return [PSO, "--input", ruta, "--particles", str(a), "--iters", str(it), "--seed", str(s)]


def parsear(alg, txt):
    csvl = [l for l in txt.splitlines() if l.startswith("CSV,")][-1].split(",")
    valido = "SI" if re.search(r"tour_valido=SI", txt) else "NO"
    if alg == "ACO":   # CSV,n,m,K,two,iters,L,Lnn,prep,t_aco,rss
        return float(csvl[6]), float(csvl[8]) + float(csvl[9]), float(csvl[10]), valido
    return float(csvl[4]), float(csvl[5]) + float(csvl[6]), float(csvl[7]), valido   # CSV,n,P,it,L,prep,t,mem


def hechas():
    if not os.path.exists(CSV):
        return set()
    with open(CSV, newline="") as f:
        return {(r["algoritmo"], int(r["n"]), int(r["agentes"]), int(r["iteraciones"]), int(r["semilla"])) for r in csv.DictReader(f)}


def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # evitar suspension
    os.makedirs(CURVAS, exist_ok=True)
    ya = hechas()
    nuevo = not os.path.exists(CSV)
    filtro_n = {int(x) for x in sys.argv[1:]}
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(COLS)
        for n, ags, its in GRID:
            if filtro_n and n not in filtro_n:
                continue
            for a in ags:
                for it in its:
                    for s in SEMILLAS:
                        ruta = instancia(n, s)
                        for alg in ("ACO", "PSO"):
                            if (alg, n, a, it, s) in ya:
                                continue
                            t0 = time.time()
                            try:
                                r = subprocess.run(comando(alg, n, a, it, s, ruta), capture_output=True, text=True, timeout=TIMEOUT)
                                txt = r.stdout
                                L, t, mem, ok = parsear(alg, txt)
                            except Exception as e:
                                print(f"FALLO {alg} n={n} a={a} it={it} s={s}: {e}", flush=True)
                                continue
                            with open(os.path.join(CURVAS, f"{alg}_n{n}_a{a}_i{it}_s{s}.txt"), "w") as g:
                                g.write(txt)
                            w.writerow([alg, n, a, it, s, f"{L:.6f}", f"{t:.3f}", f"{mem:.1f}", ok])
                            f.flush()
                            print(f"{time.strftime('%H:%M:%S')} {alg} n={n} a={a} it={it} s={s} L={L:.4f} t={t:.2f}s mem={mem:.1f}MB valido={ok} (pared {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
