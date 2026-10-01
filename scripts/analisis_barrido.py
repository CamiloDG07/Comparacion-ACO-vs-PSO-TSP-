"""Analisis del barrido PSO vs ACO: tablas, figuras y conclusiones, todo desde resultados/barrido.csv."""
import os, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "resultados", "barrido.csv")
FIG = os.path.join(RAIZ, "figuras")
GEN = os.path.join(RAIZ, "informe", "generado")
PSO_EXE = os.path.join(RAIZ, "src", "pso_tsp.exe")
os.makedirs(FIG, exist_ok=True)
os.makedirs(GEN, exist_ok=True)
C = {"ACO": "#1f77b4", "PSO": "#d62728"}
NS = [20, 200, 2000, 20000, 200000]
DPI = 300
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def num(x, d=3):
    """Numero con coma decimal y espacio fino de miles, para LaTeX."""
    s = f"{x:,.{d}f}".replace(",", "\\,").replace(".", ",")
    return s


df = pd.read_csv(CSV)
assert (df["tour_valido"] == "SI").all(), "hay tours no validos"

# ---------- L_ref ----------
opt_csv = os.path.join(RAIZ, "resultados", "optimo_n20.csv")
if not os.path.exists(opt_csv):
    filas = []
    for s in (1, 2, 3):
        r = subprocess.run([PSO_EXE, "--input", os.path.join(RAIZ, "resultados", "barrido_instancias", f"n20_s{s}.txt"),
                            "--particles", "20", "--iters", "1", "--seed", "1", "--exact", "1"], capture_output=True, text=True).stdout
        v = [l for l in r.splitlines() if l.startswith("optimo_exacto")][0].split("=")[1].split()[0]
        filas.append((s, float(v)))
    pd.DataFrame(filas, columns=["semilla", "L_opt"]).to_csv(opt_csv, index=False)
opt20 = dict(pd.read_csv(opt_csv).values)


def lref(r):
    if r["n"] == 20:
        return opt20[int(r["semilla"])]
    sub = df[(df["n"] == r["n"]) & ((df["semilla"] == r["semilla"]) if r["n"] != 200000 else True)]
    return sub["L_mejor"].min()


df["L_ref"] = df.apply(lref, axis=1)
df["L_rel"] = df["L_mejor"] / df["L_ref"]
df["evals"] = df["agentes"] * df["iteraciones"]

g = df.groupby(["algoritmo", "n", "agentes", "iteraciones"])
res = g.agg(L=("L_mejor", "mean"), L_sd=("L_mejor", "std"), t=("tiempo_s", "mean"), t_sd=("tiempo_s", "std"),
            mem=("memoria_MB", "mean"), mem_sd=("memoria_MB", "std"), rel=("L_rel", "mean"), rel_sd=("L_rel", "std"),
            ns=("semilla", "count")).reset_index().fillna(0.0)
res["evals"] = res["agentes"] * res["iteraciones"]
res.to_csv(os.path.join(RAIZ, "resultados", "barrido_resumen.csv"), index=False)


def R(alg, n, a, it):
    s = res[(res.algoritmo == alg) & (res.n == n) & (res.agentes == a) & (res.iteraciones == it)]
    return s.iloc[0] if len(s) else None


def it_presupuesto(n, b):
    """En n=200000 el presupuesto 'moderado' corresponde a 108 iteraciones (ronda anterior)."""
    return 108 if (n == 200000 and b == 100) else b


# ---------- Figura 1: razones PSO/ACO vs n ----------
BUD = (5, 20, 100)
marc = {5: "o", 20: "s", 100: "^"}
col = {5: "#9ecae1", 20: "#4292c6", 100: "#08306b"}
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
cruces = {}
for ax, (campo, tit) in zip(axs, (("L", "Calidad: $L_{PSO}/L_{ACO}$"), ("t", "Tiempo: $t_{PSO}/t_{ACO}$"), ("mem", "Memoria: $M_{PSO}/M_{ACO}$"))):
    for b in BUD:
        xs, ys = [], []
        for n in NS:
            ib = it_presupuesto(n, b)
            p, a_ = R("PSO", n, 2048, ib), R("ACO", n, 2048, ib)
            if p is None or a_ is None:
                continue
            xs.append(n); ys.append(p[campo] / a_[campo])
        ax.plot(xs, ys, marker=marc[b], color=col[b], label=f"{b} it." + (" (108 it. en n=200 000)" if b == 100 else ""))
        if campo == "L":
            cruces[b] = [(xs[i], xs[i + 1]) for i in range(len(xs) - 1) if (ys[i] - 1) * (ys[i + 1] - 1) < 0]
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Número de ciudades $n$"); ax.set_title(tit)
axs[0].set_ylabel("Razón PSO / ACO (1 = empate)")
axs[0].legend(title="2 048 agentes", fontsize=8)
txt = "; ".join(f"{b} it.: " + ("cruce entre n=" + " y ".join(f"{a}-{c}" for a, c in cruces[b]) if cruces[b] else "sin cruce") for b in BUD)
fig.suptitle("Razón PSO/ACO con 2 048 agentes. Cruce de calidad: " + txt, fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIG, "barrido_razon_vs_n.png"), dpi=DPI); plt.close(fig)

# ---------- Figura 2: L/L_ref vs n ----------
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ls = {5: ":", 20: "--", 100: "-"}
for alg in ("ACO", "PSO"):
    for b in BUD:
        xs, ys, es = [], [], []
        for n in NS:
            r = R(alg, n, 2048, it_presupuesto(n, b))
            if r is not None:
                xs.append(n); ys.append(r["rel"]); es.append(r["rel_sd"])
        ax.errorbar(xs, ys, yerr=es, color=C[alg], ls=ls[b], marker=marc[b], capsize=2, label=f"{alg}, {b} it.")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Número de ciudades $n$"); ax.set_ylabel("$L_{mejor}/L_{ref}$ (adimensional)")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "barrido_calidad_vs_n.png"), dpi=DPI); plt.close(fig)

# ---------- Figura 3a/3b: calidad vs agentes / iteraciones ----------
NS4 = [20, 200, 2000, 20000]
for nombre, fijo, eje, fv, xl in (("barrido_calidad_agentes", "iteraciones", "agentes", 20, "Agentes (partículas u hormigas)"),
                                  ("barrido_calidad_iteraciones", "agentes", "iteraciones", 500, "Iteraciones")):
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, n in zip(axs, NS4):
        for alg in ("ACO", "PSO"):
            s = res[(res.algoritmo == alg) & (res.n == n) & (res[fijo] == fv)].sort_values(eje)
            ax.errorbar(s[eje], s["L"], yerr=s["L_sd"], color=C[alg], marker="o", capsize=2, label=alg)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"$n={n:,}$".replace(",", "\\,"))
        ax.set_xlabel(xl)
    axs[0].set_ylabel("Longitud del tour $L_{mejor}$ (u. de longitud)")
    axs[0].legend()
    fig.suptitle(f"{'Iteraciones fijas en 20' if fijo == 'iteraciones' else 'Agentes fijos en 500'}", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(FIG, nombre + ".png"), dpi=DPI); plt.close(fig)

# ---------- Figura 4: tiempo vs tours evaluados ----------
fig, axs = plt.subplots(1, 5, figsize=(16, 3.6))
for ax, n in zip(axs, NS):
    for alg in ("ACO", "PSO"):
        s = res[(res.algoritmo == alg) & (res.n == n)].sort_values("evals")
        ax.errorbar(s["evals"], s["t"], yerr=s["t_sd"], color=C[alg], marker="o", ls="none", capsize=2, label=alg)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"$n={n:,}$".replace(",", "\\,"))
    ax.set_xlabel("Tours evaluados (agentes × iteraciones)")
axs[0].set_ylabel("Tiempo (s)"); axs[0].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "barrido_tiempo.png"), dpi=DPI); plt.close(fig)

# ---------- Figura 5: memoria vs agentes ----------
fig, ax = plt.subplots(figsize=(6.8, 4.4))
cm = plt.cm.viridis(np.linspace(0, 0.9, len(NS)))
for c_, n in zip(cm, NS):
    s = res[(res.n == n)].groupby(["algoritmo", "agentes"]).mem.mean().reset_index()
    p = s[s.algoritmo == "PSO"]; a_ = s[s.algoritmo == "ACO"]
    ax.plot(p["agentes"], p["mem"], "o-", color=c_, label=f"PSO, n={n:,}".replace(",", " "))
    ax.plot(a_["agentes"], a_["mem"], "x-", color=c_, alpha=0.6)
    P_ = np.array([50, 500, 2048]); ax.plot(P_, 3 * n * P_ * 4 / 1e6, ":", color=c_)
ax.plot([], [], "kx-", label="ACO (x, mismos colores)"); ax.plot([], [], "k:", label="teórica PSO $3\\,n\\,P\\cdot4$ B")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Agentes (partículas u hormigas)"); ax.set_ylabel("Memoria pico (MB)")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "barrido_memoria.png"), dpi=DPI); plt.close(fig)

# ---------- Figura 6: calidad vs tiempo ----------
fig, axs = plt.subplots(1, 5, figsize=(16, 3.6))
for ax, n in zip(axs, NS):
    for alg in ("ACO", "PSO"):
        s = res[(res.algoritmo == alg) & (res.n == n)]
        ax.scatter(s["t"], s["rel"], color=C[alg], label=alg, s=22)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"$n={n:,}$".replace(",", "\\,"))
    ax.set_xlabel("Tiempo (s)")
axs[0].set_ylabel("$L_{mejor}/L_{ref}$ (adimensional)"); axs[0].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "barrido_calidad_vs_tiempo.png"), dpi=DPI); plt.close(fig)

# ---------- Figura 7: n=200000, P=m=2048, 108 it ----------
a_, p_ = R("ACO", 200000, 2048, 108), R("PSO", 200000, 2048, 108)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.8))
for ax, (campo, yl) in zip(axs, (("L", "Longitud del tour (u. de longitud)"), ("t", "Tiempo (s)"), ("mem", "Memoria pico (MB)"))):
    vals = [a_[campo], p_[campo]]
    ax.bar(["ACO", "PSO"], vals, color=[C["ACO"], C["PSO"]])
    ax.set_yscale("log"); ax.set_ylabel(yl)
    top = max(vals)
    ax.set_ylim(min(vals) / 3, top * 6)
    ax.text(0.5, 0.93, f"PSO/ACO = {p_[campo] / a_[campo]:.2f}".replace(".", ","), transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, f"{v:,.1f}".replace(",", " ").replace(".", ","), ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "comparacion_P_igual_m.png"), dpi=DPI); plt.close(fig)

# ---------- Tablas ----------
def fila(n, a, it):
    A, P = R("ACO", n, a, it), R("PSO", n, a, it)
    if A is None or P is None:
        return None
    pm = lambda r, c, d: f"{num(r[c], d)} $\\pm$ {num(r[c + '_sd'], d)}" if r["ns"] > 1 else num(r[c], d)
    return (f"{n:,}".replace(",", "\\,") + f" & {a:,}".replace(",", "\\,") + f" & {it} & {pm(A, 'L', 3)} & {pm(P, 'L', 3)} & "
            f"{num(A['t'], 2)} & {num(P['t'], 2)} & {num(A['mem'], 1)} & {num(P['mem'], 1)} \\\\")


enc = ("\\footnotesize\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{rrrrrrrrr}\\toprule\n\\rowcolor{gray!25}\n"
       "$n$ & Agentes & It. & $L$ ACO & $L$ PSO & $t$ ACO (s) & $t$ PSO (s) & Mem. ACO (MB) & Mem. PSO (MB) \\\\\\midrule\n")
pie = "\\bottomrule\\end{tabular}%\n}\n"
corto, largo = [], []
for n in NS:
    for it in (5, 20, 100, 108):
        f_ = fila(n, 2048, it)
        if f_:
            corto.append(f_)
    for a in (50, 500, 2048):
        for it in (5, 20, 100):
            f_ = fila(n, a, it)
            if f_:
                largo.append(f_)
open(os.path.join(GEN, "tabla_resumen.tex"), "w", encoding="utf-8").write(enc + "\n".join(corto) + "\n" + pie)
open(os.path.join(GEN, "tabla_completa.tex"), "w", encoding="utf-8").write(enc + "\n".join(largo) + "\n" + pie)

# ---------- Conclusiones calculadas ----------
out = []
log = lambda s: out.append(s)
log("== Calidad por n y presupuesto (2048 agentes): L_PSO/L_ACO ==")
razL = {}
for n in NS:
    for b in BUD:
        ib = it_presupuesto(n, b)
        A, P = R("ACO", n, 2048, ib), R("PSO", n, 2048, ib)
        if A is None or P is None:
            continue
        razL[(n, b)] = P["L"] / A["L"]
        log(f"n={n} it={ib}: L_ACO={A['L']:.4f} L_PSO={P['L']:.4f} razon={P['L']/A['L']:.3f} mejor={'ACO' if A['L'] < P['L'] else 'PSO'}")
log("\n== Las 9 configuraciones emparejadas por n: ACO gana en calidad ==")
gana = {}
for n in NS4:
    k = 0; tot = 0; rz = []
    for a in (50, 500, 2048):
        for it in (5, 20, 100):
            A, P = R("ACO", n, a, it), R("PSO", n, a, it)
            tot += 1; k += A["L"] < P["L"]; rz.append(P["L"] / A["L"])
    gana[n] = (k, tot, float(np.median(rz)), min(rz), max(rz))
    log(f"n={n}: ACO gana {k}/{tot}; razon mediana {np.median(rz):.3f} (min {min(rz):.3f}, max {max(rz):.3f})")
log("\n== Beneficio de iterar (2048 agentes): mejora % de L entre 5 it. y 100 it. (108 en n=200000 vs 20) ==")
mej = {}
for n in NS:
    for alg in ("ACO", "PSO"):
        i0 = 5; i1 = 100 if n != 200000 else 20
        a0, a1 = R(alg, n, 2048, i0), R(alg, n, 2048, i1)
        m = 100 * (a0["L"] - a1["L"]) / a0["L"]
        mej[(alg, n)] = (i0, i1, m)
        log(f"{alg} n={n}: L({i0})={a0['L']:.4f} -> L({i1})={a1['L']:.4f}: mejora {m:.2f}%")
log("\n== Tiempo y memoria PSO/ACO (2048 agentes) ==")
rt, rm = {}, {}
for n in NS:
    lst_t, lst_m = [], []
    for it in (5, 20, 100, 108):
        A, P = R("ACO", n, 2048, it), R("PSO", n, 2048, it)
        if A is None or P is None:
            continue
        lst_t.append((it, P["t"] / A["t"])); lst_m.append((it, P["mem"] / A["mem"]))
        log(f"n={n} it={it}: t_ACO={A['t']:.3f} t_PSO={P['t']:.3f} razon_t={P['t']/A['t']:.3f} | M_ACO={A['mem']:.1f} M_PSO={P['mem']:.1f} razon_M={P['mem']/A['mem']:.2f}")
    rt[n] = lst_t; rm[n] = lst_m
log("\n== Tiempo en todas las configuraciones emparejadas: cuantas veces PSO es mas rapido ==")
rapido = {}
for n in NS4:
    k = 0; tot = 0; rz = []
    for a in (50, 500, 2048):
        for it in (5, 20, 100):
            A, P = R("ACO", n, a, it), R("PSO", n, a, it)
            tot += 1; k += P["t"] < A["t"]; rz.append(P["t"] / A["t"])
    rapido[n] = (k, tot, float(np.median(rz)), min(rz), max(rz))
    log(f"n={n}: PSO mas rapido en {k}/{tot}; razon_t mediana {np.median(rz):.3f} (min {min(rz):.3f}, max {max(rz):.3f})")
log("\n== Mismo tiempo: mejor L_rel alcanzable con t <= T (T = menor de los dos tiempos maximos) ==")
igt = {}
for n in NS:
    A = res[(res.algoritmo == "ACO") & (res.n == n)]; P = res[(res.algoritmo == "PSO") & (res.n == n)]
    T = min(A["t"].max(), P["t"].max())
    ba = A[A.t <= T]["rel"].min(); bp = P[P.t <= T]["rel"].min()
    dom = sum(((A["t"] <= p.t) & (A["rel"] <= p.rel)).any() for p in P.itertuples())
    igt[n] = (T, ba, bp, dom, len(P))
    log(f"n={n}: T={T:.2f}s mejor L/Lref ACO={ba if ba==ba else float('nan'):.4f} PSO={bp if bp==bp else float('nan'):.4f}; configs PSO dominadas por alguna de ACO: {dom}/{len(P)}")
log("\n== Memoria PSO medida vs teorica 3*n*P*4 B (2048 agentes) ==")
for n in NS:
    P = R("PSO", n, 2048, 5 if n != 200000 else 5)
    th = 3 * n * 2048 * 4 / 1e6
    log(f"n={n}: medida={P['mem']:.1f} MB teorica={th:.1f} MB razon={P['mem']/th:.2f}")
open(os.path.join(RAIZ, "resultados", "conclusiones_barrido.txt"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))

# ---------- Viñetas (texto generado) ----------
nf = lambda x, d=2: f"{x:,.{d}f}".replace(",", "\\,").replace(".", ",")
nl = lambda n: f"{n:,}".replace(",", "\\,")
v = []
todas_aco = all(razL[k] > 1 for k in razL)
hay_cruce = any(cruces[b] for b in BUD)
rs = ", ".join(f"$n={nl(n)}$: {nf(gana[n][2])}" for n in NS4)
v.append("\\textbf{Calidad: ACO es mejor." if todas_aco else "\\textbf{Calidad: el ganador depende de la configuración.")
v[-1] += f"}} Con 2\\,048 agentes ACO obtiene menor $L_{{mejor}}$ en {sum(razL[k] > 1 for k in razL)} de {len(razL)} combinaciones de $n$ y presupuesto; " \
         f"en las 9 configuraciones emparejadas de cada $n\\le 20\\,000$ gana en " + ", ".join(f"{gana[n][0]}/{gana[n][1]} ($n={nl(n)}$)" for n in NS4) + \
         f". Razón mediana $L_{{PSO}}/L_{{ACO}}$: {rs}."
v.append("\\textbf{Dependencia de $n$: " + ("hay cruce" if hay_cruce else "no hay cruce en el rango probado ($n=20$ a $n=200\\,000$)") + ".}" +
         (" " + "; ".join(f"{b} it.: entre $n={nl(a)}$ y ${nl(c)}$" for b in BUD for a, c in cruces[b]) + "." if hay_cruce else "") +
         " Razón $L_{PSO}/L_{ACO}$ con 2\\,048 agentes y 100 it. (108 en $n=200\\,000$): " +
         ", ".join(f"$n={nl(n)}$: {nf(razL[(n, 100)])}" for n in NS if (n, 100) in razL) + ".")
v.append("\\textbf{Presupuesto.} Pasar de 5 a 100 iteraciones (de 5 a 20 en $n=200\\,000$) mejora $L_{mejor}$, con 2\\,048 agentes, en ACO " +
         ", ".join(f"{nf(mej[('ACO', n)][2], 1)}\\,\\% ($n={nl(n)}$)" for n in NS) + " y en PSO " +
         ", ".join(f"{nf(mej[('PSO', n)][2], 1)}\\,\\%" + f" ($n={nl(n)}$)" for n in NS) + ".")
tmix = any(r < 1 for n in NS4 for r in [rapido[n][3]]) and any(r > 1 for n in NS4 for r in [rapido[n][4]])
v.append("\\textbf{Tiempo: " + ("no concluyente" if tmix else ("PSO es más rápido en todas las configuraciones emparejadas" if all(rapido[n][0] == rapido[n][1] for n in NS4) else "resultado uniforme")) + ".} Razón $t_{PSO}/t_{ACO}$ mediana en las 9 configuraciones: " +
         ", ".join(f"{nf(rapido[n][2])} (rango {nf(rapido[n][3])}--{nf(rapido[n][4])}, PSO más rápido en {rapido[n][0]}/{rapido[n][1]}; $n={nl(n)}$)" for n in NS4) +
         "; con $n=200\\,000$, 2\\,048 agentes y 108 it.: " + nf(rt[200000][-1][1]) + ".")
v.append("\\textbf{Memoria: PSO usa más en todo el rango.} Razón $M_{PSO}/M_{ACO}$ con 2\\,048 agentes: " +
         ", ".join(f"{nf(rm[n][0][1], 1)} ($n={nl(n)}$)" for n in NS) + ".")
v.append("\\textbf{A igual tiempo:} " + "; ".join(
    f"$n={nl(n)}$ ($T={nf(igt[n][0])}$ s): $L/L_{{ref}}$ ACO {nf(igt[n][1], 3)}, PSO {nf(igt[n][2], 3)}" for n in NS) + ".")
v.append("\\emph{Límites:} PSO con parámetros fijos ($w=0{,}7$, $c_1=c_2=1{,}5$) y sin búsqueda local; los resultados valen para esta implementación.")
rr = {n: R("PSO", n, 2048, 5)["mem"] / (3 * n * 2048 * 4 / 1e6) for n in NS}
mem_txt = ("En PSO con 2\\,048 partículas la memoria pico medida dividida por $3\\,n\\,P\\cdot4$ bytes vale " +
           ", ".join(f"{nf(rr[n])} ($n={nl(n)}$)" for n in NS) +
           "; para $n\\ge 2\\,000$ la fórmula describe lo medido, y en $n$ pequeña domina la memoria base del proceso.")
open(os.path.join(GEN, "memoria_texto.tex"), "w", encoding="utf-8").write(mem_txt + "\n")
open(os.path.join(GEN, "conclusiones.tex"), "w", encoding="utf-8").write("\\begin{itemize}\n" + "\n".join("\\item " + x for x in v) + "\n\\end{itemize}\n")
