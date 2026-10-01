# Comparación PSO vs. ACO para el TSP

Autores: Camilo Andrés Díaz García, Mario Jiménez y Juan David Andrade.

Implementación de PSO (Particle Swarm Optimization, codificación de claves
aleatorias) para el problema del viajante, en las mismas tres instancias del
proyecto ACO_IA (n=20, 2 000 y 200 000 ciudades), comparando los resultados
contra las cifras oficiales ya obtenidas con ACO en ese proyecto.

## Estructura

```
src/pso_tsp.cpp     Implementación de PSO en C++20 + OpenMP
informe/informe.tex Informe (estilo institucional USA)
resultados/         Logs de las corridas (igualdad_m/: PSO con P=m; tours_igualdad_m/: tours y logs de rutas)
figuras/            Rutas dibujadas (n=20, 2 000 y 200 000)
scripts/            Held-Karp de n=20, graficación de rutas, barrido (barrido.py), incorporación de n=200 000 con 108 it. y análisis (analisis_barrido.py)
```

## Compilar

```
g++ -O3 -march=native -std=c++20 -fopenmp -static src/pso_tsp.cpp -o src/pso_tsp.exe -lpsapi
```

## Ejecutar

```
src/pso_tsp.exe --n 2000 --particles 200 --iters 500 --w 0.7 --c1 1.5 --c2 1.5 --seed 1
```

Parámetros: `--n` (ciudades), `--particles`, `--iters`, `--w`/`--c1`/`--c2`
(inercia, coeficiente cognitivo, coeficiente social), `--seed`, `--exact 1`
(verificar contra Held-Karp, solo n<=20), `--tour <archivo>` (guardar el
tour final), `--input <archivo>` (instancia TSPLIB o "x y" plano).

## Barrido PSO vs ACO

```
python scripts/barrido.py                      # escribe resultados/barrido.csv (reanudable)
python scripts/incorporar_n200000_108.py       # agrega las filas de n=200 000 con 108 it.
python scripts/analisis_barrido.py             # figuras, tablas y conclusiones del informe
```

Las instancias de n=20 000 y n=200 000 se regeneran con la misma semilla (no se versionan).
