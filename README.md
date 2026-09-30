# Comparación PSO vs. ACO para el TSP

Implementación de PSO (Particle Swarm Optimization, codificación de claves
aleatorias) para el problema del viajante, en las mismas tres instancias del
proyecto ACO_IA (n=20, 2 000 y 200 000 ciudades), comparando los resultados
contra las cifras oficiales ya obtenidas con ACO en ese proyecto.

## Estructura

```
src/pso_tsp.cpp     Implementación de PSO en C++20 + OpenMP
informe/informe.tex Informe (estilo institucional USA)
resultados/         CSV y logs de las corridas
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
