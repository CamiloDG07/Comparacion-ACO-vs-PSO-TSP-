#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>
#include <omp.h>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using u32 = uint32_t;
using u64 = uint64_t;
using Clock = std::chrono::steady_clock;

static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
static double peakRssMB() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0.0;
    return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
#else
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024.0;
#endif
}

struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL) { next(); next(); }
    inline u64 next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    inline float uni() { return (next() >> 40) * (1.0f / 16777216.0f); }
};

struct Params {
    int n = 2000;
    int P = 100;
    int iters = 200;
    float w = 0.7f, c1 = 1.5f, c2 = 1.5f;
    float keyMin = -50.0f, keyMax = 50.0f;
    u64 seed = 1;
    int exact = 0;
    std::string inputFile;
    std::string tourOut;
    double timeBudget = 1e18;
};

static bool loadPointsFromFile(const std::string& path, std::vector<double>& xs,
                                std::vector<double>& ys) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    char line[1024];
    bool tsplib = false, inCoord = false;
    while (fgets(line, sizeof(line), f)) {
        std::string s(line);
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        size_t b = s.find_last_not_of(" \t\r\n");
        s = s.substr(a, b - a + 1);
        if (s.empty()) continue;
        if (!tsplib && !inCoord) {
            if (s.rfind("NAME", 0) == 0 || s.rfind("TYPE", 0) == 0 || s.rfind("COMMENT", 0) == 0 ||
                s.rfind("DIMENSION", 0) == 0 || s.rfind("EDGE_WEIGHT_TYPE", 0) == 0 ||
                s.rfind("NODE_COORD_SECTION", 0) == 0) {
                tsplib = true;
            }
        }
        if (tsplib) {
            if (s.rfind("NODE_COORD_SECTION", 0) == 0) { inCoord = true; continue; }
            if (!inCoord) continue;
            if (s.rfind("EOF", 0) == 0) break;
            double idx, x, y;
            if (sscanf(s.c_str(), "%lf %lf %lf", &idx, &x, &y) == 3) { xs.push_back(x); ys.push_back(y); }
        } else {
            double x, y;
            if (sscanf(s.c_str(), "%lf %lf", &x, &y) == 2) { xs.push_back(x); ys.push_back(y); }
        }
    }
    fclose(f);
    return !xs.empty();
}

struct Instance {
    int n = 0;
    std::vector<float> x, y;
    inline float dist(int a, int b) const {
        float dx = x[a] - x[b], dy = y[a] - y[b];
        return std::sqrt(dx * dx + dy * dy);
    }
};

static void buildInstance(Instance& I, Params& P) {
    if (!P.inputFile.empty()) {
        std::vector<double> xs, ys;
        if (!loadPointsFromFile(P.inputFile, xs, ys)) {
            fprintf(stderr, "no se pudo leer --input %s\n", P.inputFile.c_str());
            exit(1);
        }
        const int nf = (int)xs.size();
        P.n = nf;
        double minx = xs[0], maxx = xs[0], miny = ys[0], maxy = ys[0];
        for (int i = 1; i < nf; i++) {
            minx = std::min(minx, xs[i]); maxx = std::max(maxx, xs[i]);
            miny = std::min(miny, ys[i]); maxy = std::max(maxy, ys[i]);
        }
        double range = std::max(maxx - minx, maxy - miny); if (range <= 0) range = 1.0;
        I.x.resize(nf); I.y.resize(nf);
        for (int i = 0; i < nf; i++) {
            I.x[i] = (float)((xs[i] - minx) / range);
            I.y[i] = (float)((ys[i] - miny) / range);
        }
    } else {
        const int n0 = P.n;
        I.x.resize(n0); I.y.resize(n0);
        Rng rng(P.seed * 7919 + 13);
        for (int i = 0; i < n0; i++) { I.x[i] = rng.uni(); I.y[i] = rng.uni(); }
    }
    I.n = P.n;
}

static double heldKarp(const Instance& I) {
    int n = I.n; int m = n - 1;
    size_t S = (size_t)1 << m;
    std::vector<double> dp(S * m, 1e18);
    for (int j = 0; j < m; j++) dp[((size_t)1 << j) * m + j] = I.dist(0, j + 1);
    for (size_t mask = 1; mask < S; mask++)
        for (int j = 0; j < m; j++) {
            if (!(mask >> j & 1)) continue;
            double cur = dp[mask * m + j]; if (cur >= 1e17) continue;
            for (int t = 0; t < m; t++) {
                if (mask >> t & 1) continue;
                size_t nm = mask | ((size_t)1 << t);
                double v = cur + I.dist(j + 1, t + 1);
                if (v < dp[nm * m + t]) dp[nm * m + t] = v;
            }
        }
    double best = 1e18;
    for (int j = 0; j < m; j++) best = std::min(best, dp[(S - 1) * m + j] + I.dist(j + 1, 0));
    return best;
}

struct ThreadScratch {
    std::vector<std::pair<float, int>> keyed;
    std::vector<int> tour;
    void init(int n) { keyed.resize(n); tour.resize(n); }
};

static double evalTour(const Instance& I, const float* keys, ThreadScratch& S) {
    const int n = I.n;
    for (int i = 0; i < n; i++) S.keyed[i] = {keys[i], i};
    std::sort(S.keyed.begin(), S.keyed.end());
    for (int i = 0; i < n; i++) S.tour[i] = S.keyed[i].second;
    double L = 0;
    for (int i = 0; i + 1 < n; i++) L += I.dist(S.tour[i], S.tour[i + 1]);
    L += I.dist(S.tour[n - 1], S.tour[0]);
    return L;
}

int main(int argc, char** argv) {
    Params P;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--n") P.n = std::stoi(nx());
        else if (a == "--particles") P.P = std::stoi(nx());
        else if (a == "--iters") P.iters = std::stoi(nx());
        else if (a == "--w") P.w = std::stof(nx());
        else if (a == "--c1") P.c1 = std::stof(nx());
        else if (a == "--c2") P.c2 = std::stof(nx());
        else if (a == "--seed") P.seed = std::stoull(nx());
        else if (a == "--exact") P.exact = std::stoi(nx());
        else if (a == "--input") P.inputFile = nx();
        else if (a == "--tour") P.tourOut = nx();
        else if (a == "--time") P.timeBudget = std::stod(nx());
        else { fprintf(stderr, "arg desconocido: %s\n", a.c_str()); return 1; }
    }
    const int nth = omp_get_max_threads();
    auto t0 = Clock::now();

    Instance I;
    buildInstance(I, P);
    const int n = I.n, np = P.P;
    auto t1 = Clock::now();

    std::vector<float> x((size_t)n * np), v((size_t)n * np, 0.0f);
    std::vector<float> pbestX((size_t)n * np), gbestX(n);
    std::vector<double> pbestL(np, 1e300);
    double gbestL = 1e300;

    std::vector<Rng> rngs;
    for (int p = 0; p < np; p++) rngs.emplace_back(P.seed * 1000003ULL + p * 7 + 1);
    for (int p = 0; p < np; p++)
        for (int i = 0; i < n; i++) x[(size_t)p * n + i] = rngs[p].uni();

    std::vector<ThreadScratch> TS(nth);
    for (auto& t : TS) t.init(n);

    for (int p = 0; p < np; p++) {
        double L = evalTour(I, &x[(size_t)p * n], TS[0]);
        pbestL[p] = L;
        std::memcpy(&pbestX[(size_t)p * n], &x[(size_t)p * n], n * sizeof(float));
        if (L < gbestL) { gbestL = L; std::memcpy(gbestX.data(), &x[(size_t)p * n], n * sizeof(float)); }
    }

    printf("# n=%d particulas=%d iters=%d w=%.2f c1=%.2f c2=%.2f hilos=%d semilla=%llu\n",
           n, np, P.iters, P.w, P.c1, P.c2, nth, (unsigned long long)P.seed);
    printf("# preparacion: %.3f s | L_inicial=%.4f\n", secs(t0, t1), gbestL);
    printf("# iter  mejor_global  t_iter(s)  t_total(s)\n");

    auto tLoop = Clock::now();
    int itDone = 0;
    for (int it = 0; it < P.iters; it++) {
        auto ti = Clock::now();
#pragma omp parallel
        {
            ThreadScratch& S = TS[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 1)
            for (int p = 0; p < np; p++) {
                Rng local(P.seed * 2000003ULL + (u64)p * 97 + (u64)it * 131071ULL + 1);
                float* xp = &x[(size_t)p * n];
                float* vp = &v[(size_t)p * n];
                const float* pb = &pbestX[(size_t)p * n];
                for (int i = 0; i < n; i++) {
                    float r1 = local.uni(), r2 = local.uni();
                    vp[i] = P.w * vp[i] + P.c1 * r1 * (pb[i] - xp[i]) + P.c2 * r2 * (gbestX[i] - xp[i]);
                    float nv = xp[i] + vp[i];
                    if (nv < P.keyMin) { nv = P.keyMin; vp[i] = 0.0f; }
                    if (nv > P.keyMax) { nv = P.keyMax; vp[i] = 0.0f; }
                    xp[i] = nv;
                }
                double L = evalTour(I, xp, S);
                if (L < pbestL[p]) { pbestL[p] = L; std::memcpy(&pbestX[(size_t)p * n], xp, n * sizeof(float)); }
            }
        }
        for (int p = 0; p < np; p++)
            if (pbestL[p] < gbestL) { gbestL = pbestL[p]; std::memcpy(gbestX.data(), &pbestX[(size_t)p * n], n * sizeof(float)); }
        itDone = it + 1;
        double dt = secs(ti, Clock::now()), tot = secs(tLoop, Clock::now());
        if (n <= 2000 || (it % 10 == 0) || it + 1 == P.iters) {
            printf("%5d  %12.4f  %9.3f  %10.3f\n", it + 1, gbestL, dt, tot);
            fflush(stdout);
        }
        if (tot >= P.timeBudget) break;
    }
    auto t2 = Clock::now();

    ThreadScratch finalScratch; finalScratch.init(n);
    for (int i = 0; i < n; i++) finalScratch.keyed[i] = {gbestX[i], i};
    std::sort(finalScratch.keyed.begin(), finalScratch.keyed.end());
    std::vector<int> bestTour(n);
    for (int i = 0; i < n; i++) bestTour[i] = finalScratch.keyed[i].second;

    std::vector<char> seen(n, 0); bool ok = true; double Lchk = 0;
    for (int s = 0; s < n; s++) {
        int c = bestTour[s]; if (c < 0 || c >= n || seen[c]) { ok = false; break; }
        seen[c] = 1;
        int d = bestTour[(s + 1) % n];
        double dx = (double)I.x[c] - I.x[d], dy = (double)I.y[c] - I.y[d];
        Lchk += std::sqrt(dx * dx + dy * dy);
    }
    double rss = peakRssMB();
    printf("# ---- resumen ----\n");
    printf("n=%d particulas=%d iters=%d tour_valido=%s L_mejor=%.5f (verificado %.5f)\n",
           n, np, itDone, ok ? "SI" : "NO", gbestL, Lchk);
    if (P.exact && n <= 20) {
        double opt = heldKarp(I);
        printf("optimo_exacto(Held-Karp)=%.5f  gap=%.3f%%\n", opt, 100.0 * (gbestL - opt) / opt);
    }
    printf("tiempo: preparacion=%.3f s  PSO=%.3f s  total=%.3f s  (%.4f s/iter)\n",
           secs(t0, t1), secs(tLoop, t2), secs(t0, t2), secs(tLoop, t2) / std::max(1, itDone));
    printf("memoria: pico RSS=%.1f MB\n", rss);
    if (!P.tourOut.empty()) {
        FILE* f = fopen(P.tourOut.c_str(), "w");
        for (int s = 0; s < n; s++) fprintf(f, "%d %.6f %.6f\n", bestTour[s], I.x[bestTour[s]], I.y[bestTour[s]]);
        fclose(f);
    }
    printf("CSV,%d,%d,%d,%.6f,%.3f,%.3f,%.1f\n", n, np, itDone, gbestL, secs(t0, t1), secs(tLoop, t2), rss);
    return 0;
}
