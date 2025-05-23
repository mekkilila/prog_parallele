# import os
# import platform
# import time
# import argparse
# import random

# import numpy as np
# import pandas as pd
# from statistics import mean, stdev

# from rich import box
# from rich.console import Console
# from rich.table import Table

# try:
#     import psutil
# except ImportError:
#     psutil = None

# try:
#     import resource
# except ImportError:
#     resource = None

# from attention_functions.numpy_attention import attention as attention_numpy
# from attention_functions.numba_attention import attention as attention_numba
# from attention_functions.cython_attention import attention as attention_cython
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import Matern
# import scipy.stats as stats


# def configure_resources():
#     np.random.seed(42)
#     system = platform.system()
#     if system == "Linux":
#         os.sched_setaffinity(0, {0, 1, 2, 3})
#         print("[INFO] CPU affinity set to cores 0-3.")
#     elif system == "Windows":
#         if psutil:
#             try:
#                 psutil.Process().cpu_affinity([0, 1, 2, 3])
#                 print("[INFO] CPU affinity set to cores 0-3 via psutil.")
#             except Exception as e:
#                 print(f"[WARNING] Could not set CPU affinity: {e}")
#         else:
#             print("[INFO] Install psutil or use Task Manager to pin CPU on Windows.")
#     elif system == "Darwin":
#         max_mb = 2048
#         max_bytes = max_mb * 1024**2
#         if resource:
#             _, hard = resource.getrlimit(resource.RLIMIT_DATA)
#             try:
#                 resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
#                 print(f"[INFO] Memory limit set to {max_mb}MB on macOS.")
#             except Exception:
#                 print("[WARNING] Could not enforce memory limit on macOS.")
#         print("[INFO] CPU affinity not configurable on macOS via Python.")
#     else:
#         print(f"[WARNING] OS '{system}' not supported for resource limits.")


# def measure(fn, args, warmup=5, repeat=20):
#     for _ in range(warmup): fn(*args)
#     times = []
#     for _ in range(repeat):
#         start = time.perf_counter()
#         fn(*args)
#         times.append(time.perf_counter() - start)
#     return times


# # class Bandit:
# #     def __init__(self, arms):
# #         self.arms = arms
# #         self.counts = {arm: 0 for arm in arms}
# #         self.values = {arm: 0.0 for arm in arms}
# #     def select_arm(self):
# #         total = sum(self.counts.values())
# #         for arm, cnt in self.counts.items():
# #             if cnt == 0: return arm
# #         ucb = {arm: self.values[arm] + np.sqrt((2 * np.log(total)) / cnt)
# #                for arm, cnt in self.counts.items()}
# #         return max(ucb, key=ucb.get)
# #     def update(self, arm, reward):
# #         cnt = self.counts[arm] + 1
# #         self.values[arm] = ((self.values[arm] * (cnt - 1)) + reward) / cnt
# #         self.counts[arm] = cnt


# # class BayesianOptimizer:
# #     def __init__(self, grid):
# #         self.grid = grid
# #         self.X, self.y = [], []
# #         kernel = Matern(length_scale=1.0)
# #         self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
# #     def suggest(self):
# #         if len(self.X) < len(self.grid):
# #             return self.grid[len(self.X)]
# #         self.gp.fit(np.array(self.X), np.array(self.y))
# #         def EI(x, xi=0.01):
# #             mu, sigma = self.gp.predict(x.reshape(1,-1), return_std=True)
# #             best = min(self.y)
# #             imp = best - mu - xi
# #             Z = imp/sigma
# #             return imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
# #         eis = [EI(np.array(p)) for p in self.grid]
# #         return self.grid[int(np.argmax(eis))]
# #     def update(self, params, reward):
# #         self.X.append(params)
# #         self.y.append(reward)

# class Bandit:
#     def __init__(self, arms, alpha=1.0):
#         self.arms = arms
#         self.counts = {arm: 0 for arm in arms}
#         self.values = {arm: 0.0 for arm in arms}
#         self.alpha = alpha

#     def select_arm(self):
#         total = sum(self.counts.values()) + 1
#         # phase d'initialisation
#         for arm in self.arms:
#             if self.counts[arm] == 0:
#                 return arm
#         # UCB pondéré
#         ucb = {
#             arm: self.values[arm]
#                  - self.alpha * np.sqrt(np.log(total) / self.counts[arm])
#             for arm in self.arms
#         }
#         return min(ucb, key=ucb.get)  # on minimise le temps

#     def update(self, arm, reward):
#         cnt = self.counts[arm] + 1
#         self.values[arm] = (self.values[arm] * self.counts[arm] + reward) / cnt
#         self.counts[arm] = cnt

# from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# class BayesianOptimizer:
#     def __init__(self, bounds, dtype_choices, n_starts=10):
#         """
#         bounds: [(min_threads, max_threads), (min_blocks, max_blocks)]
#         dtype_choices: list of numpy dtypes (e.g. [np.float32, np.float64])
#         n_starts: number of pure-random initial evaluations
#         """
#         self.bounds = bounds
#         self.dtype_choices = list(dtype_choices)
#         self.n_starts = n_starts
#         self.X, self.y = [], []

#         # isotropic Matern + white noise: length_scale scalar applies to all 3 dims
#         kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)
#         self.gp = GaussianProcessRegressor(kernel=kernel,
#                                            normalize_y=True,
#                                            n_restarts_optimizer=2)

#     def suggest(self):
#         # during initial random warm-up
#         if len(self.X) < self.n_starts:
#             nt = random.randint(*self.bounds[0])
#             bs = random.randint(*self.bounds[1])
#             dt_idx = random.randrange(len(self.dtype_choices))
#             return [nt, bs, dt_idx]

#         # fit GP on all seen data
#         X = np.array(self.X)
#         y = np.array(self.y)
#         self.gp.fit(X, y)

#         # acquisition = Lower Confidence Bound (we minimize time)
#         def acquisition(x, kappa=2.0):
#             mu, sigma = self.gp.predict(x.reshape(1,-1), return_std=True)
#             return mu - kappa * sigma

#         # random candidate search on a small budget
#         best_val = float('inf')
#         best_pt  = None
#         for _ in range(100):
#             nt = random.randint(*self.bounds[0])
#             bs = random.randint(*self.bounds[1])
#             dt_idx = random.randrange(len(self.dtype_choices))
#             x = np.array([nt, bs, dt_idx])
#             val = acquisition(x)
#             if val < best_val:
#                 best_val, best_pt = val, (nt, bs, dt_idx)

#         return list(best_pt)

#     def update(self, params, reward):
#         # params: [nt, bs, dt_idx], reward = observed time (we minimize)
#         self.X.append(params)
#         self.y.append(reward)


# class GeneticOptimizer:
#     def __init__(self, blocks, threads, dtypes,
#                  pop_size=60, elite_frac=0.3, mut_rate=0.05):
#         self.blocks = sorted(blocks)
#         self.threads = sorted(threads)
#         self.dtypes = list(dtypes)
#         self.pop_size = pop_size
#         self.elite_frac = elite_frac
#         self.mut_rate = mut_rate

#         self.population = [
#             (random.choice(self.threads),
#              random.choice(self.blocks),
#              random.choice(self.dtypes))
#             for _ in range(pop_size)
#         ]

#     def fitnesses(self, evaluator):
#         out = []
#         for cfg in self.population:
#             t = evaluator(cfg)
#             out.append(( -t, cfg ))
#         return out

#     def select_elite(self, fit_list):
#         n_elite = max(2, int(self.elite_frac * self.pop_size))
#         sorted_ = sorted(fit_list, key=lambda x: x[0], reverse=True)
#         return [cfg for _, cfg in sorted_[:n_elite]]

#     def crossover(self, parents):
#         children = []
#         while len(children) < self.pop_size - len(parents):
#             p1, p2 = random.sample(parents, 2)
#             child = (
#                 random.choice([p1[0], p2[0]]),
#                 random.choice([p1[1], p2[1]]),
#                 random.choice([p1[2], p2[2]])
#             )
#             children.append(child)
#         return children

#     def mutate(self, population):
#         newpop = []
#         for nt, bs, dt in population:
#             if random.random() < self.mut_rate:
#                 idx = self.blocks.index(bs)
#                 idx = min(len(self.blocks)-1,
#                           max(0, idx + random.choice([-1,1])))
#                 bs = self.blocks[idx]
#             if random.random() < self.mut_rate:
#                 idx = self.threads.index(nt)
#                 idx = min(len(self.threads)-1,
#                           max(0, idx + random.choice([-1,1])))
#                 nt = self.threads[idx]
#             if random.random() < self.mut_rate:
#                 dt = random.choice(self.dtypes)
#             newpop.append((nt, bs, dt))
#         return newpop

#     def run(self, evaluator, generations=50):
#         best_cfg, best_fit = None, float('-inf')
#         for gen in range(generations):
#             fit_list = self.fitnesses(evaluator)
#             elite = self.select_elite(fit_list)
#             if fit_list[0][0] > best_fit:
#                 best_fit, best_cfg = fit_list[0]
#             children = self.crossover(elite)
#             self.population = elite + self.mutate(children)
#         return best_cfg


# def evaluate_configuration(Q, K, V, config):
#     nt, bs, dt = config
#     try:
#         q, k, v = Q.astype(dt), K.astype(dt), V.astype(dt)
#         times = measure(attention_cython, (q, k, v, nt, bs), warmup=3, repeat=5)
#         return mean(times)
#     except Exception as e:
#         print(f"[WARNING] Config {config} failed: {e}")
#         return float('inf')


# def adaptive_search(Q, K, V, blocks, threads, dtypes,
#                     method='bandit', max_iters=30):

#     grid = [(nt, bs, dt) for nt in threads for bs in blocks for dt in dtypes]

#     if method == 'bandit':
#         algo = Bandit(grid)

#     elif method == 'bayes':
#         bounds = [
#             (min(threads), max(threads)),
#             (min(blocks),  max(blocks))
#         ]
#         dtype_choices = list(dtypes)

#         algo = BayesianOptimizer(bounds, dtype_choices, n_starts=10)

#     elif method == 'ga':
#         algo = GeneticOptimizer(blocks, threads, dtypes,
#                                  pop_size=60, elite_frac=0.3, mut_rate=0.05)

#     else:
#         raise ValueError(f"Unknown method '{method}'")

#     best_cfg, best_t = None, float('inf')

#     if method == 'bandit':
#         for _ in range(max_iters):
#             cfg = algo.select_arm()
#             t = evaluate_configuration(Q, K, V, cfg)
#             algo.update(cfg, -t)     # store negative time as reward
#             if t < best_t:
#                 best_t, best_cfg = t, cfg

#     elif method == 'bayes':
#         for _ in range(max_iters):
#             nt, bs, idx_dt = algo.suggest()
#             cfg = (nt, bs, dtype_choices[idx_dt])

#             t = evaluate_configuration(Q, K, V, cfg)
#             algo.update([nt, bs, idx_dt], t)

#             if t < best_t:
#                 best_t, best_cfg = t, cfg

#     else:  # GA
#         best_cfg = algo.run(lambda cfg: evaluate_configuration(Q,K,V,cfg),
#                              generations=max_iters)
#         best_t   = evaluate_configuration(Q, K, V, best_cfg)

#     tag = {'bandit':'Bandit', 'bayes':'Bayes', 'ga':'Genetic'}[method]
#     print(f"[INFO][{tag}] Best config: {best_cfg} in {best_t:.6f}s")
#     return best_cfg


# def run_benchmark(dims=None, blocks=None, threads=None, dtypes=None,
#                   repeat=20, warmup=5, method='bandit', max_iters=30, out_dir='results'):
#     dims    = dims    or [64,128,256,400,512,768,1024]
#     threads = threads or [1,2,4,8]
#     blocks  = blocks  or [8,16,32,64]
#     dtypes  = dtypes  or [np.float32, np.float64]
#     os.makedirs(out_dir, exist_ok=True)
#     records = []
#     for dim in dims:
#         print(f"\n[INFO] Running for dimension {dim}")
#         Q = np.random.rand(dim,dim).astype(np.float32)
#         K = np.random.rand(dim,dim).astype(np.float32)
#         V = np.random.rand(dim,dim).astype(np.float32)
#         best_nt, best_bs, best_dt = adaptive_search(Q, K, V, blocks, threads, dtypes, method, max_iters)
#         Q = np.random.rand(dim,dim).astype(best_dt)
#         K = np.random.rand(dim,dim).astype(best_dt)
#         V = np.random.rand(dim,dim).astype(best_dt)
#         times_numpy  = measure(attention_numpy, (Q,K,V), warmup=1, repeat=10)
#         times_numba  = measure(attention_numba, (Q,K,V), warmup=3, repeat=10)
#         times_cython = measure(attention_cython, (Q,K,V,best_nt,best_bs), warmup=3, repeat=10)
#         ok = False
#         try:
#             ok = np.allclose(attention_numpy(Q,K,V), attention_numba(Q,K,V), atol=1e-5, rtol=1e-3)
#             ok &= np.allclose(attention_numpy(Q,K,V), attention_cython(Q,K,V,best_nt,best_bs), atol=1e-5, rtol=1e-3)
#         except:
#             pass
#         record = {
#             'dim': dim,
#             'dtype': best_dt.__name__,
#             'block_size': best_bs,
#             'threads': best_nt,
#             'mean_numpy': mean(times_numpy),
#             'stdev_numpy': stdev(times_numpy),
#             'mean_numba': mean(times_numba),
#             'stdev_numba': stdev(times_numba),
#             'mean_cython': mean(times_cython),
#             'stdev_cython': stdev(times_cython),
#             'speedup_numba': mean(times_numpy)/mean(times_numba),
#             'speedup_cython': mean(times_numpy)/mean(times_cython),
#             'all_close': ok
#         }
#         records.append(record)
#     df = pd.DataFrame(records)
#     path = os.path.join(out_dir, f"timings_summary_{method}.csv")
#     df.to_csv(path, index=False)
#     print(f"\n[INFO] Results saved to {path}")
#     # display summary table
#     df_disp = df.copy()
#     for m in ['numpy','numba','cython']:
#         df_disp[m] = df_disp[f'mean_{m}'].map(lambda x: f"{x:.2e}")
#     cols = ['dim','dtype','block_size','threads','numpy','numba','cython','speedup_numba','speedup_cython','all_close']
#     df_disp = df_disp[cols]
#     console = Console()
#     table = Table(show_header=True, header_style="bold green", box=box.SIMPLE)
#     for c in cols:
#         table.add_column(c, justify="center")
#     for _, row in df_disp.iterrows():
#         table.add_row(*map(str, row))
#     console.print(table)


# def main():
#     configure_resources()
#     parser = argparse.ArgumentParser(description="Benchmark attention methods")
#     parser.add_argument(
#         '--method', choices=['bandit', 'bayes', 'ga'],
#         default='bandit',
#         help="Méthode d'optimisation: bandit, bayes ou ga"
#     )
#     parser.add_argument(
#         '--max-iters', type=int, default=30,
#         help="Nombre d'itérations (ou de générations pour GA)"
#     )
#     args = parser.parse_args()
#     run_benchmark(
#         method=args.method,
#         max_iters=args.max_iters
#     )

# if __name__ == '__main__':
#     main()

import os
import platform
import time
import argparse
import random

import numpy as np
import pandas as pd
from statistics import mean, stdev

from rich import box
from rich.console import Console
from rich.table import Table
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import scipy.stats as stats

# Attention implementation imports
from attention_functions.numpy_attention import attention as attention_numpy
from attention_functions.numba_attention import attention as attention_numba
from attention_functions.cython_attention import attention as attention_cython

# ----------------------------------------
# Resource configuration and measurement
# ----------------------------------------
try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None


def configure_resources():
    np.random.seed(42)
    system = platform.system()
    if system == "Linux":
        # Pin to all available cores
        os.sched_setaffinity(0, set(range(os.cpu_count())))
        print(f"[INFO] CPU affinity set to all {os.cpu_count()} cores.")
    elif system == "Darwin":
        print("[INFO] macOS: manual pinning recommended via taskset or similar.")
    else:
        print(f"[WARNING] OS '{system}' resource configuration skipped.")


def measure(fn, args, warmup=5, repeat=20):
    # ensure contiguous arrays
    args = tuple(np.ascontiguousarray(a) for a in args)
    # Numba JIT-first call warmup
    if fn is attention_numba:
        fn(*args)
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return times

# -----------------
# Bandit (UCB)
# -----------------
class Bandit:
    def __init__(self, arms, alpha=1.0):
        self.arms = arms
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}
        self.alpha = alpha

    def select_arm(self):
        total = sum(self.counts.values()) + 1
        for arm, cnt in self.counts.items():
            if cnt == 0:
                return arm
        ucb = {arm: self.values[arm] - self.alpha * np.sqrt(np.log(total) / cnt)
               for arm, cnt in self.counts.items()}
        return min(ucb, key=ucb.get)

    def update(self, arm, reward):
        cnt = self.counts[arm] + 1
        self.values[arm] = (self.values[arm] * self.counts[arm] + reward) / cnt
        self.counts[arm] = cnt

# -----------------
# Bayesian Optimization
# -----------------
class BayesianOptimizer:
    def __init__(self, bounds, dtype_choices, n_starts=20):
        self.bounds = bounds
        self.dtype_choices = list(dtype_choices)
        self.n_starts = n_starts
        self.X, self.y = [], []
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=3
        )

    def suggest(self):
        if len(self.X) < self.n_starts:
            nt = random.randint(*self.bounds[0])
            bs = random.randint(*self.bounds[1])
            dt_idx = random.randrange(len(self.dtype_choices))
            return [nt, bs, dt_idx]
        X = np.array(self.X)
        y = np.array(self.y)
        self.gp.fit(X, y)
        def acq(x, kappa=2.0):
            mu, sigma = self.gp.predict(x.reshape(1,-1), return_std=True)
            return mu - kappa * sigma
        best_val = float('inf')
        best_pt = None
        for _ in range(200):
            nt = random.randint(*self.bounds[0])
            bs = random.randint(*self.bounds[1])
            dt_idx = random.randrange(len(self.dtype_choices))
            x = np.array([nt, bs, dt_idx])
            val = acq(x)
            if val < best_val:
                best_val, best_pt = val, (nt, bs, dt_idx)
        return list(best_pt)

    def update(self, params, reward):
        self.X.append(params)
        self.y.append(reward)

# -----------------
# Genetic Algorithm
# -----------------
class GeneticOptimizer:
    def __init__(self, blocks, threads, dtypes,
                 pop_size=100, elite_frac=0.3, mut_rate=0.05):
        self.blocks = sorted(blocks)
        self.threads = sorted(threads)
        self.dtypes = list(dtypes)
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.mut_rate = mut_rate
        self.population = [
            (random.choice(self.threads), random.choice(self.blocks), random.choice(self.dtypes))
            for _ in range(pop_size)
        ]

    def fitnesses(self, evaluator):
        return [(-evaluator(cfg), cfg) for cfg in self.population]

    def select_elite(self, fit_list):
        n_elite = max(2, int(self.elite_frac * self.pop_size))
        sorted_ = sorted(fit_list, key=lambda x: x[0], reverse=True)
        return [cfg for _, cfg in sorted_[:n_elite]]

    def crossover(self, parents):
        children = []
        while len(children) < self.pop_size - len(parents):
            p1, p2 = random.sample(parents, 2)
            children.append((
                random.choice([p1[0], p2[0]]),
                random.choice([p1[1], p2[1]]),
                random.choice([p1[2], p2[2]])
            ))
        return children

    def mutate(self, population):
        newpop = []
        for nt, bs, dt in population:
            if random.random() < self.mut_rate:
                idx = self.blocks.index(bs)
                bs = self.blocks[min(len(self.blocks)-1, max(0, idx + random.choice([-1,1])))]
            if random.random() < self.mut_rate:
                idx = self.threads.index(nt)
                nt = self.threads[min(len(self.threads)-1, max(0, idx + random.choice([-1,1])))]
            if random.random() < self.mut_rate:
                dt = random.choice(self.dtypes)
            newpop.append((nt, bs, dt))
        return newpop

    def run(self, evaluator, generations=100):
        best_cfg, best_fit = None, float('-inf')
        for gen in range(generations):
            fit_list = self.fitnesses(evaluator)
            elite = self.select_elite(fit_list)
            if fit_list[0][0] > best_fit:
                best_fit, best_cfg = fit_list[0]
            children = self.crossover(elite)
            self.population = elite + self.mutate(children)
        return best_cfg

# -----------------
# Core benchmark logic
# -----------------

def evaluate_configuration(Q, K, V, config):
    nt, bs, dt = config
    try:
        q, k, v = (np.ascontiguousarray(arr.astype(dt)) for arr in (Q, K, V))
        times = measure(attention_cython, (q, k, v, nt, bs), warmup=5, repeat=20)
        return mean(times)
    except Exception as e:
        print(f"[WARNING] Config {config} failed: {e}")
        return float('inf')


def adaptive_search(Q, K, V, blocks, threads, dtypes,
                    method='bandit', max_iters=100):
    grid = [(nt, bs, dt) for nt in threads for bs in blocks for dt in dtypes]
    if method == 'bandit':
        algo = Bandit(grid, alpha=1.0)
    elif method == 'bayes':
        bounds = [(min(threads), max(threads)), (min(blocks), max(blocks))]
        dtype_choices = list(dtypes)
        algo = BayesianOptimizer(bounds, dtype_choices, n_starts=20)
    elif method == 'ga':
        algo = GeneticOptimizer(blocks, threads, dtypes,
                                 pop_size=100, elite_frac=0.3, mut_rate=0.05)
    else:
        raise ValueError(f"Unknown method '{method}'")

    best_cfg, best_t = None, float('inf')
    if method == 'bandit':
        for _ in range(max_iters):
            cfg = algo.select_arm()
            t = evaluate_configuration(Q, K, V, cfg)
            algo.update(cfg, -t)
            if t < best_t:
                best_t, best_cfg = t, cfg
    elif method == 'bayes':
        for _ in range(max_iters):
            nt, bs, idx_dt = algo.suggest()
            cfg = (nt, bs, dtypes[idx_dt])
            t = evaluate_configuration(Q, K, V, cfg)
            algo.update([nt, bs, idx_dt], t)
            if t < best_t:
                best_t, best_cfg = t, cfg
    else:  # GA
        best_cfg = algo.run(lambda cfg: evaluate_configuration(Q, K, V, cfg),
                             generations=max_iters)
        best_t = evaluate_configuration(Q, K, V, best_cfg)

    tag = {'bandit': 'Bandit', 'bayes': 'Bayes', 'ga': 'Genetic'}[method]
    print(f"[INFO][{tag}] Best config: {best_cfg} in {best_t:.6f}s")
    return best_cfg


def run_benchmark(dims=None, blocks=None, threads=None, dtypes=None,
                  repeat=20, warmup=5, method='bandit', max_iters=100, out_dir='results'):
    dims    = dims    or [64,128,256,400,512,768,1024]
    threads = threads or [1,2,4,8]
    blocks  = blocks  or [8,16,32,64]
    dtypes  = dtypes  or [np.float32, np.float64]
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for dim in dims:
        print(f"\n[INFO] Running for dimension {dim}")
        Q = np.random.rand(dim,dim).astype(np.float32)
        K = np.random.rand(dim,dim).astype(np.float32)
        V = np.random.rand(dim,dim).astype(np.float32)
        best_nt, best_bs, best_dt = adaptive_search(
            Q, K, V, blocks, threads, dtypes, method, max_iters)
        Qf = Q.astype(best_dt)
        Kf = K.astype(best_dt)
        Vf = V.astype(best_dt)
        times_numpy  = measure(attention_numpy, (Qf, Kf, Vf), warmup, repeat)
        times_numba  = measure(attention_numba, (Qf, Kf, Vf), warmup, repeat)
        times_cython = measure(attention_cython, (Qf, Kf, Vf, best_nt, best_bs), warmup, repeat)
        record = {
            'dim': dim,
            'dtype': best_dt.__name__,
            'block_size': best_bs,
            'threads': best_nt,
            'mean_numpy': mean(times_numpy),
            'stdev_numpy': stdev(times_numpy),
            'mean_numba': mean(times_numba),
            'stdev_numba': stdev(times_numba),
            'mean_cython': mean(times_cython),
            'stdev_cython': stdev(times_cython),
            'speedup_numba': mean(times_numpy)/mean(times_numba),
            'speedup_cython': mean(times_numpy)/mean(times_cython)
        }
        records.append(record)
    df = pd.DataFrame(records)
    path = os.path.join(out_dir, f"timings_summary_{method}.csv")
    df.to_csv(path, index=False)
    console = Console()
    table = Table(show_header=True, header_style="bold green", box=box.SIMPLE)
    cols = list(df.columns)
    for c in cols:
        table.add_column(c, justify="center")
    for _, row in df.iterrows():
        table.add_row(*[f"{row[c]:.2e}" if isinstance(row[c], float) else str(row[c]) for c in cols])
    console.print(table)


def main():
    configure_resources()
    parser = argparse.ArgumentParser(description="Benchmark attention methods")
    parser.add_argument('--method', choices=['bandit','bayes','ga'], default='bandit')
    parser.add_argument('--max-iters', type=int, default=100)
    args = parser.parse_args()
    run_benchmark(method=args.method, max_iters=args.max_iters)

if __name__ == '__main__':
    main()
