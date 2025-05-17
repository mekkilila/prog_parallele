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

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None

from attention_functions.numpy_attention import attention as attention_numpy
from attention_functions.numba_attention import attention as attention_numba
from attention_functions.cython_attention import attention as attention_cython
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import scipy.stats as stats


def configure_resources():
    np.random.seed(42)
    system = platform.system()
    if system == "Linux":
        os.sched_setaffinity(0, {0, 1, 2, 3})
        print("[INFO] CPU affinity set to cores 0-3.")
    elif system == "Windows":
        if psutil:
            try:
                psutil.Process().cpu_affinity([0, 1, 2, 3])
                print("[INFO] CPU affinity set to cores 0-3 via psutil.")
            except Exception as e:
                print(f"[WARNING] Could not set CPU affinity: {e}")
        else:
            print("[INFO] Install psutil or use Task Manager to pin CPU on Windows.")
    elif system == "Darwin":
        max_mb = 2048
        max_bytes = max_mb * 1024**2
        if resource:
            _, hard = resource.getrlimit(resource.RLIMIT_DATA)
            try:
                resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))
                print(f"[INFO] Memory limit set to {max_mb}MB on macOS.")
            except Exception:
                print("[WARNING] Could not enforce memory limit on macOS.")
        print("[INFO] CPU affinity not configurable on macOS via Python.")
    else:
        print(f"[WARNING] OS '{system}' not supported for resource limits.")


def measure(fn, args, warmup=5, repeat=20):
    for _ in range(warmup): fn(*args)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    return times


class Bandit:
    def __init__(self, arms):
        self.arms = arms
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}
    def select_arm(self):
        total = sum(self.counts.values())
        for arm, cnt in self.counts.items():
            if cnt == 0: return arm
        ucb = {arm: self.values[arm] + np.sqrt((2 * np.log(total)) / cnt)
               for arm, cnt in self.counts.items()}
        return max(ucb, key=ucb.get)
    def update(self, arm, reward):
        cnt = self.counts[arm] + 1
        self.values[arm] = ((self.values[arm] * (cnt - 1)) + reward) / cnt
        self.counts[arm] = cnt


class BayesianOptimizer:
    def __init__(self, grid):
        self.grid = grid
        self.X, self.y = [], []
        kernel = Matern(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    def suggest(self):
        if len(self.X) < len(self.grid):
            return self.grid[len(self.X)]
        self.gp.fit(np.array(self.X), np.array(self.y))
        def EI(x, xi=0.01):
            mu, sigma = self.gp.predict(x.reshape(1,-1), return_std=True)
            best = min(self.y)
            imp = best - mu - xi
            Z = imp/sigma
            return imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
        eis = [EI(np.array(p)) for p in self.grid]
        return self.grid[int(np.argmax(eis))]
    def update(self, params, reward):
        self.X.append(params)
        self.y.append(reward)


def evaluate_configuration(Q, K, V, config):
    nt, bs, dt = config
    try:
        q, k, v = Q.astype(dt), K.astype(dt), V.astype(dt)
        times = measure(attention_cython, (q, k, v, nt, bs), warmup=3, repeat=5)
        return mean(times)
    except Exception as e:
        print(f"[WARNING] Config {config} failed: {e}")
        return float('inf')


def adaptive_search(Q, K, V, blocks, threads, dtypes, method='bandit', max_iters=30):
    grid = [(nt, bs, dt) for nt in threads for bs in blocks for dt in dtypes]
    if method=='bandit':
        algo = Bandit(grid)
    elif method=='bayes':
        mapping = {np.float32:0, np.float64:1}
        inv = {v:k for k,v in mapping.items()}
        numeric = [[nt, bs, mapping[dt]] for nt, bs, dt in grid]
        algo = BayesianOptimizer(numeric)
    else:
        raise ValueError(f"Unknown method '{method}'")
    best, best_t = None, float('inf')
    for _ in range(max_iters):
        cfg = algo.select_arm() if method=='bandit' else algo.suggest()
        if method=='bayes': cfg = (cfg[0], cfg[1], inv[cfg[2]])
        t = evaluate_configuration(Q, K, V, cfg)
        reward = -t if method=='bandit' else t
        update_arg = cfg if method=='bandit' else [cfg[0],cfg[1],mapping[cfg[2]]]
        algo.update(update_arg, reward)
        if t < best_t:
            best_t, best = t, cfg
    tag = 'Bandit' if method=='bandit' else 'Bayes'
    print(f"[INFO][{tag}] Best config: {best} in {best_t:.6f}s")
    return best


def run_benchmark(dims=None, blocks=None, threads=None, dtypes=None,
                  repeat=20, warmup=5, method='bandit', max_iters=30, out_dir='results'):
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
        best_nt, best_bs, best_dt = adaptive_search(Q, K, V, blocks, threads, dtypes, method, max_iters)
        Q = np.random.rand(dim,dim).astype(best_dt)
        K = np.random.rand(dim,dim).astype(best_dt)
        V = np.random.rand(dim,dim).astype(best_dt)
        times_numpy  = measure(attention_numpy, (Q,K,V), warmup=1, repeat=10)
        times_numba  = measure(attention_numba, (Q,K,V), warmup=3, repeat=10)
        times_cython = measure(attention_cython, (Q,K,V,best_nt,best_bs), warmup=3, repeat=10)
        ok = False
        try:
            ok = np.allclose(attention_numpy(Q,K,V), attention_numba(Q,K,V), atol=1e-5, rtol=1e-3)
            ok &= np.allclose(attention_numpy(Q,K,V), attention_cython(Q,K,V,best_nt,best_bs), atol=1e-5, rtol=1e-3)
        except:
            pass
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
            'speedup_cython': mean(times_numpy)/mean(times_cython),
            'all_close': ok
        }
        records.append(record)
    df = pd.DataFrame(records)
    path = os.path.join(out_dir, f"timings_summary_{method}.csv")
    df.to_csv(path, index=False)
    print(f"\n[INFO] Results saved to {path}")
    # display summary table
    df_disp = df.copy()
    for m in ['numpy','numba','cython']:
        df_disp[m] = df_disp[f'mean_{m}'].map(lambda x: f"{x:.2e}")
    cols = ['dim','dtype','block_size','threads','numpy','numba','cython','speedup_numba','speedup_cython','all_close']
    df_disp = df_disp[cols]
    console = Console()
    table = Table(show_header=True, header_style="bold green", box=box.SIMPLE)
    for c in cols:
        table.add_column(c, justify="center")
    for _, row in df_disp.iterrows():
        table.add_row(*map(str, row))
    console.print(table)


def main():
    configure_resources()
    parser = argparse.ArgumentParser(description="Benchmark attention methods")
    parser.add_argument('--method', choices=['bandit','bayes'], default='bandit')
    args = parser.parse_args()
    run_benchmark(method=args.method)


if __name__ == '__main__':
    main()
