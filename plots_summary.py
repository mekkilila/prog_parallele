import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import LogFormatter
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Plot benchmark summaries (mean time and speedup only).'
    )
    parser.add_argument(
        '--csv', '-c', required=True,
        help='Path to the timings summary CSV file'
    )
    args = parser.parse_args()
    csv_path = args.csv

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # vérification rapide
    required = [
        'dim','dtype',
        'mean_numpy','mean_numba','mean_cython',
        'speedup_numba','speedup_cython'
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    sns.set(style='whitegrid', palette='colorblind', font_scale=1.1)

    for dtype in df['dtype'].unique():
        df_d = df[df['dtype'] == dtype]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

        # Panel 1 : Mean Execution Time
        ax = axes[0]
        sns.lineplot(ax=ax, x='dim', y='mean_numpy',  data=df_d, label='NumPy',  marker='o')
        sns.lineplot(ax=ax, x='dim', y='mean_numba',  data=df_d, label='Numba',  marker='o')
        sns.lineplot(ax=ax, x='dim', y='mean_cython', data=df_d, label='Cython', marker='o')
        ax.set_title('Mean Execution Time')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Time (s)')
        ax.legend()
        ax.grid(True, linestyle='--', linewidth=0.5)

        # Panel 2 : Speedup Over NumPy
        ax = axes[1]
        sns.lineplot(ax=ax, x='dim', y='speedup_numba',  data=df_d, label='Numba',  marker='o')
        sns.lineplot(ax=ax, x='dim', y='speedup_cython', data=df_d, label='Cython', marker='^')
        ax.axhline(1, color='gray', linestyle='--', label='Baseline (NumPy)')
        ax.set_title('Speedup Over NumPy')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Speedup Factor')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(LogFormatter())
        ax.legend()
        ax.grid(True, linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        outname = f'plot_mean_speedup_{dtype}.png'
        fig.savefig(outname, dpi=300)
        print(f'Saved figure: {outname}')
        plt.show()

if __name__ == '__main__':
    main()
