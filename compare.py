import os 
import pandas as pd
from rich.console import Console
from rich.table import Table

def compare_methods(csv_bandit: str, csv_bayes: str, keys: list):
    df_b = pd.read_csv(csv_bandit)
    df_y = pd.read_csv(csv_bayes)

    cols = keys + ['mean_cython', 'speedup_cython']
    b = df_b[cols].copy().rename(columns={
        'mean_cython': 'mean_time_bandit',
        'speedup_cython': 'speedup_bandit'
    })
    y = df_y[cols].copy().rename(columns={
        'mean_cython': 'mean_time_bayes',
        'speedup_cython': 'speedup_bayes'
    })

    df = pd.merge(b, y, on=keys)

    # Calcul des deltas et meilleure méthode
    df['delta_time'] = df['mean_time_bayes'] - df['mean_time_bandit']
    df['best_method'] = df['delta_time'].apply(
        lambda d: 'bayes' if d < 0 else ('bandit' if d > 0 else 'equal')
    )

    # Préparation de l’affichage
    display_cols = (
        keys +
        ['mean_time_bandit', 'mean_time_bayes', 'delta_time', 'best_method']
    )

    # Affichage avec rich
    console = Console()
    table = Table(show_header=True, header_style="bold cyan")
    for name in display_cols:
        table.add_column(name, justify="center")
    for _, row in df.iterrows():
        # ici on utilise les strings de display_cols, pas table.columns
        table.add_row(*[str(row[name]) for name in display_cols])
    console.print(table)

    # Stats agrégées
    total       = len(df)
    bayes_wins  = (df['best_method'] == 'bayes').sum()
    bandit_wins = (df['best_method'] == 'bandit').sum()
    ties        = (df['best_method'] == 'equal').sum()
    avg_diff    = df['delta_time'].mean()
    stats = {
        'total_cases':   total,
        'bayes_wins':    bayes_wins,
        'bandit_wins':   bandit_wins,
        'ties':          ties,
        'pct_bayes':     bayes_wins / total * 100,
        'pct_bandit':    bandit_wins / total * 100,
        'pct_ties':      ties / total * 100,
        'avg_time_diff': avg_diff
    }

    return df, stats

df_comp, summary = compare_methods(
    'results/timings_summary_bandit.csv',
    'results/timings_summary_bayes.csv',
    ['dim']
)

df_summary = pd.DataFrame([summary])
os.makedirs('results', exist_ok=True)
df_summary.to_csv('results/summary_stats.csv', index=False)

print("[INFO] Summary saved to results/summary_stats.csv")