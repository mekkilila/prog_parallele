import os
import pandas as pd
from rich.console import Console
from rich.table import Table

def compare_three(csv_bandit: str, csv_bayes: str, csv_ga: str, keys: list):
    # Chargement
    df_b = pd.read_csv(csv_bandit)
    df_y = pd.read_csv(csv_bayes)
    df_g = pd.read_csv(csv_ga)

    # Sélection & renommage des colonnes d’intérêt
    cols = keys + ['mean_cython', 'speedup_cython']
    b = df_b[cols].copy().rename(columns={
        'mean_cython': 'mean_time_bandit',
        'speedup_cython': 'speedup_bandit'
    })
    y = df_y[cols].copy().rename(columns={
        'mean_cython': 'mean_time_bayes',
        'speedup_cython': 'speedup_bayes'
    })
    g = df_g[cols].copy().rename(columns={
        'mean_cython': 'mean_time_ga',
        'speedup_cython': 'speedup_ga'
    })

    # Fusion successive
    df = b.merge(y, on=keys).merge(g, on=keys)

    # Choix de la meilleure méthode par comparaison des temps
    df['best_method'] = df[['mean_time_bandit', 'mean_time_bayes', 'mean_time_ga']] \
        .idxmin(axis=1) \
        .str.replace('mean_time_', '')

    # Affichage avec rich
    display_cols = keys + [
        'mean_time_bandit', 'speedup_bandit',
        'mean_time_bayes',  'speedup_bayes',
        'mean_time_ga',     'speedup_ga',
        'best_method'
    ]
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    for c in display_cols:
        table.add_column(c, justify="center")
    for _, row in df.iterrows():
        table.add_row(*[f"{row[c]:.2e}" if isinstance(row[c], float) and 'mean_time' in c 
                        else str(row[c]) for c in display_cols])
    console.print(table)

    # Statistiques agrégées
    total = len(df)
    wins = {
        m: (df['best_method'] == m).sum()
        for m in ['bandit','bayes','ga']
    }
    ties = total - sum(wins.values())
    stats = {
        'total_cases': total,
        **{f"{m}_wins": wins[m] for m in wins},
        **{f"pct_{m}": wins[m] / total * 100 for m in wins},
        'ties': ties,
        'pct_ties': ties / total * 100
    }
    return df, stats

if __name__ == "__main__":
    # chemins vers tes fichiers
    csv_bandit = "results/timings_summary_bandit.csv"
    csv_bayes  = "results/timings_summary_bayes.csv"
    csv_ga     = "results/timings_summary_ga.csv"
    keys = ['dim']  # ou ['dim','dtype'] si tu veux comparer par dtype aussi

    df_comp, summary = compare_three(csv_bandit, csv_bayes, csv_ga, keys)

    # Sauvegarde détaillée et résumé
    os.makedirs('results', exist_ok=True)
    df_comp.to_csv('results/comparison_methods.csv', index=False)
    pd.DataFrame([summary]).to_csv('results/comparison_summary.csv', index=False)
    print("[INFO] Détails enregistrés dans results/comparison_methods.csv")
    print("[INFO] Résumé enregistré dans results/comparison_summary.csv")
