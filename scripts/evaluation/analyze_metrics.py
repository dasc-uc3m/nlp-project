import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Set plotting styles
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Base directory for all evaluation results and outputs
BASE_DIR = Path("scripts/evaluation")

def load_model_results(model_name):
    """Load evaluation and token throughput results for a given model."""
    base_path = BASE_DIR / model_name.lower()
    try:
        metrics_df = pd.read_csv(base_path / "evaluation_results_2.csv")
        tks_df = pd.read_csv(base_path / "tks_evaluation_results_2.csv")
        return metrics_df, tks_df
    except FileNotFoundError:
        warnings.warn(f"Warning: Results for '{model_name}' not found.")
        return None, None

def plot_bert_scores(models_data):
    """Plot BERTScore metrics (Precision, Recall, F1) for multiple models."""
    combined_df = pd.concat([
        df.assign(Model=name) for name, df in models_data.items()
    ], ignore_index=True)

    metrics = ['BERTScore_P', 'BERTScore_R', 'BERTScore_F1']
    titles = ['Precision', 'Recall', 'F1 Score']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('BERT Score Comparison Across Models', fontsize=16)

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        if metric not in combined_df.columns:
            continue
        sns.boxplot(data=combined_df, x="Model", y=metric, ax=axes[idx])
        axes[idx].set_title(title)
        axes[idx].set_ylim(0.5, 1.0)

    save_plot(fig, 'bert_scores_comparison.png')

def plot_tokens_per_second(models_tks):
    """Plot average tokens per second with error bars for each model."""
    models = list(models_tks.keys())
    means = [df['Tokens_per_second'].mean() for df in models_tks.values()]
    stds = [df['Tokens_per_second'].std() for df in models_tks.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, means, yerr=stds, capsize=5)

    ax.set_title('Average Tokens per Second by Model', fontsize=14)
    ax.set_ylabel('Tokens per Second', fontsize=12)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45)

    for i, v in enumerate(means):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')

    save_plot(fig, 'tokens_per_second_comparison.png')

def create_summary_table(models_data, models_tks):
    """Generate a CSV summary table of key evaluation metrics for each model."""
    summary = []

    for model_name in models_data:
        metrics_df = models_data[model_name]
        tks_df = models_tks[model_name]

        summary.append({
            'Model': model_name,
            'BERTScore Precision': f"{metrics_df['BERTScore_P'].mean():.3f} ± {metrics_df['BERTScore_P'].std():.3f}",
            'BERTScore Recall': f"{metrics_df['BERTScore_R'].mean():.3f} ± {metrics_df['BERTScore_R'].std():.3f}",
            'BERTScore F1': f"{metrics_df['BERTScore_F1'].mean():.3f} ± {metrics_df['BERTScore_F1'].std():.3f}",
            'Recall@3': f"{metrics_df['Recall@3'].mean():.3f}",
            'MRR': f"{metrics_df['MRR'].mean():.3f}",
            'Avg Tokens/Second': f"{tks_df['Tokens_per_second'].mean():.2f} ± {tks_df['Tokens_per_second'].std():.2f}"
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(BASE_DIR / 'metrics_summary.csv', index=False)
    return summary_df

def save_plot(fig, filename):
    """Save a matplotlib figure to the evaluation directory."""
    fig.tight_layout()
    fig.savefig(BASE_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    models = ['gemma', 'qwen', 'deepseek', 'llama3.2-q_int8']
    models_data = {}
    models_tks = {}

    for model in models:
        metrics_df, tks_df = load_model_results(model)
        if metrics_df is not None and tks_df is not None:
            models_data[model] = metrics_df
            models_tks[model] = tks_df

    if models_data:
        plot_bert_scores(models_data)
    if models_tks:
        plot_tokens_per_second(models_tks)

    if models_data and models_tks:
        summary_df = create_summary_table(models_data, models_tks)
        print("\nMetrics Summary:")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
