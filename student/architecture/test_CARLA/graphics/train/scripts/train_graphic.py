import os
import argparse
import json
import glob

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 14,
    "lines.linewidth": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================"/home/ana/Documents/Architecture_Transformers_SR/student/Arquitetura/Resultados/phase2/Proto-AW_ckpt14/train_metrics.json
# FUNÇÕES AUXILIARES
# ============================================================

def find_default_metrics_file():
    candidates = [
        "/home/ana/Documents/Architecture_Transformers_SR/student/Arquitetura/Resultados/phase2/Ours_ckpt20/train_metrics.csv",
        "/home/ana/Documents/Architecture_Transformers_SR/student/Arquitetura/Resultados/phase2/Ours_ckpt20/train_metrics.json",
        "../train_metrics.csv",
        "../train_metrics.json",
        "../../train_metrics.csv",
        "../../train_metrics.json",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    found = glob.glob("**/train_metrics.*", recursive=True)
    if found:
        return found[0]

    raise FileNotFoundError(
        "Não encontrei train_metrics.csv ou train_metrics.json. "
        "Passe o caminho usando --metrics CAMINHO_DO_ARQUIVO."
    )


def load_metrics(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)

    if ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            for key in ["metrics", "history", "train_metrics", "rows"]:
                if key in data:
                    return pd.DataFrame(data[key])

            return pd.DataFrame(data)

    raise ValueError(f"Formato não suportado: {path}")


def is_valid_metric(df, col, eps=1e-8):
    if col not in df.columns:
        return False

    values = pd.to_numeric(df[col], errors="coerce")

    if values.isna().all():
        return False

    if values.abs().max() <= eps:
        return False

    return True


def plot_metrics(df, x_col, metrics, title, ylabel, output_path):
    metrics = [m for m in metrics if is_valid_metric(df, m)]

    if not metrics:
        print(f"[SKIP] Nenhuma métrica válida para: {title}")
        return

    plt.figure(figsize=(10.0, 6.0))

    for metric in metrics:
        plt.plot(df[x_col], df[metric], label=metric, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    # plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    pdf_path = output_path + ".pdf"
    png_path = output_path + ".png"

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Caminho para train_metrics.csv ou train_metrics.json"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="train_figures",
        help="Pasta onde as figuras serão salvas"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="training",
        help="Prefixo dos arquivos de saída"
    )

    args = parser.parse_args()

    metrics_path = args.metrics if args.metrics is not None else find_default_metrics_file()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_metrics(metrics_path)

    print("\nArquivo carregado:", metrics_path)
    print("\nColunas encontradas:")
    print(df.columns)

    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)

    x_col = "epoch"

    # Converter colunas numéricas quando possível
    for col in df.columns:
        if col != x_col:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # Mostrar métricas zeradas
    zero_metrics = []
    for col in df.columns:
        if col == x_col:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if not values.isna().all() and values.abs().max() <= 1e-8:
            zero_metrics.append(col)

    if zero_metrics:
        print("\nMétricas zeradas que serão ignoradas:")
        for m in zero_metrics:
            print(" -", m)

    # ========================================================
    # GRÁFICO 1: Loss principal de treinamento
    # ========================================================

    main_losses = [
        "total_loss",
        "distill_mse",
        "action_mae",
        "speed_mse",
        "steer_mse",
    ]

    plot_metrics(
        df=df,
        x_col=x_col,
        metrics=main_losses,
        title="Training Losses",
        ylabel="Loss value",
        output_path=os.path.join(args.outdir, f"{args.prefix}_main_losses")
    )

    # ========================================================
    # GRÁFICO 2: Router / MoE
    # ========================================================

    router_metrics = [
        "router_balance_loss",
        "router_entropy",
        "old_task_new_expert_penalty",
        "new_task_old_expert_penalty",
    ]

    plot_metrics(
        df=df,
        x_col=x_col,
        metrics=router_metrics,
        title="Router and Expert Regularization",
        ylabel="Metric value",
        output_path=os.path.join(args.outdir, f"{args.prefix}_router_metrics")
    )

    # ========================================================
    # GRÁFICO 3: Geo / Interaction / Task losses
    # ========================================================

    contrastive_metrics = [
        "geo_contrastive_loss",
        "int_contrastive_loss",
        "task_contrastive_loss",
        "task_alignment_loss",
    ]

    plot_metrics(
        df=df,
        x_col=x_col,
        metrics=contrastive_metrics,
        title="Geometric, Interaction and Task Losses",
        ylabel="Loss value",
        output_path=os.path.join(args.outdir, f"{args.prefix}_contrastive_losses")
    )

    # ========================================================
    # GRÁFICO 4: Pesos das losses, se existirem
    # ========================================================

    weight_metrics = [
        "router_balance_weight",
        "router_entropy_weight",
    ]

    plot_metrics(
        df=df,
        x_col=x_col,
        metrics=weight_metrics,
        title="Regularization Weights",
        ylabel="Weight value",
        output_path=os.path.join(args.outdir, f"{args.prefix}_regularization_weights")
    )

    print("\nFinalizado.")


if __name__ == "__main__":
    main()