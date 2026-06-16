# import os
# import argparse
# import json

# import pandas as pd
# import matplotlib.pyplot as plt


# # ============================================================
# # CONFIGURAÇÃO VISUAL PARA ARTIGO
# # ============================================================

# plt.rcParams.update({
#     "font.size": 8,
#     "axes.labelsize": 9,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "legend.fontsize": 6.8,
#     "lines.linewidth": 1.7,
#     "axes.linewidth": 0.8,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

# # plt.rcParams.update({
# #     "font.size": 16,
# #     "axes.labelsize": 20,
# #     "xtick.labelsize": 15,
# #     "ytick.labelsize": 15,
# #     "legend.fontsize": 11,
# #     "lines.linewidth": 3.0,
# #     "pdf.fonttype": 42,
# #     "ps.fonttype": 42,
# # })


# # ============================================================
# # FUNÇÕES AUXILIARES
# # ============================================================

# def load_metrics(metrics_path):
#     if not os.path.exists(metrics_path):
#         raise FileNotFoundError(f"Arquivo não encontrado: {metrics_path}")

#     ext = os.path.splitext(metrics_path)[1].lower()

#     if ext == ".csv":
#         return pd.read_csv(metrics_path)

#     if ext == ".json":
#         with open(metrics_path, "r") as f:
#             data = json.load(f)

#         if isinstance(data, list):
#             return pd.DataFrame(data)

#         if isinstance(data, dict):
#             for key in ["metrics", "history", "train_metrics", "rows"]:
#                 if key in data:
#                     return pd.DataFrame(data[key])
#             return pd.DataFrame(data)

#     raise ValueError("Use um arquivo .csv ou .json")


# def safe_plot(ax, df, x_col, metric, label=None):
#     if metric not in df.columns:
#         print(f"[SKIP] Coluna não encontrada: {metric}")
#         return False

#     values = pd.to_numeric(df[metric], errors="coerce")

#     if values.isna().all():
#         print(f"[SKIP] Coluna inválida: {metric}")
#         return False

#     if values.abs().max() < 1e-10:
#         print(f"[SKIP] Coluna zerada: {metric}")
#         return False

#     ax.plot(df[x_col], values, label=label if label else metric)
#     return True


# # ============================================================
# # MAIN
# # ============================================================

# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--metrics",
#         type=str,
#         required=True,
#         help="Caminho para train_metrics.csv ou train_metrics.json"
#     )

#     parser.add_argument(
#         "--outdir",
#         type=str,
#         default="train_figures",
#         help="Pasta de saída"
#     )

#     parser.add_argument(
#         "--name",
#         type=str,
#         default="proto_girp_training_combined",
#         help="Nome do arquivo final sem extensão"
#     )

#     args = parser.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)

#     df = load_metrics(args.metrics)

#     print("\nArquivo carregado:", args.metrics)
#     print("\nColunas encontradas:")
#     print(df.columns)

#     if "epoch" not in df.columns:
#         df["epoch"] = range(1, len(df) + 1)

#     x_col = "epoch"

#     for col in df.columns:
#         if col != x_col:
#             df[col] = pd.to_numeric(df[col], errors="ignore")

#     # ========================================================
#     # FIGURA ÚNICA COM DOIS PAINÉIS
#     # ========================================================

#     # fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.7))
#     fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

#     ax1, ax2 = axes

#     # ========================================================
#     # (a) Main training losses
#     # ========================================================

#     safe_plot(ax1, df, x_col, "total_loss", "Total loss")
#     safe_plot(ax1, df, x_col, "distill_mse", "Distillation MSE")
#     safe_plot(ax1, df, x_col, "action_mae", "Action MAE")
#     safe_plot(ax1, df, x_col, "speed_mse", "Speed MSE")
#     safe_plot(ax1, df, x_col, "steer_mse", "Steering MSE")

#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Loss value")

#     # Espaço extra no topo para a legenda ficar dentro sem cobrir as curvas
#     # ax1.set_ylim(0.0, 0.42)
#     ax1.set_ylim(0.0, 0.34)

#     ax1.grid(True, alpha=0.25)

#     # ax1.legend(
#     #     loc="upper center",
#     #     ncol=2,
#     #     frameon=True,
#     #     fontsize=11,
#     #     handlelength=2.5,
#     #     borderpad=0.45,
#     #     labelspacing=0.45,
#     #     columnspacing=1.2
#     # )
#     ax1.legend(
#     loc="upper right",
#     ncol=1,
#     frameon=True,
#     fontsize=6.5,
#     handlelength=1.6,
#     borderpad=0.25,
#     labelspacing=0.25
# )

#     ax1.text(
#         0.5,
#         -0.24,
#         "(a) Main training losses",
#         transform=ax1.transAxes,
#         ha="center",
#         va="top",
#         # fontsize=17
#         fontsize=8.5
#     )

#     # ========================================================
#     # (b) Router and expert regularization
#     # ========================================================

#     safe_plot(ax2, df, x_col, "router_balance_loss", "Router balance loss")
#     safe_plot(ax2, df, x_col, "router_entropy", "Router entropy")
#     safe_plot(ax2, df, x_col, "old_task_new_expert_penalty", "Old task / new expert")
#     safe_plot(ax2, df, x_col, "new_task_old_expert_penalty", "New task / old expert")

#     ax2.set_xlabel("Epoch")
#     ax2.set_ylabel("Metric value")

#     # Espaço extra no topo para a legenda ficar dentro sem cobrir as curvas
#     ax2.set_ylim(0.0, 0.62)
#     ax2.set_ylim(0.0, 0.53)

#     ax2.grid(True, alpha=0.25)

#     # ax2.legend(
#     #     loc="upper center",
#     #     ncol=2,
#     #     frameon=True,
#     #     fontsize=11,
#     #     handlelength=2.5,
#     #     borderpad=0.45,
#     #     labelspacing=0.45,
#     #     columnspacing=1.2
#     # )

#     ax2.legend(
#     loc="upper right",
#     ncol=1,
#     frameon=True,
#     fontsize=6.5,
#     handlelength=1.6,
#     borderpad=0.25,
#     labelspacing=0.25
# )

#     ax2.text(
#         0.5,
#         -0.24,
#         "(b) Router and expert regularization",
#         transform=ax2.transAxes,
#         ha="center",
#         va="top",
#         # fontsize=17
#         fontsize=8.5
#     )

#     # Espaçamento geral
#     # fig.subplots_adjust(
#     #     left=0.07,
#     #     right=0.99,
#     #     top=0.96,
#     #     bottom=0.24,
#     #     wspace=0.30
#     # )
#     fig.subplots_adjust(
#     left=0.08,
#     right=0.99,
#     top=0.97,
#     bottom=0.26,
#     wspace=0.32
# )

#     # ========================================================
#     # SALVAR
#     # ========================================================

#     pdf_path = os.path.join(args.outdir, f"{args.name}.pdf")
#     png_path = os.path.join(args.outdir, f"{args.name}.png")

#     plt.savefig(pdf_path, bbox_inches="tight")
#     plt.savefig(png_path, dpi=300, bbox_inches="tight")
#     plt.close()

#     print(f"\n[OK] PDF salvo em: {pdf_path}")
#     print(f"[OK] PNG salvo em: {png_path}")


# if __name__ == "__main__":
#     main()

import os
import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {metrics_path}")

    ext = os.path.splitext(metrics_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(metrics_path)

    if ext == ".json":
        with open(metrics_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            for key in ["metrics", "history", "train_metrics", "rows"]:
                if key in data:
                    return pd.DataFrame(data[key])
            return pd.DataFrame(data)

    raise ValueError("Use um arquivo .csv ou .json")


def safe_plot(ax, df, x_col, metric, label=None):
    if metric not in df.columns:
        print(f"[SKIP] Coluna não encontrada: {metric}")
        return False

    values = pd.to_numeric(df[metric], errors="coerce")

    if values.isna().all():
        print(f"[SKIP] Coluna inválida: {metric}")
        return False

    if values.abs().max() < 1e-10:
        print(f"[SKIP] Coluna zerada: {metric}")
        return False

    ax.plot(df[x_col], values, label=label if label else metric)
    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="train_figures")
    parser.add_argument("--name", type=str, default="proto_girp_training_combined_horizontal")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_metrics(args.metrics)

    print("\nArquivo carregado:", args.metrics)
    print("\nColunas encontradas:")
    print(df.columns)

    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)

    x_col = "epoch"

    for col in df.columns:
        if col != x_col:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # Figura horizontal compacta para artigo em duas colunas
    # fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.65))
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.90))

    ax1, ax2 = axes

    # ========================================================
    # (a) Main training losses
    # ========================================================

    safe_plot(ax1, df, x_col, "total_loss", "Total")
    safe_plot(ax1, df, x_col, "distill_mse", "Distill.")
    safe_plot(ax1, df, x_col, "action_mae", "Action")
    safe_plot(ax1, df, x_col, "speed_mse", "Speed")
    safe_plot(ax1, df, x_col, "steer_mse", "Steer")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss value")
    ax1.set_ylim(0.0, 0.40)
    ax1.grid(True, alpha=0.25)

    ax1.text(
        0.02, 0.95,
        "(a)",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0)
    )

    ax1.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        fontsize=6.3,
        handlelength=1.4,
        borderpad=0.25,
        labelspacing=0.18
    )

    # ========================================================
    # (b) Router and expert regularization
    # ========================================================

    safe_plot(ax2, df, x_col, "router_balance_loss", "Balance")
    safe_plot(ax2, df, x_col, "router_entropy", "Entropy")
    safe_plot(ax2, df, x_col, "old_task_new_expert_penalty", "Old/New")
    safe_plot(ax2, df, x_col, "new_task_old_expert_penalty", "New/Old")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric value")
    ax2.set_ylim(0.0, 0.58)
    ax2.grid(True, alpha=0.25)

    ax2.text(
        0.02, 0.95,
        "(b)",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0)
    )

    ax2.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        fontsize=6.3,
        handlelength=1.4,
        borderpad=0.25,
        labelspacing=0.18
    )

    # fig.subplots_adjust(
    #     left=0.075,
    #     right=0.995,
    #     top=0.97,
    #     bottom=0.20,
    #     wspace=0.30
    # )

    fig.subplots_adjust(
    left=0.075,
    right=0.995,
    top=0.985,
    bottom=0.22,
    wspace=0.30
)

    pdf_path = os.path.join(args.outdir, f"{args.name}.pdf")
    png_path = os.path.join(args.outdir, f"{args.name}.png")

    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"\n[OK] PDF salvo em: {pdf_path}")
    print(f"[OK] PNG salvo em: {png_path}")


if __name__ == "__main__":
    main()