import json
import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

selected_checkpoint = "ckpt-10"

output_file = "expert_usage_phase2_combined.pdf"

colors = {
    "Expert 0": "#9D73BA",   # roxo
    "Expert 1": "#84FCA5",   # verde
    "Expert 2": "#FFA06C",   # laranja
}

methods = {
    "Proto-AW": {
        "Lane Keeping": "/home/ana/Documents/Architecture_Transformers_SR/student/Arquitetura/test_CARLA/Resultados/phase2/Proto-AW/lane_keeping/summary_results_lane_keeping.json",
        "Change Lane": "/home/ana/Documents/Architecture_Transformers_SR/student/Arquitetura/test_CARLA/Resultados/phase2/Proto-AW/change_lane/summary_results_change_lane.json",
        "Pedestrian": "/home/ana/Documents/Architecture_Transformers_SR/student/Arquitetura/test_CARLA/Resultados/phase2/Proto-AW/pedestrian/summary_results_pedestrian.json",
    },
    "Proto-GIRP": {
        "Lane Keeping": "/home/ana/Documents/Architecture_Transformers_SR/student/Arquitetura/test_CARLA/Resultados/phase2/Ours/lane_keeping/summary_results_lane_keeping.json",
        "Change Lane": "/home/ana/Documents/Architecture_Transformers_SR/student/Arquitetura/test_CARLA/Resultados/phase2/Ours/change_lane/summary_results_change_lane.json",
        "Pedestrian": "/home/ana/Documents/Architecture_Transformers_SR/student/Arquitetura/test_CARLA/Resultados/phase2/Ours/pedestrian/summary_results_pedestrian.json",
    },
}


# ============================================================
# CONFIGURAÇÃO VISUAL PARA ARTIGO
# ============================================================

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return data


def find_checkpoint(data, checkpoint):
    """
    Procura o checkpoint dentro do JSON.
    Aceita tanto lista direta quanto dict com chave 'results'.
    """
    if isinstance(data, dict):
        if "results" in data:
            data = data["results"]
        else:
            raise ValueError(
                "JSON em formato inesperado: não encontrei uma lista de resultados."
            )

    for item in data:
        if item.get("checkpoint") == checkpoint:
            return item

    available = [item.get("checkpoint") for item in data]
    raise ValueError(
        f"Checkpoint {checkpoint} não encontrado.\n"
        f"Checkpoints disponíveis: {available}"
    )


def read_method_results(json_files, checkpoint):
    tasks = []
    expert0 = []
    expert1 = []
    expert2 = []
    success_rates = []

    for task_name, json_path in json_files.items():
        data = load_json(json_path)
        result = find_checkpoint(data, checkpoint)

        tasks.append(task_name)
        expert0.append(result["expert_usage_rate_0"] * 100)
        expert1.append(result["expert_usage_rate_1"] * 100)
        expert2.append(result["expert_usage_rate_2"] * 100)
        success_rates.append(result["success_rate"] * 100)

    return {
        "tasks": tasks,
        "expert0": np.array(expert0),
        "expert1": np.array(expert1),
        "expert2": np.array(expert2),
        "success_rates": np.array(success_rates),
    }


def add_vertical_labels(ax, x, values, bottoms):
    for i, (v, b) in enumerate(zip(values, bottoms)):

        if v >= 6:
            ax.text(
                x[i],
                b + v / 2,
                f"{v:.1f}%",
                ha="center",
                va="center",
                fontsize=5.8,
                fontweight="bold",
                color="black",
            )

        elif v >= 2:
            if b + v + 2 > 103:
                ax.text(
                    x[i],
                    b + v - 1.0,
                    f"{v:.1f}%",
                    ha="center",
                    va="top",
                    fontsize=5.4,
                    fontweight="bold",
                    color="black",
                )
            else:
                ax.text(
                    x[i],
                    b + v + 1.0,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                    fontweight="bold",
                    color="black",
                )


def plot_method(ax, method_name, result, panel_label):
    tasks = result["tasks"]
    expert0 = result["expert0"]
    expert1 = result["expert1"]
    expert2 = result["expert2"]
    success_rates = result["success_rates"]

    x = np.arange(len(tasks))
    width = 0.56

    b0 = ax.bar(
        x,
        expert0,
        width,
        color=colors["Expert 0"],
        label="Expert 0",
        edgecolor="black",
        linewidth=0.25,
    )

    b1 = ax.bar(
        x,
        expert1,
        width,
        bottom=expert0,
        color=colors["Expert 1"],
        label="Expert 1",
        edgecolor="black",
        linewidth=0.25,
    )

    b2 = ax.bar(
        x,
        expert2,
        width,
        bottom=expert0 + expert1,
        color=colors["Expert 2"],
        label="Expert 2",
        edgecolor="black",
        linewidth=0.25,
    )

    add_vertical_labels(ax, x, expert0, np.zeros_like(expert0))
    add_vertical_labels(ax, x, expert1, expert0)
    add_vertical_labels(ax, x, expert2, expert0 + expert1)

    for i, sr in enumerate(success_rates):
        ax.text(
            x[i],
            103,
            f"SR = {sr:.0f}%",
            ha="center",
            va="bottom",
            fontsize=5.8,
            fontweight="bold",
            color="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=6.2, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=6.5)

    ax.grid(axis="y", linestyle="--", alpha=0.25)

    ax.text(
        0.02,
        0.96,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
    )

    ax.text(
        0.5,
        -0.24,
        method_name,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return b0, b1, b2


# ============================================================
# MAIN
# ============================================================

def main():
    results = {}

    for method_name, json_files in methods.items():
        results[method_name] = read_method_results(
            json_files=json_files,
            checkpoint=selected_checkpoint,
        )

    print(f"Checkpoint: {selected_checkpoint}\n")

    for method_name, result in results.items():
        print("=" * 60)
        print(method_name)
        print("=" * 60)

        for i, task in enumerate(result["tasks"]):
            print(task)
            print(f"  Success rate: {result['success_rates'][i]:.1f}%")
            print(f"  Expert 0: {result['expert0'][i]:.1f}%")
            print(f"  Expert 1: {result['expert1'][i]:.1f}%")
            print(f"  Expert 2: {result['expert2'][i]:.1f}%")
            print()

    # Figura horizontal compacta para artigo em duas colunas
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 2.90),
        sharey=True,
    )

    ax1, ax2 = axes

    handles = plot_method(
        ax=ax1,
        method_name="Proto-AW",
        result=results["Proto-AW"],
        panel_label="(a)",
    )

    plot_method(
        ax=ax2,
        method_name="Proto-GIRP",
        result=results["Proto-GIRP"],
        panel_label="(b)",
    )

    ax1.set_ylabel(
        "Average Expert Usage (%)",
        fontsize=7.5,
        fontweight="bold",
    )

    # Uma única legenda para os dois painéis
    fig.legend(
        handles,
        ["Expert 0", "Expert 1", "Expert 2"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=3,
        frameon=False,
        prop={"weight": "bold", "size": 6.8},
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.83,
        bottom=0.27,
        wspace=0.16,
    )

    # Salvar
    pdf_path = output_file
    png_path = output_file.replace(".pdf", ".png")

    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)

    plt.show()

    print(f"Figura salva em: {pdf_path}")
    print(f"Figura salva em: {png_path}")


if __name__ == "__main__":
    main()