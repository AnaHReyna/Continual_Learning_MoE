# import json
# import numpy as np
# import matplotlib.pyplot as plt


# # ============================================================
# # CONFIGURAÇÃO
# # ============================================================

# json_files = {"Lane Keeping": "../Resultados/phase2/Proto-AW/lane_keeping/summary_results_lane_keeping.json",
#               "Change Lane": "../Resultados/phase2/Proto-AW/change_lane/summary_results_change_lane.json",
#               "Pedestrian": "../Resultados/phase2/Proto-AW/pedestrian/summary_results_pedestrian.json",
#             }

# selected_checkpoint = "ckpt-10"
# output_file = "expert_usage_protogirp_ckpt10.png"

# # Cores dos experts (troque aqui se quiser)
# colors = { "Expert 0": "#9D73BA",   # azul
#            "Expert 1": "#84FCA5",   # laranja
#            "Expert 2": "#FFA06C",   # verde
#         }


# # ============================================================
# # FUNÇÕES AUXILIARES
# # ============================================================

# def load_json(path):
#     with open(path, "r") as f:
#         data = json.load(f)
#     return data


# def find_checkpoint(data, checkpoint):
#     """
#     Procura o checkpoint dentro do JSON.
#     Aceita tanto lista direta quanto dict com chave 'results'.
#     """
#     if isinstance(data, dict):
#         if "results" in data:
#             data = data["results"]
#         else:
#             raise ValueError("JSON em formato inesperado: não encontrei lista de resultados.")

#     for item in data:
#         if item.get("checkpoint") == checkpoint:
#             return item

#     available = [item.get("checkpoint") for item in data]
#     raise ValueError(
#         f"Checkpoint {checkpoint} não encontrado.\n"
#         f"Checkpoints disponíveis: {available}"
#     )


# # ============================================================
# # LEITURA DOS RESULTADOS
# # ============================================================

# tasks = []
# expert0 = []
# expert1 = []
# expert2 = []
# # success_rates = []

# for task_name, json_path in json_files.items():
#     data = load_json(json_path)
#     result = find_checkpoint(data, selected_checkpoint)

#     tasks.append(task_name)
#     expert0.append(result["expert_usage_rate_0"] * 100)
#     expert1.append(result["expert_usage_rate_1"] * 100)
#     expert2.append(result["expert_usage_rate_2"] * 100)
#     # success_rates.append(result["success_rate"] * 100)


# # Converter para arrays
# expert0 = np.array(expert0)
# expert1 = np.array(expert1)
# expert2 = np.array(expert2)
# # success_rates = np.array(success_rates)

# # Inverter ordem para ficar bonito no horizontal
# tasks = tasks[::-1]
# expert0 = expert0[::-1]
# expert1 = expert1[::-1]
# expert2 = expert2[::-1]
# # success_rates = success_rates[::-1]


# # # ============================================================
# # # PLOT
# # # ============================================================

# # fig, ax = plt.subplots(figsize=(9, 5.5))

# # y = np.arange(len(tasks))

# # ax.barh(y, expert0, color=colors["Expert 0"], label="Expert 0")
# # ax.barh(y, expert1, left=expert0, color=colors["Expert 1"], label="Expert 1")
# # ax.barh(y, expert2, left=expert0 + expert1, color=colors["Expert 2"], label="Expert 2")

# # ax.set_yticks(y)
# # ax.set_yticklabels(tasks)
# # ax.set_xlabel("Average Expert Usage (%)")
# # ax.set_xlim(0, 100)
# # ax.set_title(f"Average Expert Usage - Proto-GIRP ({selected_checkpoint})")

# # # Grade leve
# # ax.grid(axis="x", linestyle="--", alpha=0.4)

# # # Legenda fora da área do gráfico
# # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)


# # ============================================================
# # PLOT - versão mais limpa para artigo
# # ============================================================

# fig, ax = plt.subplots(figsize=(9.5, 4.8))

# y = np.arange(len(tasks))

# # Cores mais suaves/profissionais
# # colors = {
# #     "Expert 0": "#7E57C2",   # roxo
# #     "Expert 1": "#66E88F",   # verde claro
# #     "Expert 2": "#FF9F68",   # laranja
# # }

# ax.barh(y, expert0, color=colors["Expert 0"], label="Expert 0", edgecolor="black", linewidth=0.3)
# ax.barh(y, expert1, left=expert0, color=colors["Expert 1"], label="Expert 1", edgecolor="black", linewidth=0.3)
# ax.barh(y, expert2, left=expert0 + expert1, color=colors["Expert 2"], label="Expert 2", edgecolor="black", linewidth=0.3)

# ax.set_yticks(y)
# ax.set_yticklabels(tasks, fontsize=11, fontweight="bold")
# ax.set_xlabel("Average Expert Usage (%)", fontsize=11, fontweight="bold")

# # deixa espaço para o SR à direita
# ax.set_xlim(0, 112)

# ax.tick_params(axis="x", labelsize=10)
# ax.grid(axis="x", linestyle="--", alpha=0.25)

# # Legenda acima, mais limpa
# ax.legend(
#     loc="lower center",
#     bbox_to_anchor=(0.5, 1.02),
#     ncol=3,
#     frameon=False,
#     fontsize=10,
#     prop={"weight": "bold", "size": 10}
# )

# # Título opcional: para artigo, eu deixaria sem título
# # ax.set_title(f"Average Expert Usage - Proto-GIRP$_{{P2}}$ ({selected_checkpoint})", fontsize=12)


# # ============================================================
# # RÓTULOS
# # ============================================================

# def add_labels(values, lefts, ypos):
#     for i, (v, l) in enumerate(zip(values, lefts)):
#         if v >= 6:
#             ax.text(
#                 l + v / 2,
#                 ypos[i],
#                 f"{v:.1f}%",
#                 ha="center",
#                 va="center",
#                 fontsize=9,
#                 fontweight="bold",
#                 color="black"
#             )
#         elif v >= 2:
#             ax.text(
#                 l + v + 0.8,
#                 ypos[i],
#                 f"{v:.1f}%",
#                 ha="left",
#                 va="center",
#                 fontsize=8,
#                 fontweight="bold",
#                 color="black"
#             )


# add_labels(expert0, np.zeros_like(expert0), y)
# add_labels(expert1, expert0, y)
# add_labels(expert2, expert0 + expert1, y)


# # # Success rate à direita
# # for i, sr in enumerate(success_rates):
# #     ax.text(104, y[i], f"SR = {sr:.0f}%", ha="left", va="center", fontsize=10,fontweight="bold")

# # Remove bordas desnecessárias
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)


# for label in ax.get_xticklabels():
#     label.set_fontweight("bold")

# for label in ax.get_yticklabels():
#     label.set_fontweight("bold")


# plt.tight_layout()

# plt.savefig(output_file, dpi=300, bbox_inches="tight")
# plt.savefig(output_file.replace(".png", ".pdf"), bbox_inches="tight")

# plt.show()

# print(f"Figura salva em: {output_file}")
# print(f"Figura salva em: {output_file.replace('.png', '.pdf')}")




# # ============================================================
# # RÓTULOS DOS SEGMENTOS
# # ============================================================

# def add_labels(values, lefts, ypos):
#     for i, (v, l) in enumerate(zip(values, lefts)):
#         if v >= 8:
#             # segmento grande: escreve dentro
#             ax.text(
#                 l + v / 2,
#                 ypos[i],
#                 f"{v:.1f}%",
#                 ha="center",
#                 va="center",
#                 fontsize=9
#             )
#         elif v >= 3:
#             # segmento médio/pequeno: escreve fora
#             ax.text(
#                 l + v + 1.0,
#                 ypos[i],
#                 f"{v:.1f}%",
#                 ha="left",
#                 va="center",
#                 fontsize=8
#             )
#         # se for menor que 3, omite para não poluir




# # ============================================================
# # OPCIONAL: COLOCAR SUCCESS RATE AO LADO DO NOME DA TAREFA
# # ============================================================

# # for i, sr in enumerate(success_rates):
# #     ax.text(
# #         102, y[i],
# #         f"SR = {sr:.0f}%",
# #         ha="left",
# #         va="center",
# #         fontsize=9
# #     )

# # plt.tight_layout()
# # plt.savefig(output_file, dpi=300, bbox_inches="tight")
# # plt.show()

# # print(f"Figura salva em: {output_file}")





import json
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Arquivos JSON do método que você quer plotar
# Neste caso: Proto-AW Phase 2
# json_files = {
#     "Lane Keeping": "../Resultados/phase2/Proto-AW/lane_keeping/summary_results_lane_keeping.json",
#     "Change Lane": "../Resultados/phase2/Proto-AW/change_lane/summary_results_change_lane.json",
#     "Pedestrian": "../Resultados/phase2/Proto-AW/pedestrian/summary_results_pedestrian.json",
# }

json_files = {
    "Lane Keeping": "../Resultados/phase2/Ours/lane_keeping/summary_results_lane_keeping.json",
    "Change Lane": "../Resultados/phase2/Ours/change_lane/summary_results_change_lane.json",
    "Pedestrian": "../Resultados/phase2/Ours/pedestrian/summary_results_pedestrian.json",
}

selected_checkpoint = "ckpt-10"

# method_name = "Proto-AW$_{P2}$"

# output_file = "expert_usage_proto_aw_p2_ckpt10_vertical.png"
method_name = "Proto-GIRP$_{P2}$"
output_file = "expert_usage_proto_girp_p2_ckpt10_vertical.png"

# Cores dos experts
colors = {
    "Expert 0": "#9D73BA",   # roxo
    "Expert 1": "#84FCA5",   # verde
    "Expert 2": "#FFA06C",   # laranja
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def load_json(path):
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


# ============================================================
# LEITURA DOS RESULTADOS
# ============================================================

tasks = []
expert0 = []
expert1 = []
expert2 = []
success_rates = []

for task_name, json_path in json_files.items():
    data = load_json(json_path)
    result = find_checkpoint(data, selected_checkpoint)

    tasks.append(task_name)

    expert0.append(result["expert_usage_rate_0"] * 100)
    expert1.append(result["expert_usage_rate_1"] * 100)
    expert2.append(result["expert_usage_rate_2"] * 100)

    success_rates.append(result["success_rate"] * 100)


# Converter para arrays
expert0 = np.array(expert0)
expert1 = np.array(expert1)
expert2 = np.array(expert2)
success_rates = np.array(success_rates)


# ============================================================
# PRINT DOS VALORES
# ============================================================

print(f"Method: {method_name}")
print(f"Checkpoint: {selected_checkpoint}")
print()

for i, task in enumerate(tasks):
    print(task)
    print(f"  Success rate: {success_rates[i]:.1f}%")
    print(f"  Expert 0: {expert0[i]:.1f}%")
    print(f"  Expert 1: {expert1[i]:.1f}%")
    print(f"  Expert 2: {expert2[i]:.1f}%")
    print()


# ============================================================
# PLOT VERTICAL STACKED BAR
# ============================================================

fig, ax = plt.subplots(figsize=(7.8, 5.2))

x = np.arange(len(tasks))
width = 0.58

ax.bar(
    x,
    expert0,
    width,
    color=colors["Expert 0"],
    label="Expert 0",
    edgecolor="black",
    linewidth=0.3,
)

ax.bar(
    x,
    expert1,
    width,
    bottom=expert0,
    color=colors["Expert 1"],
    label="Expert 1",
    edgecolor="black",
    linewidth=0.3,
)

ax.bar(
    x,
    expert2,
    width,
    bottom=expert0 + expert1,
    color=colors["Expert 2"],
    label="Expert 2",
    edgecolor="black",
    linewidth=0.3,
)


# ============================================================
# EIXOS E ESTILO
# ============================================================

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11, fontweight="bold")

ax.set_ylabel(
    "Average Expert Usage (%)",
    fontsize=11,
    fontweight="bold",
)

ax.set_ylim(0, 108)

ax.tick_params(axis="y", labelsize=10)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.25,
)

# Legenda acima
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    prop={"weight": "bold", "size": 10},
)

# Título opcional
# Para artigo, você pode deixar sem título e explicar na caption.
# Se quiser título, descomente:
# ax.set_title(
#     f"Average Expert Usage - {method_name} ({selected_checkpoint})",
#     fontsize=12,
#     fontweight="bold",
# )


# ============================================================
# RÓTULOS DOS SEGMENTOS
# ============================================================

def add_vertical_labels(values, bottoms):
    for i, (v, b) in enumerate(zip(values, bottoms)):

        # Segmentos grandes: label dentro da barra
        if v >= 6:
            ax.text(
                x[i],
                b + v / 2,
                f"{v:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
            )

        # Segmentos pequenos: label acima do segmento
        elif v >= 2:
            # Se o label ficaria muito alto, coloca dentro/abaixo
            if b + v + 2 > 103:
                ax.text(
                    x[i],
                    b + v - 1.0,
                    f"{v:.1f}%",
                    ha="center",
                    va="top",
                    fontsize=8,
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
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                )

        # Valores menores que 2% são omitidos para não poluir


add_vertical_labels(expert0, np.zeros_like(expert0))
add_vertical_labels(expert1, expert0)
add_vertical_labels(expert2, expert0 + expert1)


# ============================================================
# SUCCESS RATE OPCIONAL ACIMA DAS BARRAS
# ============================================================

for i, sr in enumerate(success_rates):
    ax.text(
        x[i],
        103,
        f"SR = {sr:.0f}%",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="black",
    )


# ============================================================
# NEGRITO NOS EIXOS
# ============================================================

for label in ax.get_xticklabels():
    label.set_fontweight("bold")

for label in ax.get_yticklabels():
    label.set_fontweight("bold")


# Remove bordas desnecessárias
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# SALVAR FIGURA
# ============================================================

plt.tight_layout()

plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.savefig(output_file.replace(".png", ".pdf"), bbox_inches="tight")

plt.show()

print(f"Figura salva em: {output_file}")
print(f"Figura salva em: {output_file.replace('.png', '.pdf')}")