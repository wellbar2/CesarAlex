"""Visualizações apresentadas pela interface."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.figure import Figure
from matplotlib_venn import venn3


def create_sankey(counts: dict[str, int]) -> go.Figure:
    labels = [
        "Arquivos Enviados",
        "Tem ORCID",
        "Sem ORCID",
        "Com ORCID + OpenAlex",
        "Com ORCID + Sem OpenAlex",
        "Sem ORCID + OpenAlex por DOI",
        "Sem ORCID + Sem OpenAlex",
    ]
    figure = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                textfont={"family": "Arial, sans-serif", "size": 14, "color": "black"},
                node={
                    "pad": 18,
                    "thickness": 22,
                    "line": {"color": "black", "width": 0.8},
                    "label": labels,
                    "color": [
                        "lightblue",
                        "lightgreen",
                        "salmon",
                        "gold",
                        "orange",
                        "mediumseagreen",
                        "lightcoral",
                    ],
                },
                link={
                    "source": [0, 0, 1, 1, 2, 2],
                    "target": [1, 2, 3, 4, 5, 6],
                    "value": [
                        counts["files_with_orcid"],
                        counts["files_without_orcid"],
                        counts["files_with_orcid_with_openalex"],
                        counts["files_with_orcid_without_openalex"],
                        counts["files_without_orcid_with_openalex"],
                        counts["files_without_orcid_without_openalex"],
                    ],
                    "color": "rgba(160,160,160,0.55)",
                },
            )
        ]
    )
    figure.update_layout(
        title={"text": "Currículos com: ORCID e OpenAlex", "x": 0.0},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        height=420,
        font={"family": "Arial, sans-serif", "size": 14, "color": "black"},
    )
    return figure


def create_venn(counts: dict[str, int], total: int, title: str) -> Figure:
    subsets = (
        counts["Lattes_Apenas"],
        counts["ORCID_Apenas"],
        counts["Lattes_ORCID_Apenas"],
        counts["OpenAlex_Apenas"],
        counts["Lattes_OpenAlex_Apenas"],
        counts["ORCID_OpenAlex_Apenas"],
        counts["Lattes_ORCID_OpenAlex"],
    )
    colors = ("#009B3A", "#6A8EAE", "#D47A70")
    region_ids = ["100", "010", "110", "001", "101", "011", "111"]
    region_colors = _venn_region_colors(region_ids, colors)

    figure = plt.figure(figsize=(8, 8))
    venn = venn3(
        subsets=subsets,
        set_labels=("Lattes", "ORCID", "OpenAlex"),
        alpha=0.6,
        set_colors=colors,
    )
    for region_id, count in zip(region_ids, subsets, strict=True):
        label = venn.get_label_by_id(region_id)
        if label:
            percentage = count / total * 100 if total else 0.0
            label.set_text(f"{count}\n({percentage:.2f}%)")
            label.set_fontsize(14)
    for label in venn.set_labels:
        if label:
            label.set_fontweight("bold")
            label.set_fontsize(16)

    legend_labels = {
        "100": "Somente Lattes",
        "010": "Somente ORCID",
        "001": "Somente OpenAlex",
        "110": "Lattes ∩ ORCID",
        "101": "Lattes ∩ OpenAlex",
        "011": "ORCID ∩ OpenAlex",
        "111": "Lattes ∩ ORCID ∩ OpenAlex",
    }
    handles = [
        mpatches.Patch(
            facecolor=region_colors.get(region_id, (0.8, 0.8, 0.8, 0.6)),
            edgecolor="black",
            label=legend_labels[region_id],
        )
        for region_id in ("100", "010", "001", "110", "101", "011", "111")
    ]
    plt.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=True,
        title="Regiões do diagrama",
    )
    plt.title(title, fontsize=18)
    plt.tight_layout()
    return figure


def _venn_region_colors(region_ids: list[str], colors: tuple[str, str, str]):
    temporary_figure = plt.figure(figsize=(1, 1))
    temporary_venn = venn3(
        subsets=(1, 1, 1, 1, 1, 1, 1),
        set_labels=("Lattes", "ORCID", "OpenAlex"),
        alpha=0.6,
        set_colors=colors,
    )
    region_colors = {}
    for region_id in region_ids:
        patch = temporary_venn.get_patch_by_id(region_id)
        if patch is not None:
            region_colors[region_id] = patch.get_facecolor()
    plt.close(temporary_figure)
    return region_colors
