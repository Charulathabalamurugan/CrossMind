"""
Visualization utilities for PyVis and Plotly.
"""
from typing import List, Dict, Optional, Set
import json
from pyvis.network import Network
import plotly.graph_objects as go
import networkx as nx

NODE_COLOR_PALETTE = {
    "CustomerNeed": "#f43f5e",
    "Obstacle": "#f59e0b",
    "Mechanism": "#10b981",
    "Technology": "#3b82f6",
    "Material": "#a855f7",
    "Property": "#06b6d4",
    "Device": "#ec4899",
    "Concept": "#94a3b8"
}


def networkx_to_pyvis_html(
    graph: nx.Graph,
    highlighted_nodes: Optional[Set[str]] = None,
    highlighted_edges: Optional[Set[tuple]] = None,
    height: int = 600,
) -> str:
    if highlighted_nodes is None:
        highlighted_nodes = set()
    if highlighted_edges is None:
        highlighted_edges = set()

    net = Network(height=f"{height}px", bgcolor="#0b0c13", font_color="#f8fafc")
    net.toggle_physics(True)

    for node, data in graph.nodes(data=True):
        label = data.get("label", "Concept")
        color = NODE_COLOR_PALETTE.get(label, NODE_COLOR_PALETTE["Concept"])
        size = 30 if node in highlighted_nodes else 18
        border = "#ffffff" if node in highlighted_nodes else color

        net.add_node(
            node,
            label=node,
            title=f"{node} ({label})<br>{data.get('description','')}\nPapers: {len(data.get('paper_ids', []))}",
            color={
                "background": color,
                "border": border,
                "highlight": {"background": "#ff007f", "border": "#ffffff"}
            },
            size=size,
        )

    for u, v, data in graph.edges(data=True):
        rel_type = data.get("type", "ASSOCIATED_WITH")
        edge_key = (u, v)
        edge_color = "#ff007f" if edge_key in highlighted_edges else "#94a3b8"
        width = 4 if edge_key in highlighted_edges else 1.5

        net.add_edge(
            u,
            v,
            label=rel_type,
            color=edge_color,
            width=width,
            arrows="to"
        )

    return net.generate_html()


def path_metrics_figure(path_data: Dict) -> go.Figure:
    labels = ["Semantic Similarity", "Citation Weight", "Path Strength"]
    values = [
        path_data["metrics"].get("similarity", 0.0),
        path_data["metrics"].get("citation_weight", 0.0),
        path_data["metrics"].get("path_strength", 0.0)
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#818cf8", "#f59e0b", "#10b981"]
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(range=[0, 1])
    )
    return fig
