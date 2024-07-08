"""Implement dataset related functions"""

import logging
from typing import Dict, List, Optional, Tuple

from data.mind_dataset import MIND_DGL
import dgl # type: ignore

logger = logging.getLogger(__name__)



def load_data(dataset_name: str, dataset_dir: str, cfg=None, add_self_loop=False) -> Tuple[dgl.DGLGraph, int, int]:
    """Load dataset

    Parameters
    ----------
    dataset_name : str
        Dataset name
    dataset_dir : str
        Directory to save the dataset
    add_self_loop : bool, optional

    Returns
    -------
    dgl.DGLGraph
        Graph in DGL format
    """
    graph = dgl.DGLGraph()
    if dataset_name == "mind":
        dataset = MIND_DGL(cfg, force_reload=False)
        graph = dataset.graph
        return graph, dataset
        # graph.ndata["label"] = graph.ndata["label"].float()
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")
