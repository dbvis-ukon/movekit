import itertools
import networkx as nx
from functools import reduce
import numpy as np
import pandas as pd

from .preprocess import *
from .clustering import *
from .feature_extraction import *


def network_time_graphlist(preprocessed_data, object_type='delaunay_object'):
    """
    Calculates a network list for each timestep based on delaunay triangulation (currently only one available)
    :param preprocessed_data: Pandas DataFrame containing movement records
    :param object_type: delaunay_object - currently the only one available.
    :return: List of nx graphs, containing, singular group and relational attributes on nodes, graph and edges.
    """
    # Get spateial objects
    spatial_obj = get_spatial_objects(preprocessed_data, group_output=True)

    # Obtain specified objects
    del_dict = spatial_obj[['time', object_type]].to_dict('list')

    # Point Labels
    aidlst = sorted(list(set(preprocessed_data['animal_id'])))
    ids = list(range(0, len(aidlst)))
    id_dictionary = dict(zip(ids, aidlst))

    # Euclidean dists grouped into time
    dists_time = timewise_dict(euclidean_dist(preprocessed_data))

    # Pointwise features grouped into time
    data_groups_time = timewise_dict(
        centroid_medoid_computation(extract_features(preprocessed_data)))

    # Groupwise features
    grp = timewise_dict(get_group_data(preprocessed_data))

    # Python3 list to hold a graph for each time step-
    graph_list = []

    # Create a new graph for each time step, compute edges, add them to graph and
    # finally, add graph to Python3 list-
    for item in range(0, len(del_dict[object_type])):
        # Create an empty graph with no nodes and no edges.
        G = nx.Graph()

        # Add path to graph-
        for path in del_dict[object_type][item].simplices:
            nx.add_path(G, path)

        # Append graph attributes
        G.graph = grp[item + 1].to_dict('records')[0]

        # Append node attributes
        nx.set_node_attributes(G, data_groups_time[item + 1].to_dict('index'))
        #G.nodes = data_groups_time[item+1]

        # Relabel nodes
        G = nx.relabel_nodes(G, id_dictionary)

        dists = dists_time[item + 1].drop(['time', 'animal_id', 'x', 'y'],
                                          axis=1).to_dict()

        values = list(
            itertools.chain.from_iterable(pd.DataFrame(dists).values.tolist()))

        keys = list(
            itertools.product(list(dists_time[item + 1]['animal_id']),
                              list(dists.keys())))

        edgelabs = dict(zip(keys, values))

        for k, v in edgelabs.items():
            edgelabs[k] = {'distance': v}

        for k in list(edgelabs):
            if k not in G.edges():
                del edgelabs[k]

        # Append edge attributes
        nx.set_edge_attributes(G, edgelabs)
        #G.edges = edgelabs

        # Add graph of current time step to list-
        graph_list.append(G)

    return graph_list
