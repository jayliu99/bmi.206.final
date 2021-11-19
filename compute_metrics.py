import networkx as nx
import numpy as np
import sys
from typing import List, Tuple, Iterable, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv


def createGraph(file_name):
    """Converts adjacency matrix (.csv) to a directed graph."""

    # Read the first line to determine the number of columns
    with open(file_name, 'r') as f:
        ncols = len(next(f).split(','))

    x = np.genfromtxt(file_name, delimiter=',', dtype=int, names=True,
                      usecols=range(1, ncols))  # skip the first column

    labels = x.dtype.names

    # create y to split x into a full array
    # y is a view of x, so it will not require much additional memory
    y = x.view(dtype=('int', len(x.dtype)))

    # create directed graph from y
    G = nx.from_numpy_matrix(y, create_using=nx.DiGraph())
    G = nx.relabel_nodes(G, dict(zip(range(ncols-1), labels)))

    return G


def cascade(facet_node, G, cascade_set):
    """Recursively computes the cascade set of a given node in a graph.

Keyword arguments:
facet_node -- a node for which to compute the cascade set
G -- a graph object
cascade_set -- an empty set; the function will place contents of cascade set in this object
"""

    # Initialize a new copy of the graph
    network_mod = G.copy()

    # Compute in-degree of the input graph
    indeg_prev = G.in_degree()

    # Remove out-flow connections of the facet node
    out_flow_edges = G.out_edges(facet_node)
    network_mod.remove_edges_from(out_flow_edges)

    # Compute in-degree after removing affected edges
    indeg = network_mod.in_degree()

    # Find differences of the in-degree before and after locking the facet node
    blocked_nodes = list({k[0]: indeg[k[0]] for k in set(
        indeg) - set(indeg_prev) if indeg[k[0]] == 0}.keys())
    for x in blocked_nodes:
        if x not in cascade_set:
            cascade_set.add(x)

    # Recursively do the above processes until there is
    # no further out-flow connections of the input facet_node.
    if len(blocked_nodes) > 0:
        cascade(blocked_nodes, network_mod, cascade_set)


def cascade_test():
    """Tests implementation of cascade algorithm. (see Table 2 of paper)"""
    G2 = nx.DiGraph()
    G2.add_edges_from([('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'D'),
                      ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'F')])
    cascade_set = set()
    cascade(['A'], G2, cascade_set)
    print(len(cascade_set))  # Should return 4


def bridging_centrality(network: nx.DiGraph,
                        betweenness_centralities: Union[List, Tuple, np.ndarray, None],
                        node_ids: Union[List, Tuple, Iterable,
                                        nx.classes.reportviews.NodeView] = None,
                        return_betweenness=False):
    """
    Compute bridging centrality of nodes in node_ids for provided network. 
    Bridging centrality is computed as the pairwise product of betweenness centrality and the bridging coefficient. 
    The betweenness centrality measurement is a fraction of the number of shortest paths between all nodes that pass through a given node; 
        these nodes have high influence on total information transit and tend to be bottleneck nodes between modules but can also be nodes that have high connectivity.

    The bridging coefficient measures the inverse degree of a given node divided by the summed inverse degree of the neighbors of that node.
        For a node to have high bridging coefficient, the node must have low degree but be connected to highly connected nodes; we assume these will be bridges between modules. 

    Inputs:
        [network]: a networkx DiGraph object
        [betweenness_centralities]: a list of betweenness centralities for nodes in network. If None, betweenness centrality will be computed and returned.
        [node_ids]: a list/tuple/NodeView (must be iterable) of node_ids. If None, network.nodes will be used
        [return_betweenness]: toggle returning of betweenness centrality. Will toggle to True if betweenness_centrality is None

    Outputs:
        [bridging_centrality_values]: a list of bridging centrality metrics with length equal to the number of nodes passed or in network
        [betweenness_centralities]: optional: a list of betweenness centralities for nodes used

    """

    if node_ids is None:
        node_ids = network.nodes

    assert len(node_ids) > 0
    try:
        iter(node_ids)
    except:
        raise ValueError(
            "You tried to pass arg node_ids that was not iterable. Please pass a list, tuple, or nx NodeView")

    if betweenness_centralities is None:  # if user didn't give betweenness centralities, compute
        centrality = nx.betweenness_centrality(network)
        betweenness_centralities = [
            centrality.get(node_id) for node_id in node_ids]
        #return_betweenness = True

    # we need to get the bridging coefficient for e/ node, so we'll loop through them all and then
    # multiply pairwise with the betweenness_centralities

    bridging_centrality_values = []
    for node_index, node_id in enumerate(node_ids):
        bridging_coefficient = compute_bridging_coefficient(network, node_id)
        bridging_centrality = bridging_coefficient * \
            betweenness_centralities[node_index]
        bridging_centrality_values.append(bridging_centrality)

    if return_betweenness:
        return bridging_centrality_values, betweenness_centralities
    else:
        return bridging_centrality_values


def compute_bridging_coefficient(network: nx.DiGraph, node_id: int):
    # computes bridging coefficient from network for node_id

    assert isinstance(network, nx.DiGraph)

    if node_id not in network.nodes:
        raise ValueError(
            f"Passed node {node} but that wasn't a valid node in the network you provided")

    node_degree = network.degree[node_id]

    graph_neighbors = network.neighbors(node_id)

    if network.in_degree[node_id] == 0 or network.out_degree[node_id] == 0:
        return 0  # if this node isnt connected to anything we should return 0

    else:  # eq is (degree_of_node_i)**-1 / sum(degrees_of_node_js)**-1 for all j in the neighbor graph of node i
        neighbor_degrees = np.array(
            [network.degree[neighbor_id] for neighbor_id in graph_neighbors])
        neighbor_degrees = np.nan_to_num(1 / neighbor_degrees)

        sum_of_inverse_ndegrees = np.sum(neighbor_degrees)
        betweenness_centrality = (1 / node_degree) / sum_of_inverse_ndegrees
        return betweenness_centrality


def main():
    G = createGraph(sys.argv[1])

    #c_metrics = {}  # Dictionary object to store centrality metrics
    bridging_centrality_values = []
    cascade_number_values = []

    # Calculate betweeness centrality (necessary for computing bridging centrality)
    centrality = nx.betweenness_centrality(G)

    for node in G.nodes:
        print("Calculating for node " + node + "...")
        cascade_set = set()

        cascade([node], G, cascade_set)  # Run cascade algorithm

        # Run bridging coefficient algorithm
        betweenness_centrality = [centrality.get(node)]
        bridge_c = bridging_centrality(
            G, betweenness_centrality, node_ids=[node], return_betweenness=False)[0]
        #print("Cascade number = " + str(len(cascade_set)))
        #print("Bridging centrality = " + str(bridge_c))

        # Store centrality metrics as tuple: (bridging number, cascade number)
        #c_metrics[node] = (bridge_c, len(cascade_set))
        bridging_centrality_values.append(str(bridge_c))
        cascade_number_values.append(str(len(cascade_set)))
    
    # Export data as csv's
    with open('bridges.csv', 'w', newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(bridging_centrality_values)
    with open('cascades.csv', 'w', newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(cascade_number_values)

    # Create plot of bridging centrality vs. cascade number
    # plt.scatter(bridging_centrality_values, cascade_number_values)
    # plt.xlabel("Bridging centrality")
    # plt.ylabel("Cascade number")
    # corrcoef = np.corrcoef(bridging_centrality_values, cascade_number_values)[0, 1].round(3)
    # plt.suptitle("Scatter of bridging centrality and cascade number")
    # plt.title(f"Pearson r = {corrcoef}")
    # plt.savefig('plot_1.png')

    #print(c_metrics)


if __name__ == "__main__":
    main()
