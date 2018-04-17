
"""
This file contains a list of examples with different layouts that can be
obtained using the ``add_grid_edges`` method.
"""
import argparse
import numpy as np
import maxflow
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# region Helper Functions
def plot_graph_2d(graph, nodes_shape, plot_weights=True, plot_terminals=True, font_size=7, title=None):
    X, Y = np.mgrid[:nodes_shape[0], :nodes_shape[1]]
    aux = np.array([Y.ravel(), X[::-1].ravel()]).T
    positions = {i: aux[i] for i in range(25)}
    positions['s'] = (-1, nodes_shape[0] / 2.0 - 0.5)
    positions['t'] = (nodes_shape[1], nodes_shape[0] / 2.0 - 0.5)

    nxgraph = graph.get_nx_graph()
    if not plot_terminals:
        nxgraph.remove_nodes_from(['s', 't'])

    plt.clf()
    nx.draw(nxgraph, pos=positions)

    if plot_weights:
        edge_labels = {}
        for u, v, d in nxgraph.edges(data=True):
            edge_labels[(u,v)] = d['weight']
        nx.draw_networkx_edge_labels(nxgraph,
                                     pos=positions,
                                     edge_labels=edge_labels,
                                     label_pos=0.3,
                                     font_size=font_size)
    plt.axis('equal')
    if title:
        print('TITLE', title)
        plt.suptitle(title)
    plt.show()
# endregion


# region Plot Functions
def A():
    # Standard 4-connected grid
    g = maxflow.Graph[int]()
    nodeids = g.add_grid_nodes((5, 5))
    g.add_grid_edges(nodeids, 1)
    plot_graph_2d(g, nodeids.shape, plot_terminals=False, title='Standard 4-connected grid')


def B():
    # 8-connected grid
    g = maxflow.Graph[int]()
    nodeids = g.add_grid_nodes((5, 5))
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [1, 1, 1]])
    # Also structure = maxflow.moore_structure(ndim=2, directed=True)
    g.add_grid_edges(nodeids, 1, structure=structure, symmetric=True)
    plot_graph_2d(g, nodeids.shape, plot_terminals=False, title='8-connected grid')


def C():
    # 24-connected 5x5 neighborhood
    g = maxflow.Graph[int]()
    nodeids = g.add_grid_nodes((5, 5))
    structure = np.array([[1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1],
                          [1, 1, 0, 1, 1],
                          [1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1]])
    g.add_grid_edges(nodeids, 1, structure=structure, symmetric=False)
    plot_graph_2d(g, nodeids.shape, plot_terminals=False, plot_weights=False, title='24-connected 5x5 neighborhood')


def D():
    # Diagonal, not symmetric
    g = maxflow.Graph[int]()
    nodeids = g.add_grid_nodes((5, 5))
    structure = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 1]])
    g.add_grid_edges(nodeids, 1, structure=structure, symmetric=False)
    plot_graph_2d(g, nodeids.shape, plot_terminals=False, title='Diagonal, not symmetric')


def E():
    # Central node connected to every other node
    g = maxflow.Graph[int]()
    nodeids = g.add_grid_nodes((5, 5)).ravel()

    central_node = nodeids[12]
    rest_of_nodes = np.hstack([nodeids[:12], nodeids[13:]])

    nodeids = np.empty((2, 24), dtype=np.int_)
    nodeids[0] = central_node
    nodeids[1] = rest_of_nodes

    structure = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0]])
    g.add_grid_edges(nodeids, 1, structure=structure, symmetric=False)
    plot_graph_2d(g, (5, 5), plot_terminals=False, title='Central node connected to every other node')
# endregion


funcs = {'A': A,
         'B': B,
         'C': C,
         'D': D,
         'E': E}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('plots', nargs='+', choices=funcs.keys(), help='list of plots you want to show')
    args = parser.parse_args()
    for plot in args.plots:
        print('PLOT {}'.format(plot))
        funcs[plot]()
