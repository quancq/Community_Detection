import utils
from graph import Graph
import time, os
import pandas as pd
import numpy as np


class Louvain:
    def __init__(self, graph):
        self.graph = graph

    def detect_communities(self):
        start_time = time.time()
        graph = self.graph
        while True:
            self._move_nodes(graph)
            break

        exec_time = time.time() - start_time
        print("Detect communities done. Time : {:.2f} seconds".format(exec_time))

    def _move_nodes(self, graph):
        graph = Graph()
        while True:
            is_moved = False
            for v in graph.get_vertices():
                # Find best community for local move
                max_modularity_delta = -1
                adjacency_vertices = graph.get_adjacency_vertices(v)
                neighbor_communities = set([graph.get_community(v) for v in adjacency_vertices])

                for community in neighbor_communities:
                    if community == graph.get_community(v):
                        continue

                    delta = graph.get_modularity_delta(v, community)
                    if delta > max_modularity_delta:
                        max_modularity_delta = delta
                        # Move v to new community
                        graph.set_community(v, community)

                if max_modularity_delta > 0:
                    is_moved = True

            if not is_moved:
                break
            break
