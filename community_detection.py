import utils
from graph import Graph
import time, os
import pandas as pd
import numpy as np
import igraph
import sys


class Louvain:
    def __init__(self, graph):
        self.graph = graph

    def detect_communities(self, max_iter=10):
        start_time = time.time()
        print("Start detect communities ...")
        if max_iter is None:
            max_iter = sys.maxsize

        graph = self.graph
        iter = 0
        history = []
        graph.summary()
        while iter < max_iter:
            iter += 1
            print("Iter {}/{} ...".format(iter, max_iter))
            is_moved = self._move_nodes(graph)
            graph, map_old_vertex_new_vertex = self._aggregate_graph(graph)
            history.append(map_old_vertex_new_vertex)

            if not is_moved:
                break

        exec_time = time.time() - start_time
        map_vertex_community = self.get_final_community(history)
        num_communities = len(set(map_vertex_community.values()))
        print("Detect {} communities done. Time : {:.2f} seconds".format(num_communities, exec_time))

        return map_vertex_community

    def _move_nodes(self, graph):
        # graph = Graph()
        is_moved = False
        num_moves = 0
        while True:
            is_converge = True
            for v in graph.get_vertices():
                # Find best community for local move
                max_modularity_delta = -sys.maxsize
                best_community = -1
                adjacency_vertices = graph.get_adjacency_vertices(v)
                neighbor_communities = set([graph.get_community(v) for v in adjacency_vertices])

                # print("Candidate -> V : {}, Neighbor_Community : {}".format(v, neighbor_communities))
                for community in neighbor_communities:
                    curr_community = graph.get_community(v)
                    if community == curr_community:
                        continue

                    delta = graph.get_modularity_delta(v, curr_community, community)
                    if delta > max_modularity_delta:
                        max_modularity_delta = delta
                        best_community = community

                if max_modularity_delta > 0:
                    is_converge = False
                    # Move v to new community
                    if num_moves % 1000 == 0:
                        print("Move number {} ...".format(num_moves))
                    graph.move_community(v, best_community)
                    is_moved = True
                    num_moves += 1
                    if num_moves % 1000 == 0:
                        print("Modularity delta : ", max_modularity_delta)
                        # exit()

                if num_moves > 1e4 or (0 < max_modularity_delta < 1e-6):
                    is_converge = True
                    break

            if is_converge:
                break

        return is_moved

    def _aggregate_graph(self, graph):
        map_old_vertex_new_vertex = {}
        # communities = list(graph.map_community_vertices.keys())
        communities = list(set(graph.communities))
        num_communities = len(communities)
        print("Nummber comm : ", num_communities)

        for c_id, c in enumerate(communities):
            vertices = graph.map_community_vertices.get(c)
            for v in vertices:
                map_old_vertex_new_vertex.update({v: c_id})

        new_graph = Graph(num_vertices=num_communities)
        for c1_id in range(num_communities):
            for c2_id in range(c1_id, num_communities):
                c1, c2 = communities[c1_id], communities[c2_id]
                weight = graph.map_community_pair_weight[c1].get(c2)
                if weight is not None:
                    new_graph.add_edge(c1_id, c2_id, weight)
        print("\nAggregate... => Graph : {} vertices, {} edges".format(new_graph.num_vertices, new_graph.num_edges))
        return new_graph, map_old_vertex_new_vertex

    def get_final_community(self, history):
        map_vertex_community = history[0]
        for i in range(1, len(history)):
            # print(map_vertex_community)
            org_vertices = list(map_vertex_community.keys())
            for org_v in org_vertices:
                map_vertex_community[org_v] = history[i][map_vertex_community[org_v]]

        return map_vertex_community


if __name__ == "__main__":
    graph = Graph(num_vertices=7)
    edge_list = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 4), (3, 5), (3, 6), (4, 5), (5, 6)]
    # edge_list = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 4), (3, 5), (4, 5)]
    for src, dst in edge_list:
        graph.add_edge(src, dst)

    model = Louvain(graph)
    map_vertex_community = model.detect_communities()

    print(map_vertex_community)

