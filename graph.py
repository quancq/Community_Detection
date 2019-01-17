import utils
import os, time


class Graph:
    def __init__(self, num_vertices=10):
        self.num_vertices = num_vertices
        self.map_edge_weight = {}                           # Weight between 2 edges
        for i in range(num_vertices):
            # map = {j: None for j in range(0, num_vertices)}
            self.map_edge_weight.update({i: {}})

        self.map_community_vertices = {}
        self.map_community_pair_weight = {}                 # Weight between 2 communities
        self.communities = []
        self.weights = [0 for _ in range(num_vertices)]     # Weight of each vertex (sum edge weights)
        self.map_vertex_community_weight = {}
        self.map_community_weight = {}                      # Weight of each community
        self.init_singleton_community()

    def init_singleton_community(self):
        for i in range(self.num_vertices):
            self.map_community_vertices[i] = [i]

        num_communities = len(self.map_community_vertices)
        for i in range(num_communities):
            map = {}
            for j in range(num_communities):
                w = self.map_edge_weight[i].get(j)
                if w is not None:
                    map[j] = w

            self.map_community_pair_weight[i] = map

        self.communities = [i for i in range(self.num_vertices)]
        self.map_community_weight = {c: 0 for c in range(num_communities)}

    def add_edge(self, src, dst, weight=0):
        self.map_edge_weight[src][dst] = weight
        self.map_edge_weight[dst][src] = weight

        c_src = self.communities[src]
        c_dst = self.communities[dst]
        old_weight = self.map_community_pair_weight[c_src].get(c_dst, 0)
        self.map_community_pair_weight[c_src][c_dst] = old_weight + weight

        self.weights[src] += weight
        if src != dst:
            self.weights[dst] += weight

        old_weight = self.map_vertex_community_weight.get((src, c_dst), 0)
        self.map_vertex_community_weight[(src, c_dst)] = old_weight + weight
        if src != dst:
            old_weight = self.map_vertex_community_weight.get((dst, c_src), 0)
            self.map_vertex_community_weight[(dst, c_src)] = old_weight + weight

        old_weight = self.map_community_weight[self.communities[src]]
        self.map_community_weight[self.communities[src]] = old_weight + weight
        if src != dst:
            old_weight = self.map_community_weight[self.communities[dst]]
            self.map_community_weight[self.communities[dst]] = old_weight + weight

    def get_vertices(self):
        return [i for i in range(self.num_vertices)]

    def get_adjacency_vertices(self, v):
        return list(self.map_edge_weight[v].keys())

    def get_community(self, v):
        return self.communities[v]

    def set_community(self, vertex, new_community):
        old_community = self.get_community(vertex)

        # Update weight between community pairs
        for c, vertices_in_c in self.map_community_vertices.items():
            weight = 0  # Weight between v and vertices in c
            for v_in_c in vertices_in_c:
                weight += self.map_edge_weight[vertex].get(v_in_c, 0)

            old_weight = self.map_community_pair_weight[old_community].get(c, 0)
            self.map_community_pair_weight[old_community][c] = old_weight - weight
            old_weight = self.map_community_pair_weight[new_community].get(c, 0)
            self.map_community_pair_weight[old_community][c] = old_weight + weight

        adjacency_vertices = self.get_adjacency_vertices(vertex)
        adjacency_vertices_plus = adjacency_vertices + vertex
        for adjacency_v in adjacency_vertices_plus:
            old_weight = self.map_vertex_community_weight.get((adjacency_v, old_community), 0)
            self.map_vertex_community_weight[(adjacency_v, old_community)] = \
                old_weight - self.map_edge_weight[vertex][adjacency_v]

            old_weight = self.map_vertex_community_weight.get((adjacency_v, new_community), 0)
            self.map_vertex_community_weight[(adjacency_v, new_community)] = \
                old_weight + self.map_edge_weight[vertex][adjacency_v]

        decrease_weight = 0
        for adjacency_v in adjacency_vertices_plus:
            if adjacency_v not in self.map_community_vertices[old_community]:
                decrease_weight += self.map_edge_weight[vertex][adjacency_v]
        self.map_community_weight[old_community] -= decrease_weight

        increase_weight = 0
        for adjacency_v in adjacency_vertices_plus:
            if adjacency_v not in self.map_community_vertices[new_community]:
                increase_weight += self.map_edge_weight[vertex][adjacency_v]
        self.map_community_weight[new_community] += increase_weight

        self.communities[vertex] = new_community
        self.map_community_vertices.get(old_community).remove(vertex)
        self.map_community_vertices.get(new_community).append(vertex)

    def get_modularity_delta(self, vertex, community):
        weight_community_c_c = self.map_community_pair_weight[community].get(community, 0)

