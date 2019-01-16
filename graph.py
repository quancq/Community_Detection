import utils
import os, time


class Graph:
    def __init__(self, num_vertices=10):
        self.num_vertices = num_vertices
        self.map_edge_weight = {}
        for i in range(num_vertices - 1):
            map = {j: None for j in range(i + 1, num_vertices)}
            self.map_edge_weight.update({i: map})

        self.map_community_vertices = {}
        self.map_community_weight = {}
        self.init_singleton_community()

    def init_singleton_community(self):
        for i in range(self.num_vertices):
            self.map_community_vertices[i] = [i]

        num_communities = len(self.map_community_vertices)
        for i in range(num_communities - 1):
            map = {j: self.map_edge_weight[i][j] for j in range(i + 1, num_communities)}
            self.map_community_weight[i] = map

        self.communities = [i for i in range(self.num_vertices)]

    def add_edge(self, src, dst, weight=0):
        self.map_edge_weight[src][dst] = weight
