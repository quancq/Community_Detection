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
            is_converge = self._move_nodes(graph)

        exec_time = time.time() - start_time
        print("Detect communities done. Time : {:.2f} seconds".format(exec_time))

    def _move_nodes(self, graph):
        graph = Graph()

