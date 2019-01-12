import utils
import project_utils as putils
import pandas as pd
import numpy as np
import os, time
from igraph import Graph


if __name__ == "__main__":
    pass
    start_time = time.time()
    path = "./Dataset/refined_data/temp/tags_relationship.csv"
    df = utils.load_csv(path)
    df["Tag1"] = df["Tag1"].apply(lambda x: "_" + str(x))
    df["Tag2"] = df["Tag2"].apply(lambda x: "_" + str(x))
    # print(df.head())

    weights = df["Occurrence"].values.tolist()
    threshold = np.quantile(weights, 0.995)
    min_w, max_w, avg_w = np.min(weights), np.max(weights), np.average(weights)
    print("Min = {}, Max = {}, Avg = {}, Threshold = {}".format(min_w, max_w, avg_w, threshold))
    tags1 = df["Tag1"].values.tolist()
    tags2 = df["Tag2"].values.tolist()
    tags = tags1
    tags.extend(tags2)

    vertices = list(set(tags))
    n_vertices = len(vertices)
    # print("Number vertices : ", n_vertices)
    edge_list = list(zip(tags1, tags2))
    # print(edge_list[:3])

    # Build graph
    g = Graph()
    g.add_vertices(n_vertices)
    g.vs["name"] = vertices
    g.vs["label"] = vertices
    g.vs["style"] = "filled"
    g.add_edges(edge_list)
    g.es["weight"] = weights
    print(g.summary())

    print("\nAfter delete vertices")
    g.vs.select(_degree_le=5).delete()
    print(g.summary())

    # Community detection
    cluster = g.community_multilevel(return_levels=True, weights=weights)[-1]
    print(cluster.summary())
    # print([c.summary() for c in cluster])
    num_communities = len(cluster)
    # print(c.membership)
    labels = cluster.membership
    g.es.select(weight_lt=threshold)["style"] = "invis"
    print("\nAfter delete edges")
    print(g.summary())

    colors = list(np.linspace(0, 0xFFFFFF, num_communities+1)[1:].astype(int))
    colors = ["#{:06X}".format(color) for color in colors]
    g.vs["fillcolor"] = [colors[label] for label in labels]

    # Save graph
    save_path = "./Visualize/graph.dot"
    utils.make_parent_dirs(save_path)
    g.write_dot(save_path)

    exec_time = time.time() - start_time
    print("Time : {:.2f} seconds".format(exec_time))
