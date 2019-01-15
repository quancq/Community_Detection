import utils
import project_utils as putils
import pandas as pd
import numpy as np
import os, time
from igraph import Graph
from subprocess import check_call
import random


def archive_pipeline():
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

    # colors = list(np.linspace(0, 0xFFFFFF, num_communities+1)[1:].astype(int))
    colors = utils.generate_colors(num_communities)
    colors = ["#{:06X}".format(color) for color in colors]
    g.vs["fillcolor"] = [colors[label] for label in labels]

    # Save graph
    save_dot_path = "./Visualize/graph.dot"
    save_pdf_path = "./Visualize/graph.pdf"
    utils.make_parent_dirs(save_dot_path)
    g.write_dot(save_dot_path)
    check_call(['sfdp', "-Goverlap=false", '-Tpdf', save_dot_path, '-o', save_pdf_path])

    exec_time = time.time() - start_time
    print("Time : {:.2f} seconds".format(exec_time))


def is_common_community(graph, edge, prob=0.7):
    is_common = graph.vs[edge.source]["community"] == graph.vs[edge.target]["community"]
    if not is_common:
        r = random.random()
        if r > prob:
            is_common = True

    return is_common


if __name__ == "__main__":
    pass
    start_time = time.time()

    path = "./Dataset/category_input"
    selected_categories = utils.load_list(path)
    print("Selected {} categories : {}".format(len(selected_categories), selected_categories))

    tags = []
    map_pair_tag_occ = {}
    for i, cat in enumerate(selected_categories):
        print("{}/{} ...".format(i+1, len(selected_categories)))
        path = "./Dataset/refined_data/Pair_Tag/{}.csv".format(cat)
        df = utils.load_csv(path)
        for _, row in df.iterrows():
            tag1, tag2, occ = "_{}".format(row["Tag1"]), "_{}".format(row["Tag2"]), int(row["Num_Occurrence"])
            tags.extend([tag1, tag2])
            old_occ = map_pair_tag_occ.get((tag1, tag2), 0)
            map_pair_tag_occ.update({(tag1, tag2): old_occ + occ})

    vertices = list(set(tags))
    n_vertices = len(vertices)
    # print("Number vertices : ", n_vertices)
    tag_pairs, weights = [], []
    for tag_pair, occ in map_pair_tag_occ.items():
        tag_pairs.append(tag_pair)
        weights.append(occ)

    threshold = np.quantile(weights, 0.99)

    edge_list = tag_pairs
    print(edge_list[:3])

    # Build graph
    g = Graph()
    g.add_vertices(n_vertices)
    g.vs["name"] = vertices
    g.vs["label"] = vertices
    g.vs["style"] = "filled"
    g.add_edges(edge_list)
    # for src, dst in edge_list:
    #     try:
    #         g.add_edge(src, dst)
    #     except:
    #         print(src, dst)
    #         exit()
    g.es["weight"] = weights
    print(g.summary())

    # print("\nAfter delete vertices")
    # g.vs.select(_degree_le=5).delete()
    # print(g.summary())

    # Community detection
    cluster = g.community_multilevel(return_levels=True, weights=weights)[0]
    # print(cluster.summary())
    # print([c.summary() for c in cluster])
    num_communities = len(cluster)
    # print(c.membership)
    labels = cluster.membership
    g.vs["community"] = labels
    g.es.select(weight_lt=threshold)["style"] = "invis"
    g.es.select(lambda e: not is_common_community(g, e, prob=0.9)).delete()
    print("\nAfter delete edges")
    print(g.summary())

    colors = list(np.linspace(0, 0xFFFFFF, num_communities+1)[1:].astype(int))
    # colors = utils.generate_colors(num_communities)
    colors = ["#{:06X}".format(color) for color in colors]
    g.vs["fillcolor"] = [colors[label] for label in labels]

    # Save graph
    names = "_".join(selected_categories)
    save_dot_path = "./Visualize/graph_{}_{}.dot".format(names, num_communities)
    save_pdf_path = "./Visualize/graph_{}_{}.pdf".format(names, num_communities)
    utils.make_parent_dirs(save_dot_path)
    g.write_dot(save_dot_path)
    check_call(['sfdp', "-Goverlap=false", "-Goutputorder=edgesfirst", '-Tpdf', save_dot_path, '-o', save_pdf_path])

    print(cluster.summary())
    exec_time = time.time() - start_time
    print("Time : {:.2f} seconds".format(exec_time))
