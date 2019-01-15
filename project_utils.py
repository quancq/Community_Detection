import utils
import pandas as pd
import os, time
from collections import defaultdict


def get_all_pairs(lst):
    for i in range(0, len(lst) - 1):
        for j in range(i+1, len(lst)):
            yield lst[i], lst[j]


def load_dataset():
    dataset_dir = "./Dataset/full_data/"
    dataset_paths = utils.get_file_paths(dataset_dir)

    for path in dataset_paths[:1]:
        data = utils.load_json(path)
        print(data[0])
        print(len(data))


def refine_raw_data(raw_dataset_dir, refined_dataset_dir):
    start_time = time.time()

    map_tag_info = {}
    # map_category_tag_post = {}

    raw_names = utils.get_file_names(raw_dataset_dir)
    for i, raw_name in enumerate(raw_names):
        map_pair_tag_occurrence = {}
        category = raw_name[:raw_name.find(".")]
        # map_tag_post = defaultdict(list)
        posts = utils.load_json(os.path.join(raw_dataset_dir, raw_name))
        print("{}/{} Refine {} ({} posts) ...".format(i+1, len(raw_names), category, len(posts)))

        # print(posts[0])
        for post_id, post in enumerate(posts):
            tags = post.get("Tags", [])

            # Update tag info
            # for tag in tags:
            #     map_tag_post[tag].append(post_id)

                # map_cat_ids = map_tag_info.get(tag)
                # if map_cat_ids is None:
                #     map_cat_ids = {}
                #     map_tag_info[tag] = map_cat_ids
                # ids = map_cat_ids.get(category)
                # if ids is None:
                #     ids = []
                #     map_cat_ids[category] = ids
                # ids.append(post_id)

            # Update tag relationship
            for tag1, tag2 in get_all_pairs(tags):
                pred_tag, succ_tag = min(tag1, tag2), max(tag1, tag2)
                occurrence = map_pair_tag_occurrence.get((pred_tag, succ_tag), 0)
                map_pair_tag_occurrence.update({(pred_tag, succ_tag): occurrence + 1})

        # map_category_tag_post[category] = dict(map_tag_post)
        save_path = os.path.join(refined_dataset_dir, "Pair_Tag", "{}.csv".format(category))
        df = []
        for (tag1, tag2), num_occ in map_pair_tag_occurrence.items():
            df.append((tag1, tag2, num_occ))
        df = pd.DataFrame(df, columns=["Tag1", "Tag2", "Num_Occurrence"])
        utils.save_csv(df, save_path)
        print("Total pair_tags : ", len(map_pair_tag_occurrence))
        # break

    # Save result
    # save_path = os.path.join(refined_dataset_dir, "category_tag_post.json")
    # utils.save_json(map_category_tag_post, save_path)

    # save_path = os.path.join(refined_dataset_dir, "tags_info.json")
    # utils.save_json(map_tag_info, save_path)
    #
    # save_path = os.path.join(refined_dataset_dir, "tags_relationship.csv")
    # df = []
    # for (tag1, tag2), occ in map_pair_tag_occurrence.items():
    #     df.append((tag1, tag2, occ))
    # df = pd.DataFrame(df, columns=["Tag1", "Tag2", "Occurrence"])
    # utils.save_csv(df, save_path)

    print("Total tags  all category : ", len(map_tag_info))

    exec_time = time.time() - start_time
    print("Time : {:.2f} seconds".format(exec_time))


if __name__ == "__main__":
    pass
    raw_dataset_dir = "./Dataset/raw_data/full_data"
    refined_dataset_dir = "./Dataset/refined_data/"
    refine_raw_data(raw_dataset_dir, refined_dataset_dir)
