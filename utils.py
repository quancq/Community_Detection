import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.externals import joblib

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def get_time_str(time=datetime.now(), fmt=DEFAULT_TIME_FORMAT):
    try:
        return time.strftime(fmt)
    except:
        return ""


def get_time_obj(time_str, fmt=DEFAULT_TIME_FORMAT):
    try:
        return datetime.strptime(time_str, fmt)
    except:
        return None


def transform_time_fmt(time_str, src_fmt, dst_fmt=DEFAULT_TIME_FORMAT):
    time_obj = get_time_obj(time_str, src_fmt)
    time_str = get_time_str(time_obj, dst_fmt)
    return time_str


def mkdirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def make_parent_dirs(path):
    dir = path[:path.rfind("/")]
    mkdirs(dir)


def get_all_file_paths(dir):
    file_paths = []
    for root, dirs, files in os.walk(dir):
        # print(files)
        files = [os.path.join(root, file) for file in files]
        file_paths.extend(files)

    return file_paths


def get_file_paths(parent_dir):
    file_paths = [os.path.join(parent_dir, file) for file in os.listdir(parent_dir)]
    file_paths = [file_path for file_path in file_paths if os.path.isfile(file_path)]
    return file_paths


def get_file_names(parent_dir):
    file_names = [file_name for file_name in os.listdir(parent_dir)
                  if os.path.isfile(os.path.join(parent_dir, file_name))]
    return file_names


def save_list(lst, save_path):
    if len(lst) == 0:
        return
    make_parent_dirs(save_path)

    with open(save_path, "w") as f:
        f.write("\n".join(lst))

    print("Save data (size = {}) to {} done".format(len(lst), save_path))


def save_csv(df, save_path):
    if df.shape[0] == 0:
        return
    make_parent_dirs(save_path)

    df.to_csv(save_path, index=False)
    print("Save data (size = {}) to {} done".format(df.shape[0], save_path))


def save_xlsx(df, save_path):
    if df.shape[0] == 0:
        return
    make_parent_dirs(save_path)
    df.to_excel(save_path, index=False)
    print("Save data (size = {}) to {} done".format(df.shape[0], save_path))


def save_json(data, save_path, mode="w"):
    if len(data) == 0:
        return

    make_parent_dirs(save_path)

    if mode == "a":
        data.update(load_json(save_path))

    with open(save_path, 'w') as f:
        json.dump(data, f, default=MyEncoder)

    print("Save json data (size = {}) to {} done".format(len(data), save_path))


def load_csv(path, **kwargs):
    data = None
    try:
        data = pd.read_csv(path, **kwargs)
        print("Read csv data (size = {}) from {} done".format(data.shape[0], path))
    except:
        print("Error when load csv data from ", path)
    return data


def load_csvs(paths):
    df = []
    for path in paths:
        df.append(load_csv(path))

    df = pd.concat(df, ignore_index=True, sort=False)
    print("Load {} files (size={}) csv done".format(len(paths), df.shape[0]))
    return df


def load_csvs_in_dir(parent_dir):
    paths = get_file_paths(parent_dir)
    df = load_csvs(paths)
    return df


def load_xlsx(path):
    data = None
    try:
        data = pd.read_excel(path)
        print("Read data (size = {}) from {} done".format(data.shape[0], path))
    except:
        print("Error when load csv data from ", path)
    return data


def load_json(path):
    data = {}
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except:
        print("Error when load file ", path)
        data = {}

    # print("Load json data (size = {}) from {} done".format(len(data), path))
    return data


def load_list(path):
    data = []
    with open(path, 'r') as f:
        data = f.read().strip().split("\n")

    print("Load list data (size = {}) from {} done".format(len(data), path))
    return data


def load_str(path):
    data = ""
    with open(path, 'r') as f:
        data = f.read().strip()

    return data


def convert_df_to_dict(df, col_key=None, col_value=None):
    if col_key is None or col_key not in list(df.columns):
        return {}

    if col_value is None:
        return df.set_index(col_key).to_dict("index")
    else:
        result = {k: v for k, v in zip(df[col_key].values.tolist(), df[col_value].values.tolist())}
        return result


def save_sklearn_model(model, save_path):
    make_parent_dirs(save_path)
    joblib.dump(model, save_path)
    print("Save sklearn model to {} done".format(save_path))


def load_sklearn_model(model_path):
    model = joblib.load(model_path)
    return model


def generate_colors(n):
    colors = np.ones(n, dtype=np.int32)
    for i in range(3):
        color_channels = np.random.randint(0, 255, size=n)
        colors += color_channels * 256
    return colors.tolist()


if __name__ == "__main__":
    pass
