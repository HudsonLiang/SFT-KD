import os
import glob
from pprint import pprint
from datasets import load_from_disk, load_dataset, DatasetDict

# optional modelscope import (may not be installed)
try:
    import modelscope
    from modelscope.msdatasets import MsDataset
except Exception:
    modelscope = None
    MsDataset = None

# /Users/hudson_1/projects/SFT-KD/scripts/qa_loader.py

DATASET_NAME = "psychology-10k-Deepseek-R1-zh"
CANDIDATE_BASE_DIRS = [
    os.path.expanduser("~/.cache/modelscope/datasets"),
    os.path.expanduser("~/.cache/modelscope"),
    os.path.expanduser("~/.cache/huggingface/datasets"),
    os.path.expanduser("~/.cache/huggingface"),
    os.getcwd(),
    os.path.expanduser("~/.local/share/modelscope/datasets"),
]


def find_dataset_dir(name, bases):
    for base in bases:
        if not os.path.exists(base):
            continue
        for root, dirs, _ in os.walk(base):
            for d in dirs:
                if name in d:
                    return os.path.join(root, d)
    return None


def try_load_from_files(path):
    # try load_from_disk (arrow/dataset saved)
    # try modelscope first
    try:

        # try a top-level helper if provided
        try:
            if hasattr(modelscope, "load_dataset"):
                ds = modelscope.load_dataset(path)
                return ds
        except Exception:
            pass

        # try the msdatasets API
        try:

            # common patterns: MsDataset.load(path) or MsDataset(path)
            if hasattr(MsDataset, "load"):
                try:
                    ds = MsDataset.load(path)
                    return ds
                except Exception:
                    pass
            try:
                ds = MsDataset(path)
                return ds
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass

    # fallback to huggingface arrow/dataset saved on disk
    try:
        ds = load_from_disk(path)
        return ds
    except Exception:
        pass

    # detect common file formats
    patterns = {
        "jsonl": ["*.jsonl", "*.ndjson"],
        "json": ["*.json"],
        "csv": ["*.csv"],
        "tsv": ["*.tsv"],
    }
    for fmt, pats in patterns.items():
        files = []
        for p in pats:
            files.extend(sorted(glob.glob(os.path.join(path, p))))
        if files:
            try:
                if fmt == "jsonl":
                    return load_dataset("json", data_files=files)
                if fmt == "json":
                    return load_dataset("json", data_files=files)
                if fmt == "csv":
                    return load_dataset("csv", data_files=files)
                if fmt == "tsv":
                    return load_dataset("csv", data_files=files, delimiter="\t")
            except Exception:
                continue
    # as a last resort, try loading the directory as a dataset script name
    try:
        return load_dataset(path)
    except Exception:
        return None


def main():
    path = find_dataset_dir(DATASET_NAME, CANDIDATE_BASE_DIRS)
    if not path:
        print(f"未找到缓存数据集目录: {DATASET_NAME}")
        return

    ds = try_load_from_files(path)
    if ds is None:
        print(f"无法从目录加载数据集: {path}")
        return

    # normalize to a Dataset (choose first split if DatasetDict)
    if isinstance(ds, DatasetDict):
        split_name = next(iter(ds.keys()))
        split = ds[split_name]
    else:
        split_name = None
        split = ds

    print(f"加载成功: {path}")
    if split_name:
        print(f"使用分割: {split_name}")

    n = min(5, len(split))
    print(f"前 {n} 条：")
    for i in range(n):
        pprint(split[i])


if __name__ == "__main__":
    main()