import os
import glob
from pprint import pprint
from datasets import load_from_disk, load_dataset, DatasetDict
import modelscope
from modelscope.msdatasets import MsDataset

# /Users/hudson_1/projects/SFT-KD/scripts/qa_loader.py

DATASET_NAME = "psychology-10k-Deepseek-R1-zh"

def main():
    ds = modelscope.load_dataset(DATASET_NAME)

    # normalize to a Dataset (choose first split if DatasetDict)
    if isinstance(ds, DatasetDict):
        split_name = next(iter(ds.keys()))
        split = ds[split_name]
    else:
        split_name = None
        split = ds

    print(f"加载成功: {DATASET_NAME}")
    if split_name:
        print(f"使用分割: {split_name}")

    n = min(5, len(split))
    print(f"前 {n} 条：")
    for i in range(n):
        pprint(split[i])


if __name__ == "__main__":
    main()