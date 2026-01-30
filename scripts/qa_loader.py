from pprint import pprint
from datasets import load_from_disk, DatasetDict
from modelscope.utils.file_utils import get_dataset_cache_root
import os
from modelscope.msdatasets import MsDataset

DATASET_NAME = "Kedreamix/psychology-10k-Deepseek-R1-zh"

def main():
    cache_dir = get_dataset_cache_root()
    # 构造 HuggingFace 数据集缓存路径
    local_dataset_path = os.path.join(cache_dir, DATASET_NAME)

    if not os.path.exists(local_dataset_path):
        print(f"未找到本地缓存数据集: {local_dataset_path}")
        return

    # 使用 MsDataset 加载数据集
    ds = MsDataset.load(local_dataset_path)
    hf_ds=ds.to_hf_dataset()
    
    iterable_dataset=hf_ds.to_iterable_dataset()
    next(iter(iterable_dataset))
    list(iterable_dataset.take(3))
    for i, item in enumerate(iterable_dataset.take(3)):
        print(item)

if __name__ == "__main__":
    main()
