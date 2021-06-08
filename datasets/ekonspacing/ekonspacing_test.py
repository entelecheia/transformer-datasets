# %%
from pprint import pprint

from datasets import load_dataset

test_file = "../data/ekonspacing/test_small.txt"
val_file = "../data/ekonspacing/val_small.txt"
train_file = "../data/ekonspacing/train_small.txt"

# %%
dataset = load_dataset(
    "ekonspacing.py",
    name="small",
    data_files={"train": str(train_file), "validation": str(val_file), "test": str(test_file)},
    download_mode="force_redownload",
)
print(dataset)
# %%
pprint(dataset["train"][0])
pprint(dataset["test"][0])
# %%
