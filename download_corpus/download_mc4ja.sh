# Download MC4 (Ja)
cd ../data
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
time git lfs pull --include "multilingual/c4-ja.*.json.gz"

# -> Download *.josn.gz

# (Option) uncompress .gz (not keeping the original .gz)
gunzip *.jsonl.gz
