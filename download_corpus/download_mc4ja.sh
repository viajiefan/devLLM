# Download MC4 (Ja) https://huggingface.co/datasets/allenai/c4
cd ../data
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
time git lfs pull --include "multilingual/c4-ja.*.json.gz"

# ->  *.josn.gz

# (Option) uncompress .gz (not keeping the original .gz)
# gunzip *.json.gz

# (Option) concat
# cat c4-ja*.jsonl > tmp/c4-ja.jsonl

# (Option) split (job時間制限を考えて1file 5G以下に)
# 行の途中で分割しない https://qiita.com/b-mente/items/0a57e65687d67b4ac582
# N file(ここでは160)に分割する場合　出力ファイルベースファイル名+数字出力
# split -n l/160 -d --additional-suffix=.jsonl c4-ja.jsonl c4-ja-
# -> c4-ja-00, ...
