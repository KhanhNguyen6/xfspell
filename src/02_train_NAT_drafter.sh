#!/bin/bash

mkdir -p models

bin_path=bin/gtc
lr=0.0005
dropout=0.1
warmup=4000
size=25
seed=1
max_tokens=4096
update_freq=1
update=1000000

# path to GAD repository
PATH_TO_GAD=/home/user/unilm/decoding/GAD

python3 $PATH_TO_GAD/train.py ${bin_path} --arch block \
    --noise block_mask --share-all-embeddings --criterion glat_loss --label-smoothing 0.1 \
    --lr ${lr} --warmup-init-lr 1e-7 --stop-min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates ${warmup} \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_lev_modified --max-tokens ${max_tokens} \
    --weight-decay 0.01 --dropout ${dropout} --encoder-layers 6 --encoder-embed-dim 256 --decoder-layers 6 \
    --decoder-embed-dim 256 --fp16 --max-source-positions 1000 --max-target-positions 1000 --max-update ${update}\
    --seed ${seed} --clip-norm 5 --save-dir ./models/nat_drafter01 \
    --src-embedding-copy --log-interval 10 --log-format json \
    --user-dir block_plugins --block-size ${size} --total-up ${update} \
    --update-freq ${update_freq} --decoder-learned-pos --encoder-learned-pos --apply-bert-init --activation-fn gelu \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints \
    --max-epoch 1
