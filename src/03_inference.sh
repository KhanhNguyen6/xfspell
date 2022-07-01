#!/bin/bash

data_dir=bin/gtc # the dir that contains dict files
checkpoint_path=models/nat_drafter01/checkpoint_best.pt # the dir that contains NAT drafter checkpoint
AR_checkpoint_path=models/at_verifier01/checkpoint_best.pt # the dir that contains AT verifier checkpoint
input_path=data/gtc/test.fr # the dir that contains bpe test files
output_path=data/gtc/test.en # the dir for outputs

strategy='gad' # fairseq, AR, gad
batch=32
beam=5

beta=5
tau=3.0
block_size=25

src=fr
tgt=en

# path to GAD repository
PATH_TO_GAD=/home/user/unilm/decoding/GAD

python3 $PATH_TO_GAD/inference.py ${data_dir} --path ${checkpoint_path} \
      --user-dir block_plugins --task translation_lev_modified --remove-bpe --max-sentences 20 --source-lang ${src} \
      --target-lang ${tgt} --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 \
      --gen-subset test --AR-path ${AR_checkpoint_path} --input-path ${input_path} --output-path ${output_path} \
      --block-size ${block_size} --beta ${beta} --tau ${tau} --batch ${batch} --beam ${beam} --strategy ${strategy}