#!/bin/bash

mkdir -p bin/gtc

cat data/gtc/all.tsv | cut -f1 | python3 src/tokenize.py > data/gtc/all.tok.fr
cat data/gtc/all.tsv | cut -f2 | python3 src/tokenize.py > data/gtc/all.tok.en

cat data/gtc/all.tok.fr | awk 'NR%20!=0' > data/gtc/train.tok.fr
cat data/gtc/all.tok.fr | awk 'NR%20==0' > data/gtc/dev.tok.fr

cat data/gtc/all.tok.en | awk 'NR%20!=0' > data/gtc/train.tok.en
cat data/gtc/all.tok.en | awk 'NR%20==0' > data/gtc/dev.tok.en

fairseq-preprocess --source-lang fr --target-lang en \
    --joined-dictionary \
    --trainpref data/gtc/train.tok \
    --validpref data/gtc/dev.tok \
    --destdir bin/gtc