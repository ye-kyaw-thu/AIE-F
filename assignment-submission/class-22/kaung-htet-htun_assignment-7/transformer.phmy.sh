#!/bin/bash

## Written by Ye Kyaw Thu, Affiliated Professor, CADT, Cambodia
## for NMT Experiments between Burmese and Ethnic Languages
## used Marian NMT Framework for training
## Last updated: 23 May 2022


model_folder="model.transformer.phmy";
mkdir ${model_folder};
data_path="/home/phantom/Documents/assignment_7_testing/kaung-htet-htun_assignment-7/g2p-par";
src="ph"; tgt="my";

~/marian/build/marian \
    --model ${model_folder}/model.npz --type transformer \
    --train-sets ${data_path}/train.${src} ${data_path}/train.${tgt} \
    --max-length 200 \
    --vocabs ${data_path}/vocab/vocab.${src}.yml ${data_path}/vocab/vocab.${tgt}.yml \
    --mini-batch 64 -w 512 --maxi-batch 100 \
    --early-stopping 10 \
    --valid-freq 10000 --save-freq 10000 --disp-freq 500 \
    --valid-metrics cross-entropy perplexity bleu \
    --valid-sets ${data_path}/dev.${src} ${data_path}/dev.${tgt} \
    --valid-translation-output ${model_folder}/valid.${src}-${tgt}.output --quiet-translation \
    --valid-mini-batch 64 \
    --beam-size 4 --normalize 0.6 \
    --log ${model_folder}/train.log --valid-log ${model_folder}/valid.log \
    --enc-depth 2 --dec-depth 2 \
    --dim-emb 256 \
    --transformer-dim-ffn 1024 \
    --transformer-heads 4 \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --transformer-dropout 0.3 --label-smoothing 0.1 \
    --learn-rate 0.0003 --lr-warmup 0 --lr-decay-inv-sqrt 16000 --lr-report \
    --clip-norm 5 \
    --tied-embeddings \
    --devices 0 --sync-sgd --seed 1111 \
    --dump-config > ${model_folder}/${src}-${tgt}.config.yml
    
time ~/marian/build/marian -c ${model_folder}/${src}-${tgt}.config.yml  2>&1 | tee ${model_folder}/transformer-${src}-${tgt}.log