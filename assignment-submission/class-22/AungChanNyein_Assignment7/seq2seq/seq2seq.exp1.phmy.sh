#!/bin/bash
export PATH=/kaggle/working/bins:$PATH
model_folder="/kaggle/working/nmt/model.seq2seq.exp1.phmy"

cat > $model_folder/config.yml << 'EOF'
type: s2s
train-sets:
  - /kaggle/working/nmt/g2p-par/train.ph
  - /kaggle/working/nmt/g2p-par/train.my
valid-sets:
  - /kaggle/working/nmt/g2p-par/dev.ph
  - /kaggle/working/nmt/g2p-par/dev.my
vocabs:
  - /kaggle/working/nmt/g2p-par/vocab/vocab.ph.yml
  - /kaggle/working/nmt/g2p-par/vocab/vocab.my.yml
model: /kaggle/working/nmt/model.seq2seq.exp1.phmy/model.npz
max-length: 100
mini-batch: 64
maxi-batch: 100
workspace: 3000
enc-depth: 2
enc-type: alternating
enc-cell: lstm
enc-cell-depth: 2
dec-depth: 2
dec-cell: lstm
dec-cell-base-depth: 2
dec-cell-high-depth: 2
tied-embeddings: true
layer-normalization: true
skip: true
dropout-rnn: 0.3
dropout-src: 0.3
learn-rate: 0.0003
lr-decay-inv-sqrt: 16000
lr-report: true
valid-freq: 5000
save-freq: 5000
disp-freq: 500
valid-metrics:
  - cross-entropy
  - perplexity
  - bleu
early-stopping: 10
beam-size: 12
normalize: 0
log: /kaggle/working/nmt/model.seq2seq.exp1.phmy/train.log
valid-log: /kaggle/working/nmt/model.seq2seq.exp1.phmy/valid.log
overwrite: true
keep-best: true
seed: 1111
devices: [0, 1]
sync-sgd: true
quiet-translation: true
EOF

marian --config $model_folder/config.yml
