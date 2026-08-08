#!/usr/bin/env bash
cd /workspace/exp/ph-my/mert-work
/workspace/tools/mosesdecoder/bin/extractor --sctype BLEU --scconfig case:true  --scfile run4.scores.dat --ffile run4.features.dat -r /workspace/exp/clean-data/dev.my -n run4.best100.out.gz
