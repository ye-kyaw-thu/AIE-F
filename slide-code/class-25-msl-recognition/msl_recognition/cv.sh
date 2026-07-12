#!/bin/bash

time bash scripts/07_cross_validate.sh bilstm     cv_bilstm     5
time bash scripts/07_cross_validate.sh transformer cv_transformer 5
time bash scripts/07_cross_validate.sh stgcn       cv_stgcn       5
