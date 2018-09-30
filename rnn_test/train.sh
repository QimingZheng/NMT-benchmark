#!/bin/bash

if [ ! -d data ]; then
  ln -s /data/data1/v-qizhe/data data
fi

#export CUDA_VISIBLE_DEVICES="4,5"
export CUDA_VISIBLE_DEVICES="0,1"

python train.py \
    --mode="train"\
    --src_file_name="data/train.en"\
    --tgt_file_name="data/train.de"\
    --src_vocab_file="data/vocab.50K.en"\
    --tgt_vocab_file="data/vocab.50K.de"\
    --batch_size=128 \
    --variable_update="parameter_server" \
    --independent_replica="true" \
    --dropout=0.4 \
    --unit_type="lstm" \
    --num_units=512 \
    --beam_width=10 \
    --forget_bias=0.8 \
    --embedding_dim=512 \
    --use_attention=True \
    --src_max_len=50 \
    --tgt_max_len=50 \
    --num_encoder_layers=4 \
    --num_decoder_layers=4 \
    --encoder_type="bi" \
    --direction="bi" \
    --max_gradient_norm=5.0 \
    --learning_rate 0.0003 \
#    --prefetch_data_to_device="true" \
