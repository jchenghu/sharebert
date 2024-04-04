## ShareBERT: Embeddings Are Capable of Learning Hidden Layers

Implementation of the work "[ShareBERT: Embeddings Are Capable of Learning Hidden Layers](https://ojs.aaai.org/index.php/AAAI/article/view/29781)" [[Pdf](https://ojs.aaai.org/index.php/AAAI/article/view/29781/31348)],
presented at the 38th Annual AAAI Conference on Artificial Intelligence (AAAI-38).

## Requirements

Software:
- python (>= 3.7)
- torch 
- wandb
- deepspeed

More details can be found in `requirements.txt`.

Disk:
- We recommend at least 100 GB of free SSD disk memory for the data preparation.

GPU:
- Our models were trained on NVIDIA A100 (40GB), in case of lower device memory, 
 we suggest to decrease `--train_micro_batch_size_per_gpu`.

Data preparation details can be found in`./dataset/README.md`.

###### Disclaimer
At the moment the code has been tested on single GPU. Multi-GPU support will be added in the 
future. Feel free to open an issue if you need it urgently.

## Usage

### Pre-training

Assuming a single GPU the command for the training of ShareBERT Base:
```
python run_pretraining.py \
    --local_rank=0 \
    --model_type bert-mlm \
    --tokenizer_name bert-base-uncased \
    --hidden_act gelu \
    --hidden_size 2048 \
    --factor_size 384 \
    --num_hidden_layers 12 \
    --num_attention_heads 16 \
    --intermediate_size 4096 \
    --hidden_dropout_prob 0.1 \
    --attention_probs_dropout_prob 0.1 \
    --encoder_ln_mode pre-ln \
    --lr 1e-3 \
    --train_batch_size 4000 \
    --train_micro_batch_size_per_gpu 250 \
    --lr_schedule step \
    --curve linear \
    --warmup_proportion 0.06 \
    --gradient_clipping 0.5 \
    --optimizer_type adamw \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_eps 1e-6 \
    --max_steps 23000 \
    --dataset_path <hdf5_dir> \
    --output_dir <save_path> \
    --print_steps 100 \
    --num_epochs_between_checkpoints 3000 \
    --job_name pretraining_experiment \
    --project_name sharebert-pretraining \
    --validation_epochs 6 \
    --validation_epochs_begin 1 \
    --validation_epochs_end 1 \
    --validation_begin_proportion 0.05 \
    --validation_end_proportion 0.01 \
    --validation_micro_batch 16 \
    --deepspeed \
    --data_loader_type dist \
    --do_validation \
    --seed 42 \
    --fp16 \
    --fp16_backend ds \
    --layer_norm_type pytorch \
    --total_training_time 3000.0 \
    --early_exit_time_marker 3000.0 &> output.txt &
```
training will use the samples located in `<hdf5_dir>` and the final model will be saved in `<save_path>`. 
Set `factor_size` to 128 for ShareBERT Small, 384 in case of ShareBERT Base, and 768 for ShareBERT Large.
In the latter case, set also `num_hidden_layers` to 6.

It might be necessary to set `CUDA_VISIBLE_DEVICES=0 python run_pretrainin.py ...` in some environments.

### Fine-Tuning

Fine-Tuning command:
```
python run_glue.py \
     --local_rank=0 \
     --model_name_or_path <save_path>/epoch... \
     --task_name <task_name> \
     --max_seq_length 128 \
     --output_dir ./finetuning-out/ \
     --overwrite_output_dir \
     --do_train --do_eval --do_predict \
      --evaluation_strategy steps \
      --per_device_train_batch_size 64 --gradient_accumulation_steps 1 \
      --per_device_eval_batch_size 64 \
      --learning_rate 5e-5 \
      --weight_decay 0.01 \
      --eval_steps 50 --evaluation_strategy steps \
      --max_grad_norm 1.0 \
      --num_train_epochs 5 \
      --fp16 \
      --fp16_backend apex \
      --lr_scheduler_type polynomial \
      --warmup_steps 50 &> fine_output.txt &
```
where `<task_name>` can be `sst2, mnli, qqp, mrpc, cola, stsb, rte,` or `qnli`

### Pretrained models

Pretrained models can be found in the following [drive](https://drive.google.com/file/d/1U36Bov_C-EjOGJmXeNqWzLw7_eEb344l/view?usp=sharing).

## Acknowledgments

If you find this repository useful, please consider citing (no obligation):
```
@inproceedings{hu2024sharebert,
  title={ShareBERT: Embeddings Are Capable of Learning Hidden Layers},
  author={Hu, Jia Cheng and Cavicchioli, Roberto and Berardinelli, Giulia and Capotondi, Alessandro},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={18225--18233},
  year={2024}
}
```

Repository is based on [academy-budget-bert](https://github.com/IntelLabs/academic-budget-bert/),
we thank the authors for the beautiful code and the sharing of their work.