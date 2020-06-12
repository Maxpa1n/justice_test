##### run
```bash
export  CUDA_VISIBLE_DEVICES="1,0"
python3 run_multiple_choice.py   --model_name_or_path  albert_model_pretrain/  \
                                --task_name  justice_race  \
                                --output_dir  output/  \
                                --data_dir  data/  \
                                --do_train  \
                                --do_eval  \
                                --per_device_train_batch_size  2 \
                                --per_device_eval_batch_size  2 \
                                --num_train_epochs 3 \
                                --max_seq_length  256 \
                                --overwrite_output_dir \
                                --learning_rate 5e-5
```


##### distributed run
```Bash
export  CUDA_VISIBLE_DEVICES="2,3"
python -m torch.distributed.launch --nproc_per_node 2  run_multiple_choice.py  --model_name_or_path  albert_model_pretrain/  \
                                --evaluate_during_training   \
                                --task_name  justice_race  \
                                --output_dir  output/  \
                                --data_dir  data/  \
                                --do_train  \
                                --do_eval  \
                                --per_device_train_batch_size  2 \
                                --per_device_eval_batch_size  2 \
                                --num_train_epochs 3 \
                                --max_seq_length  256 \
                                --overwrite_output_dir \
                                --learning_rate 1e-5
```