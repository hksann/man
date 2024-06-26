python main_plot.py \
    --image_dir /kaggle/input/images/iu_xray/images \
    --ann_path /kaggle/input/images/iu_xray/annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 100 \
    --batch_size 1 \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 3 \
    --topk 32 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 7580 \
    --beam_size 1 \
    --save_dir results/iu_xray/ \
    --log_period 1000 \
    --load /kaggle/working/man/results/iu_xray/model_best.pth
