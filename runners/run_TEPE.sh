CUDA_VISIBLE_DEVICES=2 python3 main.py\
    --n_blocks=2\
    --batch_size=256\
    --window_size=60\
    --train_split=0.8\
    --name=TEPE\
    > TEPE.log 2>&1