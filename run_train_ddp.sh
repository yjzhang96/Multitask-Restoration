## single node multi-GPU
python -m torch.distributed.launch --nproc_per_node=2 \
        train.py --config=configs/config_test.yaml
