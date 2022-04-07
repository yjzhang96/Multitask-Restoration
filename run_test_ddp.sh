python -m torch.distributed.launch --nproc_per_node=2 \
        test.py --config=configs/config_test.yaml
