## single node multi-GPU
python -m torch.distributed.launch --nproc_per_node=2 \
        train.py --config=configs/config_train.yaml


## multi node multi GPU
python -m torch.distributed.launch --nproc_per_node=2 \
           --nnodes=2 --node_rank=0 --master_addr="192.168.66.228" \
           --master_port=1234 train.py --config=configs/config_train.yaml