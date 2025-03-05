mkdir -p /path/to/checkpoint/directory/

args="
--data /hy-tmp/MomentumSMoE/data_directory/wikitext-103/ \
--base_arch transformer \
--architecture sgsgsgsgsgsgsgsgsgsgsgsg \
--gate_name smoe \
--nlayers 12 \
--hid-sz 512 \
--inner-hid-sz 512 \
--nheads 8 \
--block-sz 1024 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 5000 \
--niter 80 \
--batch-sz 24 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /path/to/checkpoint/directory/smoe.pt \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=4 --use_env train_causal.py $args

echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=4 --use_env train_causal.py $args --resume --full-eval-mode
