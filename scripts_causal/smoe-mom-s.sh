mkdir -p /hy-tmp/checkpoint/

args="
--data /hy-tmp/data_directory/wikitext-103/ \
--base_arch transformer \
--architecture smsmsd \
--gate_name smoe \
--nlayers 3 \
--hid-sz 128 \
--inner-hid-sz 128 \
--nheads 8 \
--block-sz 256 \
--attn-span 256 \
--dropout 0.7 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--niter 60 \
--batch-sz 12 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--gamma1 0.8 \
--gamma2 1.0 \
--mu 0.7 \
--beta1 0.9 \
--beta2 0.999 \
--checkpoint /hy-tmp/checkpoint/smoe.pt \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=1 --use_env train_causal.py $args

echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=1 --use_env train_causal.py $args --resume --full-eval-mode