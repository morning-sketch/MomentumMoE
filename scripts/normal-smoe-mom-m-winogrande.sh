mkdir -p /hy-tmp/checkpoint-normal-acc/

args="
--data /hy-tmp/data_directory/piqa/ \
--base_arch transformer \
--architecture smsmsmsmsmsm \
--gate_name smoe \
--nlayers 6 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 128 \
--attn-span 128 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 4000 \
--niter 2 \
--batch-sz 24 \
--batch-split 4 \
--nbatches 2500 \
--distributed \
--gamma1 1.0 \
--gamma2 1.0 \
--mu 0.7 \
--beta1 0.9 \
--beta2 0.999 \
--checkpoint /hy-tmp/checkpoint-normal-acc/smoe.pt \
--pretrained_weight /hy-tmp/normal-smoe/smoe.pt \
--cmp-sz 16 \
"

# echo "Training ..."
# CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=1 --use_env train_causal.py $args

echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=1 --use_env train_causal.py $args --resume --full-eval-mode
