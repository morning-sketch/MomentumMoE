mkdir -p /path/to/checkpoint/directory/

args="
--data /hy-tmp/MomentumSMoE/data_directory/wikitext-103/ \
--base_arch transformer \
--architecture smsmsm \
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
--batch-sz 96 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--gamma1 0.8 \
--gamma2 1.0 \
--mu 0.7 \
--beta1 0.9 \
--beta2 0.999 \
--checkpoint /path/to/checkpoint/directory/smoe.pt \
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=4 --use_env train_causal.py $args

echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=4 --use_env train_causal.py $args --resume --full-eval-mode