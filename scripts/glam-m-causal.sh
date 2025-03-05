mkdir -p /hy-tmp/checkpoint/

args="
--data /hy-tmp/data_directory/wikitext-103/ \
--base_arch glam \
--architecture sgsfsgsfsgsfsgsfsgsfscsf \
--gate_name smoe \
--nlayers 6 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 32 \
--attn-span 128 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.00007 \
--lr-warmup 4000 \
--niter 120 \
--batch-sz 2 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /hy-tmp/checkpoint/glam-m-causal.pt \
"
export PYTHONOPTIMIZE=1
echo "Training ..."
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=1 --use_env train_causal.py $args

echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=1 --use_env train_causal.py $args --resume --full-eval-mode