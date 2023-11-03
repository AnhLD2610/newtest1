# Example:
# gpu=0 lr_mul=2 scale_emb_or_prj=prj bash train_multi30k_de_en.sh
# The better setting is:
# gpu=0 lr_mul=0.5 scale_emb_or_prj=emb bash train_multi30k_de_en.sh
CUDA_VISIBLE_DEVICES=${gpu} python3 train.py \
-label_smoothing \
-proj_share_weight \
-scale_emb_or_prj 'emb' \
-lr_mul 0.5 \
-warmup 4000 \
-epoch 10 \
-seed 1 \
-output_dir output \
-use_tb \
-proj_share_weight \
-embs_share_weight \
# -train_path "/media/data/thanhnb/newtest/data/C#/train.csv"\
# -val_path "/media/data/thanhnb/newtest/data/C#/test.csv"\
