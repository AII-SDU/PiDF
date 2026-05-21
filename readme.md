
(正文)
python exp1.py \
  --csv ./mp_data/exp42_perovskite_like_subset.csv \
  --outdir runs/section1_logreg \
  --split random \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --seeds 0 1 2 3 4 \
  --save-clean-dataset

python3 plot_exp1.py


python3 exp2.py \
  --csv ./mp_data/exp42_perovskite_like_subset.csv \
  --outdir runs/section2_logreg \
  --protocols host family \
  --method-set full \
  --dedup-key auto \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --model-type logreg \
  --seeds 0 1 2 3 4

python exp3.py \
  --csv ./mp_data/exp42_perovskite_like_subset.csv \
  --outdir runs/section3_logreg \
  --split random \
  --dedup-key auto \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --model-type logreg \
  --seeds 0 1 2 3 4 \
  --save-clean-dataset

python exp4.py \
  --csv ./mp_data/exp42_perovskite_like_subset.csv \
  --outdir runs/section4_logreg \
  --split random \
  --dedup-key auto \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --model-type logreg \
  --initial-size 20 \
  --query-size 10 \
  --n-rounds 8 \
  --selected-strategy hybrid_0.25_0.45_0.30 \
  --seeds 0 1 2 3 4 \
  --save-query-history \
  --save-clean-dataset

python exp5.py \
  --labeled-csv ./mp_data/exp42_perovskite_like_subset.csv \
  --pool-csv ./mp_data/multicomponent_candidates.csv \
  --outdir runs/section5_logreg \
  --representation "PiDF + ML" \
  --label-mode ehull \
  --ehull-threshold 0.05 \
  --dedup-key auto \
  --pool-scoring-mode full_refit \
  --seeds 0 1 2 3 4 \
  --top-n 20 \
  --top-k-family 100 \
  --save-clean-datasets