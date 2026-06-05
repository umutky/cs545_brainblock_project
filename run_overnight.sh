#!/bin/bash
# ============================================================
# CS545 BrainBlock — Overnight Training
#
# 5 deney, sıralı, seed=42:
#   1. PPO  R2  MLP      mask  no-div   2M steps
#   2. PPO  R2  MLP      mask  div=1.0  2M steps
#   3. PPO  R1  MLP      mask  no-div   2M steps  (sparse karşılaştırma)
#   4. PPO  R2  CNN+MLP  mask  no-div   3M steps  lr=1e-4
#   5. SAC  R2  MLP      mask  no-div   5M steps  (en sonda)
#
# Her train sonrası stochastic + deterministic eval (final + best model).
#
# Çalıştır:
#   bash run_overnight.sh 2>&1 | tee logs/overnight.txt
# ============================================================

source venv/bin/activate
set -e

SEED=42

log() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "  $(date)"
    echo "════════════════════════════════════════════════════════════"
}

eval_ppo() {
    local MODEL_PATH=$1
    local ENCODER=$2
    local REWARD=$3
    local OUT=$4
    local MODE=$5   # "stochastic" veya "deterministic"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIP (model bulunamadı): $MODEL_PATH"
        return
    fi
    local STOCH_FLAG=""
    [ "$MODE" = "stochastic" ] && STOCH_FLAG="--stochastic"
    python -m member_umut.evaluate \
        --model "$MODEL_PATH" \
        --encoder "$ENCODER" \
        --reward "$REWARD" \
        --episodes 1000 \
        --seed $SEED \
        --render-solutions 10 \
        --render-trace \
        --output-dir "$OUT" \
        $STOCH_FLAG
}

eval_sac() {
    local MODEL_PATH=$1
    local ENCODER=$2
    local REWARD=$3
    local OUT=$4
    local MODE=$5
    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIP (model bulunamadı): $MODEL_PATH"
        return
    fi
    local STOCH_FLAG=""
    [ "$MODE" = "stochastic" ] && STOCH_FLAG="--stochastic"
    python -m member_umut.sac_evaluate \
        --model "$MODEL_PATH" \
        --encoder "$ENCODER" \
        --reward "$REWARD" \
        --episodes 1000 \
        --seed $SEED \
        --render-solutions 10 \
        --render-trace \
        --output-dir "$OUT" \
        $STOCH_FLAG
}

log "BAŞLADI — 5 deney, seed=$SEED"
echo "Tahmini süre: ~26 saat"
echo ""

# ─────────────────────────────────────────────────────────────
# 1. PPO R2 MLP — temiz baseline
# ─────────────────────────────────────────────────────────────
log "1/5  TRAIN  PPO R2 MLP  (2M, no-div)"
python -m member_umut.train \
    --reward shaped \
    --encoder mlp \
    --seed $SEED \
    --total-timesteps 2000000 \
    --lr 3e-4 \
    --entropy-coef 0.01 \
    --log-interval 10 \
    --save-interval 100 \
    --output-dir results/ppo_r2_mlp_seed42

log "1/5  EVAL  PPO R2 MLP — stochastic (final + best)"
eval_ppo results/ppo_r2_mlp_seed42/final_model.pt mlp shaped \
    results/ppo_r2_mlp_seed42/eval_final_stoch stochastic
eval_ppo results/ppo_r2_mlp_seed42/best_model.pt mlp shaped \
    results/ppo_r2_mlp_seed42/eval_best_stoch stochastic

log "1/5  EVAL  PPO R2 MLP — deterministic (final + best)"
eval_ppo results/ppo_r2_mlp_seed42/final_model.pt mlp shaped \
    results/ppo_r2_mlp_seed42/eval_final_det deterministic
eval_ppo results/ppo_r2_mlp_seed42/best_model.pt mlp shaped \
    results/ppo_r2_mlp_seed42/eval_best_det deterministic

# ─────────────────────────────────────────────────────────────
# 2. PPO R2 MLP + Diversity Bonus
# ─────────────────────────────────────────────────────────────
log "2/5  TRAIN  PPO R2 MLP  (2M, div=1.0)"
python -m member_umut.train \
    --reward shaped \
    --encoder mlp \
    --seed $SEED \
    --total-timesteps 2000000 \
    --lr 3e-4 \
    --entropy-coef 0.01 \
    --diversity-bonus 1.0 \
    --log-interval 10 \
    --save-interval 100 \
    --output-dir results/ppo_r2_mlp_div1_seed42

log "2/5  EVAL  PPO R2 MLP div=1.0 — stochastic (final + best)"
eval_ppo results/ppo_r2_mlp_div1_seed42/final_model.pt mlp shaped \
    results/ppo_r2_mlp_div1_seed42/eval_final_stoch stochastic
eval_ppo results/ppo_r2_mlp_div1_seed42/best_model.pt mlp shaped \
    results/ppo_r2_mlp_div1_seed42/eval_best_stoch stochastic

log "2/5  EVAL  PPO R2 MLP div=1.0 — deterministic (final + best)"
eval_ppo results/ppo_r2_mlp_div1_seed42/final_model.pt mlp shaped \
    results/ppo_r2_mlp_div1_seed42/eval_final_det deterministic
eval_ppo results/ppo_r2_mlp_div1_seed42/best_model.pt mlp shaped \
    results/ppo_r2_mlp_div1_seed42/eval_best_det deterministic

# ─────────────────────────────────────────────────────────────
# 3. PPO R1 MLP — sparse reward karşılaştırması
# ─────────────────────────────────────────────────────────────
log "3/5  TRAIN  PPO R1 MLP  (2M, sparse, no-div)"
python -m member_umut.train \
    --reward sparse \
    --encoder mlp \
    --seed $SEED \
    --total-timesteps 2000000 \
    --lr 3e-4 \
    --entropy-coef 0.01 \
    --log-interval 10 \
    --save-interval 100 \
    --output-dir results/ppo_r1_mlp_seed42

log "3/5  EVAL  PPO R1 MLP — stochastic (final + best)"
eval_ppo results/ppo_r1_mlp_seed42/final_model.pt mlp sparse \
    results/ppo_r1_mlp_seed42/eval_final_stoch stochastic
eval_ppo results/ppo_r1_mlp_seed42/best_model.pt mlp sparse \
    results/ppo_r1_mlp_seed42/eval_best_stoch stochastic

log "3/5  EVAL  PPO R1 MLP — deterministic (final + best)"
eval_ppo results/ppo_r1_mlp_seed42/final_model.pt mlp sparse \
    results/ppo_r1_mlp_seed42/eval_final_det deterministic
eval_ppo results/ppo_r1_mlp_seed42/best_model.pt mlp sparse \
    results/ppo_r1_mlp_seed42/eval_best_det deterministic

# ─────────────────────────────────────────────────────────────
# 4. PPO R2 CNN+MLP — mimari karşılaştırması
# ─────────────────────────────────────────────────────────────
log "4/5  TRAIN  PPO R2 CNN+MLP  (3M, lr=1e-4, no-div)"
python -m member_umut.train \
    --reward shaped \
    --encoder cnn_mlp \
    --seed $SEED \
    --total-timesteps 3000000 \
    --lr 1e-4 \
    --entropy-coef 0.01 \
    --log-interval 10 \
    --save-interval 100 \
    --output-dir results/ppo_r2_cnn_seed42

log "4/5  EVAL  PPO R2 CNN+MLP — stochastic (final + best)"
eval_ppo results/ppo_r2_cnn_seed42/final_model.pt cnn_mlp shaped \
    results/ppo_r2_cnn_seed42/eval_final_stoch stochastic
eval_ppo results/ppo_r2_cnn_seed42/best_model.pt cnn_mlp shaped \
    results/ppo_r2_cnn_seed42/eval_best_stoch stochastic

log "4/5  EVAL  PPO R2 CNN+MLP — deterministic (final + best)"
eval_ppo results/ppo_r2_cnn_seed42/final_model.pt cnn_mlp shaped \
    results/ppo_r2_cnn_seed42/eval_final_det deterministic
eval_ppo results/ppo_r2_cnn_seed42/best_model.pt cnn_mlp shaped \
    results/ppo_r2_cnn_seed42/eval_best_det deterministic

# ─────────────────────────────────────────────────────────────
# 5. SAC R2 MLP — off-policy karşılaştırma (en sonda)
# ─────────────────────────────────────────────────────────────
log "5/5  TRAIN  SAC R2 MLP  (5M, buffer=1M, batch=512)"
python -m member_umut.sac_train \
    --reward shaped \
    --encoder mlp \
    --seed $SEED \
    --total-timesteps 5000000 \
    --lr 1e-4 \
    --lr-alpha 3e-4 \
    --buffer-size 1000000 \
    --batch-size 512 \
    --learning-starts 20000 \
    --log-interval 10000 \
    --save-interval 500000 \
    --output-dir results/sac_r2_mlp_seed42

log "5/5  EVAL  SAC R2 MLP — stochastic (final + best)"
eval_sac results/sac_r2_mlp_seed42/final_model.pt mlp shaped \
    results/sac_r2_mlp_seed42/eval_final_stoch stochastic
eval_sac results/sac_r2_mlp_seed42/best_model.pt mlp shaped \
    results/sac_r2_mlp_seed42/eval_best_stoch stochastic

log "5/5  EVAL  SAC R2 MLP — deterministic (final + best)"
eval_sac results/sac_r2_mlp_seed42/final_model.pt mlp shaped \
    results/sac_r2_mlp_seed42/eval_final_det deterministic
eval_sac results/sac_r2_mlp_seed42/best_model.pt mlp shaped \
    results/sac_r2_mlp_seed42/eval_best_det deterministic

# ─────────────────────────────────────────────────────────────
log "TÜM DENEYLER TAMAMLANDI"
echo ""
echo "Sonuçlar:"
for d in results/ppo_r2_mlp_seed42 results/ppo_r2_mlp_div1_seed42 \
          results/ppo_r1_mlp_seed42 results/ppo_r2_cnn_seed42 \
          results/sac_r2_mlp_seed42; do
    if [ -d "$d/eval_best_stoch" ]; then
        succ=$(python3 -c "import json; d=json.load(open('$d/eval_best_stoch/eval_results.json')); print(f\"{d['success_rate']:.3f}\")" 2>/dev/null || echo "?")
        uniq=$(python3 -c "import json; d=json.load(open('$d/eval_best_stoch/eval_results.json')); print(d['unique_solutions_found'])" 2>/dev/null || echo "?")
        echo "  $(basename $d): success=$succ  unique=$uniq"
    fi
done
