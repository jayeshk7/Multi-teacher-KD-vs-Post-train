# runs one lane: the seeds (SEEDS, default 0 1 2) of one recipe or sequential chain, one after another on the given gpus.
# a lane that starts from a first stage waits for that stage's log.json. finished runs are skipped, so a lane can restart
# run: SEEDS=0 nohup bash run.sh <lane> <gpus> > logs/lane_<lane>.log 2>&1 &
#
# lanes on 2 gpus (student, both teachers): multi_kd
# lanes on 1 gpu: kd_math_chain, kd_code_chain, multi_sft, rl_code, rl_math, multi_rl, refs, instruct_a6000

LANE=$1
export CUDA_VISIBLE_DEVICES=$2
# avoids gpu memory fragmentation from the growing kv caches
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INSTRUCT=Qwen/Qwen2.5-0.5B-Instruct
mkdir -p logs

sft_kd() {  # name start domains loss seed
    [ -f $1/results_last/summary.json ] && { echo "skip $1"; return; }
    echo "$(date -u +%F' '%T) start $1"
    SEED=$5 START=$2 DOMAINS=$3 LOSS=$4 OUT=$1 python -u train_sft_kd.py > logs/$1.log 2>&1
    rc=$?
    echo "$(date -u +%F' '%T) end $1, exit $rc"
}

grpo() {  # name start domains seed
    [ -f $1/results_best/summary.json ] && { echo "skip $1"; return; }
    echo "$(date -u +%F' '%T) start $1"
    SEED=$4 START=$2 DOMAINS=$3 OUT=$1 python -u train_grpo.py > logs/$1.log 2>&1
    rc=$?
    echo "$(date -u +%F' '%T) end $1, exit $rc"
}

evaluate() {  # name model
    [ -f $1/results/summary.json ] && { echo "skip $1"; return; }
    echo "$(date -u +%F' '%T) start $1"
    MODEL=$2 OUT=$1 python -u eval_model.py > logs/$1.log 2>&1
    rc=$?
    echo "$(date -u +%F' '%T) end $1, exit $rc"
}

wait_for() {  # a first stage's training has ended, its best adapter is final
    until [ -f $1/log.json ]; do sleep 60; done
}

# gpu check: instruct's gsm8k answers in fp32 instead of bf16 (check/fp32_gen.py), once per gpu type
if [ $LANE = fp32_check ]; then
    echo "$(date -u +%F' '%T) start eval_fp32_gpu$2"
    python -u check/fp32_gen.py > logs/eval_fp32_gpu$2.log 2>&1
    echo "$(date -u +%F' '%T) end eval_fp32_gpu$2, exit $?"
    exit
fi

# reference models, once
if [ $LANE = refs ]; then
    evaluate eval_instruct $INSTRUCT
    evaluate eval_teacher_math Qwen/Qwen2.5-Math-7B-Instruct
    evaluate eval_teacher_code Qwen/Qwen2.5-Coder-7B-Instruct
    exit
fi
# instruct again on the a6000, because greedy answers depend on the gpu type
if [ $LANE = instruct_a6000 ]; then
    evaluate eval_instruct_a6000 $INSTRUCT
    exit
fi

for s in ${SEEDS:-0 1 2}; do
    case $LANE in
        # part I: kd and sft
        kd_math_chain) sft_kd inst_kd_math_s$s $INSTRUCT math kd $s
                       sft_kd inst_seq_kd_math_then_kd_code_s$s inst_kd_math_s$s/best code kd $s ;;
        kd_code_chain) sft_kd inst_kd_code_s$s $INSTRUCT code kd $s
                       sft_kd inst_seq_kd_code_then_kd_math_s$s inst_kd_code_s$s/best math kd $s ;;
        multi_kd)      sft_kd inst_multi_kd_s$s $INSTRUCT math,code kd $s ;;
        multi_sft)     sft_kd inst_multi_gt_sft_s$s $INSTRUCT math,code gt_sft $s
                       sft_kd inst_multi_t_sft_s$s $INSTRUCT math,code t_sft $s ;;
        # part II: rl, the sequential ones start from part I's first stages
        rl_code)       wait_for inst_kd_math_s$s
                       grpo inst_seq_kd_math_then_rl_code_s$s inst_kd_math_s$s/best code $s ;;
        rl_math)       wait_for inst_kd_code_s$s
                       grpo inst_seq_kd_code_then_rl_math_s$s inst_kd_code_s$s/best math $s ;;
        multi_rl)      grpo inst_multi_rl_s$s $INSTRUCT math,code $s ;;
        # check run: kd math at lora rank 48 (alpha 96), 1 seed. waits until multi_kd seed 2 frees the gpu
        kd_math_r48)   until [ -f inst_multi_kd_s2/results_last/summary.json ]; do sleep 60; done
                       LORA_R=48 sft_kd inst_kd_math_r48_s$s $INSTRUCT math kd $s ;;
    esac
done
