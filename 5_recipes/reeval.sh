# benchmarks each run's best checkpoint again on the rtx 6000 ada, so every reported number comes from one gpu type.
# runs already benchmarked on the ada are skipped (inst_kd_math_s0 is redone as a check that both paths agree).
# one eval at a time, only while the ada's queue runs a grpo job or is empty and <= 25 gb of gpu memory is in use.
# pauses while check/reeval.pause exists. a failed eval is marked and not retried
# run: nohup bash reeval.sh 3 queues/reeval.txt > logs/lane_reeval.log 2>&1 &
export CUDA_VISIBLE_DEVICES=$1
LIST=$2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p eval_ada logs

while true; do
    left=0
    started=0
    for run in $(cat $LIST); do
        own=$run/results_best/summary.json
        [ -f eval_ada/${run}_best/results/summary.json ] && continue
        [ -f eval_ada/${run}_best/failed ] && continue
        [ $run != inst_kd_math_s0 ] && [ -f $own ] && grep -q "6000 Ada" $own && continue
        left=$((left + 1))
        grep -qs heldout_code $own || continue
        [ -f check/reeval.pause ] && break
        grep -E "job:|queue empty" logs/queue_gpu$1.log | tail -1 | grep -qE "job: (multi_rl|rl_code|rl_math) |queue empty" || break
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)
        [ $used -gt 25000 ] && break
        echo "$(date -u +%F' '%T) start eval_ada $run"
        MODEL=$run/best OUT=eval_ada/${run}_best python -u eval_model.py > logs/eval_ada_$run.log 2>&1
        rc=$?
        [ $rc -ne 0 ] && mkdir -p eval_ada/${run}_best && touch eval_ada/${run}_best/failed
        echo "$(date -u +%F' '%T) end eval_ada $run, exit $rc"
        started=1
        break
    done
    [ $left -eq 0 ] && break
    [ $started -eq 0 ] && sleep 300
done
echo "$(date -u +%F' '%T) reeval done"
