# waits for a running kd run to end, then runs the given seed runs one after another on the same two gpus.
# usage: nohup bash chain.sh <pid> <its OUT_DIR> <gpus> <seed>:<script> ... > chain_<gpus>.log 2>&1 < /dev/null &
source ~/jayk_fml/env.sh
cd /tmp/jayk_runs/2_sequential_kd
pid=$1; first_out=$2; gpus=$3; shift 3

while kill -0 "$pid" 2>/dev/null; do sleep 30; done
echo "$(date -u) run $pid ($first_out) ended"
# its checkpoints go only if both stages were benchmarked and every file is on the hub
ls $first_out/results_stage2_*/summary.json > /dev/null 2>&1 && python verify_delete.py $first_out

for job in "$@"; do
  seed=${job%%:*}; script=${job#*:}
  out="$(grep -oP '^OUT_DIR = "\K[a-z_]+' $script)_seed$seed"
  # both gpus must be nearly empty, someone else may have taken them meanwhile
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpus | awk '$1 > 1000' | wc -l)" -ne 0 ]; do sleep 60; done
  echo "$(date -u) starting SEED=$seed $script -> $out"
  SEED=$seed CUDA_VISIBLE_DEVICES=$gpus python -u $script > ${script%.py}_seed$seed.log 2>&1
  code=$?
  echo "$(date -u) SEED=$seed $script exited with $code"
  [ $code -eq 0 ] && python verify_delete.py $out
done
echo "$(date -u) chain done"
