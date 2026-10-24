# deletes a finished run's local checkpoints once every file is on the hub with the same sha256 (weights) or size
import glob
import hashlib
import os
import shutil
import sys

from huggingface_hub import HfApi

out_dir = sys.argv[1]
local = sorted(f for f in glob.glob(f"{out_dir}/stage*/*") if os.path.isfile(f))
info = {p.path: p for p in HfApi().get_paths_info("jajamken/FML_project", local)} if local else {}


def on_hub(f):
    p = info.get(f)
    if p is None:
        return False
    if p.lfs:
        return hashlib.sha256(open(f, "rb").read()).hexdigest() == p.lfs.sha256
    return os.path.getsize(f) == p.size


if local and all(on_hub(f) for f in local):
    for d in glob.glob(f"{out_dir}/stage*"):
        shutil.rmtree(d)
    print(f"{out_dir}: all {len(local)} files on the hub, local checkpoints deleted")
else:
    print(f"{out_dir}: not all files on the hub, local checkpoints kept")
