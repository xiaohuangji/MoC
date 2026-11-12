"""Single-node, four-rank runtime shared by the three paper entry points."""
from contextlib import contextmanager
from datetime import timedelta
import os

import torch
import torch.distributed as dist

from io_utils import verify_base


@contextmanager
def four_gpus(base):
    if int(os.environ.get("WORLD_SIZE", 0)) != 4 or int(os.environ.get("LOCAL_WORLD_SIZE", 0)) != 4:
        raise RuntimeError("Use torchrun with four GPUs on the same Linux host")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(4)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    try:
        if dist.get_rank() == 0:
            verify_base(base)
        dist.barrier()
        yield dist.get_rank(), device
    finally:
        dist.destroy_process_group()
