"""Create the one shared K25 FFN overlay from pinned, unlabeled C4 windows."""
import argparse
import gc
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--c4-dir", type=Path, required=True, help="Directory containing the eight C4 gzip shards")
    parser.add_argument("--output", type=Path, required=True, help="New directory on shared local storage")
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    import recon_capture as capture
    import recon_storage as storage
    from distributed import four_gpus
    from glue_model import build
    from io_utils import CONFIG, read, sha, tensor_sha, write
    from prepare_c4 import prepare
    from recon_fit import fit, required_indices

    out = args.output
    with four_gpus(args.base) as (rank, device):
        if rank == 0:
            out.mkdir(parents=True, exist_ok=False)
            prepare(args.base, args.c4_dir, out / "windows.json")
        dist.barrier()
        groups = read(out / "windows.json")
        model = build(args.base, "mnli", device).eval().requires_grad_(False)
        records = {k: capture.initial(model, groups[k][rank::4], device, storage.RAMRecords())
                   for k in ("fit", "heldout")}
        del groups
        counts = [16384 * 128 // 4] * 4
        assert records["fit"].tokens == counts[rank]
        layers = model.base_model.model.model.layers
        for index, block in enumerate(layers):
            required = required_indices(counts, index, rank)
            owner, indices = storage.mapped(None, "required", (len(required),), torch.int64)
            indices.copy_(required)
            del required
            previous = layers[index - 1] if index else None
            train, train_handles = capture.capture(block, records["fit"], device, indices, previous)
            held, held_handles = capture.capture(block, records["heldout"], device, None, previous)
            write(out / f"capture_{index:02d}_rank{rank}.json",
                  {"cache_sha256": {"fit_" + n: tensor_sha(v) for n, v in
                                    zip(("input", "residual", "target"), train)}})
            resident = tuple(t.to(device) for t in held)
            fit(block, (train, indices, counts[rank]), resident, index, device, rank, out)
            del train, held, indices, resident
            capture.release(train_handles + held_handles + [owner])
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
        if rank == 0:
            entries = [read(out / f"layer_{i:02d}.json") for i in range(32)]
            assert all(v["passed"] and v["selected_relative_block_mse"] <= v["initial_relative_block_mse"]
                       for v in entries)
            write(out / "OVERLAY.json", dict(passed=True, generic=True, method="k25", layers=entries,
                  sources_sha256=sha(CONFIG), partition_sha256=read(CONFIG)["partitions"],
                  fit_windows=16384, labels_used=False, original_dense_start=True))
        dist.barrier()


if __name__ == "__main__":
    main()
