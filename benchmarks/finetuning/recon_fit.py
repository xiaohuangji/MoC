"""Four-rank capture and validation, original shape-preserving updates on rank zero."""
import gc
import hashlib
from pathlib import Path
import time
import torch
import torch.distributed as dist
from io_utils import read, write, sha, tensor_sha

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
STEPS, INTERVAL, TOKEN_BATCH, RECON_LR = 8192, 256, 1024, 3e-5


class ReconstructionMLP(torch.nn.Module):
    """Raw-gate K25 used only to fit the unfrozen FFN weights on C4."""

    def __init__(self, original):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = original.gate_proj, original.up_proj, original.down_proj
        self.act_fn = original.act_fn
        self.k = self.up_proj.base_layer.out_features // 4

    def forward(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        with torch.no_grad():
            ids = gate.topk(self.k, dim=-1, largest=True, sorted=False).indices
        product = self.act_fn(gate.gather(-1, ids))*up.gather(-1, ids)
        return self.down_proj(torch.zeros_like(gate).scatter(-1, ids, product))


def sample_groups(counts,generator,batch):
    assert len(counts)==4 and min(counts)>0
    chosen=torch.randint(sum(counts),(batch,),generator=generator)
    return [chosen[(chosen>=sum(counts[:r]))&(chosen<sum(counts[:r+1]))]-sum(counts[:r]) for r in range(4)]


def required_indices(counts,layer,rank,steps=STEPS,batch=1024):
    generator=torch.Generator().manual_seed(20261022+layer)
    selected=[torch.zeros(1,dtype=torch.long)]
    for _ in range(steps): selected.append(sample_groups(counts,generator,batch)[rank])
    return torch.cat(selected).unique(sorted=True)


def positions(required,ids):
    mapped=torch.searchsorted(required,ids)
    assert bool((mapped<len(required)).all())
    assert torch.equal(required[mapped],ids),'Requested uncached original token'
    return mapped


def gather(values, ids, device):
    assert len(values) == 3 and ids.ndim == 1
    assert all(v.device == values[0].device for v in values)
    index = ids.to(values[0].device)
    return tuple(v[index].to(device) for v in values)


def finite_guard(parameters):
    assert all(p.grad is not None for p in parameters)
    assert bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all())
    assert torch.cuda.max_memory_allocated() / 2**30 <= 39


@torch.no_grad()
def evaluate(module, data, device):
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    for start in range(0, len(data[0]), 256):
        x, residual, target = (v[start:start+256].to(device) for v in data)
        with torch.autocast('cuda', dtype=torch.bfloat16): prediction = residual + module(x)
        assert torch.isfinite(prediction).all()
        totals[0] += (prediction.float()-target.float()).square().double().sum()
        totals[1] += target.float().square().double().sum()
    dist.all_reduce(totals)
    assert totals[1] > 0 and torch.isfinite(totals).all()
    return float(totals[0] / totals[1])


def snapshot(parameters):
    return {name: p.detach().cpu().clone() for name, p in parameters.items()}


def loss_from_parts(numerators,denominators):
    denominator=torch.stack(denominators).sum()
    assert torch.isfinite(denominator) and denominator>0
    loss=torch.stack([n.double()/denominator for n in numerators]).sum()
    assert torch.isfinite(loss)
    return loss


def fit(block,data,heldout,layer,device,rank,out):
    out=Path(out)
    assert dist.get_world_size()==4
    assert STEPS==8192 and INTERVAL==256 and TOKEN_BATCH==1024
    block.mlp=ReconstructionMLP(block.mlp)
    module=block.mlp.eval()
    parameters={n:getattr(module,n).base_layer.weight for n in PROJECTIONS}
    for p in parameters.values():p.data=p.data.float();p.requires_grad_(rank==0)
    from contextlib import ExitStack
    import recon_storage as fast_storage
    values,required,original_count=data
    cache_tensors={name:tensor_sha(t) for name,t in zip(('input','residual','target'),values)}
    recorded=read(out/f'capture_{layer:02d}_rank{rank}.json')['cache_sha256']
    assert all(cache_tensors[n]==recorded['fit_'+n] for n in cache_tensors)
    local=dict(storage='shared_ram_exact_sampling_union',buffers=fast_storage.describe(values),
        indices=fast_storage.describe((required,)),tensor_sha256=cache_tensors,count=original_count,
        required_sha256=tensor_sha(required),stored_count=len(required))
    identities=[None]*4;dist.all_gather_object(identities,local)
    counts=[x['count'] for x in identities]
    assert min(counts)>0
    all_data=[]
    shared_contexts=ExitStack()
    cache_bytes=sum(x['stored_count'] for x in identities)*sum(v.shape[1]*v.element_size() for v in values)
    all_required=[]
    if rank==0:
        for info in identities:
            values=shared_contexts.enter_context(fast_storage.open_shared(info['buffers']))
            assert {n:tensor_sha(v) for n,v in zip(('input','residual','target'),values)}==info['tensor_sha256']
            all_data.append(values)
            indices=shared_contexts.enter_context(fast_storage.open_shared(info['indices']))[0]
            assert tensor_sha(indices)==info['required_sha256']
            all_required.append(indices)
            del values
            del indices
        write(out/f'cache_storage_layer{layer:02d}.json',dict(passed=True,bytes=cache_bytes,
            storage='cpu_shared_ram',device_cache_limit_bytes=8*1024**3,
            values_order_shapes_unchanged=True,counts=counts,duplicate_disk_cache=False))
    initial_loss=evaluate(module,heldout,device)
    best_loss=initial_loss;best_step=0;best_weights=snapshot(parameters)
    history=[dict(step=0,relative_block_mse=initial_loss)]
    seconds=0.0;started=time.monotonic()
    generator=torch.Generator().manual_seed(20261022+layer)
    order=[hashlib.sha256() for _ in range(4)];consumed=[0]*4
    opt=torch.optim.AdamW(list(parameters.values()),lr=RECON_LR,weight_decay=0,eps=1e-8,foreach=False) if rank==0 else None
    for end in range(INTERVAL,STEPS+1,INTERVAL):
        if rank==0:
            for step in range(end-INTERVAL+1,end+1):
                torch.cuda.synchronize();tick=time.monotonic()
                groups=[];empty=[]
                for r,ids in enumerate(sample_groups(counts,generator,TOKEN_BATCH)):
                    order[r].update(ids.numpy().tobytes());consumed[r]+=len(ids)
                    empty.append(len(ids)==0)
                    if len(ids)==0:ids=torch.zeros(1,dtype=torch.long)
                    groups.append(gather(all_data[r],positions(all_required[r],ids),device))
                opt.zero_grad(set_to_none=True)
                warm=max(1,STEPS//10)
                factor=step/warm if step<=warm else (STEPS-step+1)/(STEPS-warm)
                for g in opt.param_groups:g['lr']=RECON_LR*factor
                numerators=[];denominators=[]
                for is_empty,(x,residual,target) in zip(empty,groups):
                    with torch.autocast('cuda',dtype=torch.bfloat16):prediction=residual+module(x)
                    numerator=(prediction.float()-target.float()).square().sum()
                    denominator=target.float().square().sum().detach().double()
                    if is_empty:numerator=numerator*0;denominator=denominator*0
                    numerators.append(numerator);denominators.append(denominator)
                loss=loss_from_parts(numerators,denominators)
                loss.backward();finite_guard(list(parameters.values()))
                norm=torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.0,error_if_nonfinite=True)
                opt.step();torch.cuda.synchronize();seconds+=time.monotonic()-tick
            write(out/f'progress_layer{layer:02d}.json',dict(layer=layer,step=end,pure_update_seconds=seconds,
                mean_update_seconds=seconds/end,allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                reserved_gib=torch.cuda.max_memory_reserved()/2**30,fit_gpu_rank=0,heldout_world_size=4,
                grad_norm=float(norm),lr=opt.param_groups[0]['lr']))
        with torch.no_grad():
            for p in parameters.values():dist.broadcast(p,src=0)
        measured=evaluate(module,heldout,device)
        history.append(dict(step=end,relative_block_mse=measured))
        if measured<best_loss:best_loss=measured;best_step=end;best_weights=snapshot(parameters)
    if rank==0:
        budget=dict(counts=counts,consumed=consumed,orders=[h.hexdigest() for h in order],pure_update_seconds=seconds)
    else:budget=None
    obj=[budget];dist.broadcast_object_list(obj,src=0);budget=obj[0]
    with torch.no_grad():
        for n,p in parameters.items():p.copy_(best_weights[n])
        x=heldout[0][:min(256,len(heldout[0]))].to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):before=module(x).clone()
    del opt,best_weights
    for p in parameters.values():p.requires_grad_(False);p.grad=None;p.data=p.data.bfloat16()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):after=module(x)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    exported=evaluate(module,heldout,device)
    assert abs(exported-best_loss)<=1e-10
    weights=snapshot(parameters);hashes={n:tensor_sha(t) for n,t in weights.items()}
    all_hashes=[None]*4;dist.all_gather_object(all_hashes,hashes);assert all(h==hashes for h in all_hashes)
    filename=f'layer_{layer:02d}.pt'
    if rank==0:
        torch.save(weights,out/(filename+'.tmp'));(out/(filename+'.tmp')).replace(out/filename)
        saved=torch.load(out/filename,map_location='cpu',weights_only=True)
        assert {n:tensor_sha(t) for n,t in saved.items()}==hashes
        write(out/f'layer_{layer:02d}.json',dict(passed=True,layer=layer,k=module.k,filename=filename,
            file_sha256=sha(out/filename),tensor_sha256=hashes,selected_step=best_step,
            initial_relative_block_mse=initial_loss,selected_relative_block_mse=best_loss,
            export_relative_block_mse=exported,history=history,steps=STEPS,global_token_batch=TOKEN_BATCH,
            total_sampled_tokens=STEPS*TOKEN_BATCH,pure_update_seconds=seconds,
            fit_stage_seconds=time.monotonic()-started,export_probe_bitwise_equal=True,all_rank_export_identity=True,
            execution='four-rank capture and heldout, rank0 shape-preserving shard accumulation',cache_identities=identities,
            transient_cache_deleted_after_layer=True))
    write(out/f'layer_{layer:02d}_rank{rank}.json',dict(passed=True,steps=STEPS,
        local_tokens=budget['consumed'][rank],local_order_sha256=budget['orders'][rank],cache_token_counts=counts,
        pure_update_seconds=budget['pure_update_seconds'] if rank==0 else 0.0,
        selected_step=best_step,hashes=hashes,gradient_updates_on_rank0=True))
    del all_data,all_required,weights,before,after,x,parameters
    shared_contexts.close()
    gc.collect();torch.cuda.empty_cache();dist.barrier()
