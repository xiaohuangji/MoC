"""Replay real decoder arguments, with separate Dense and rolling-student boundaries."""
import torch
import recon_storage as storage


class CaptureStop(Exception):
    pass


def move(value, device):
    if isinstance(value, torch.Tensor): return value.detach().to(device)
    if isinstance(value, tuple): return tuple(move(v, device) for v in value)
    if isinstance(value, list): return [move(v, device) for v in value]
    if isinstance(value, dict): return {k: move(v, device) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)): return value
    raise TypeError(type(value))


@torch.no_grad()
def initial(model, rows, device, record_store=None):
    from glue_data import collate
    records = [] if record_store is None else record_store
    pending = {}
    def hook(_module, args, kwargs):
        assert not kwargs.get('use_cache') and kwargs.get('past_key_value') is None
        hidden = args[0].detach().cpu()
        pending['record'] = dict(teacher=hidden, student=hidden.clone(), tail=move(args[1:], 'cpu'),
                                kwargs=move(kwargs, 'cpu'))
        raise CaptureStop
    handle = model.base_model.model.model.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for start in range(0, len(rows), 8):
            group = rows[start:start+8]
            batch = collate(group)
            batch.pop('labels'); batch.pop('sample_id')
            try:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    model(**{k: v.to(device) for k, v in batch.items()})
            except CaptureStop:
                record = pending.pop('record')
                record['mask'] = batch['attention_mask'].bool()
                record['sample_ids'] = [r['sample_id'] for r in group]
                records.append(record)
                del record
            else: raise RuntimeError('Initial boundary hook not visited')
    finally:
        handle.remove()
    return records


@torch.no_grad()
def capture(block,records,device,required=None,previous_block=None):
    total=records.tokens if hasattr(records,'tokens') else sum(int(record['mask'].sum()) for record in records)
    count=total if required is None else len(required)
    arrays=None;owners=[];offset=0;stored=0
    for record in records:
        tail,kwargs=move(record['tail'],device),move(record['kwargs'],device)
        mask=record['mask'].to(device)
        student=record['student'].to(device)
        if previous_block is not None:
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                student=student+previous_block.mlp(previous_block.post_attention_layernorm(student))
            assert torch.isfinite(student).all()
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            out=block(record['teacher'].to(device),*tail,**kwargs)
            teacher=out[0] if isinstance(out,tuple) else out
        assert torch.isfinite(teacher).all()
        record['teacher']=teacher.cpu()
        got={}
        def residual_hook(_module,args):got['residual']=args[0].detach()
        def input_hook(_module,args):got['x']=args[0].detach();raise CaptureStop
        handles=[block.post_attention_layernorm.register_forward_pre_hook(residual_hook),
                 block.mlp.register_forward_pre_hook(input_hook)]
        try:
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                try:block(student,*tail,**kwargs)
                except CaptureStop:pass
                else:raise RuntimeError('FFN input hook not reached')
            assert set(got)=={'x','residual'}
            n=int(mask.sum())
            if required is None:local=None;amount=n
            else:
                left=int(torch.searchsorted(required,offset));right=int(torch.searchsorted(required,offset+n))
                local=(required[left:right]-offset).to(device);amount=right-left
            tensors=(got['x'][mask],got['residual'][mask],teacher[mask])
            values=tuple((value if local is None else value[local]).cpu() for value in tensors)
            if arrays is None:
                allocated=[storage.mapped(None,f'capture_{i}',(count,value.shape[1]),value.dtype) for i,value in enumerate(values)]
                owners=[h for h,_ in allocated];arrays=tuple(t for _,t in allocated)
            for dest,value in zip(arrays,values):dest[stored:stored+amount].copy_(value)
            record['student']=got['residual'].cpu()
            offset+=n;stored+=amount
        finally:
            for handle in handles:handle.remove()
    assert offset==total and stored==count
    return arrays,owners


def release(owners):
    for owner in owners:
        del storage.BUFFERS[owner.tensor.data_ptr()]
        owner.close()
