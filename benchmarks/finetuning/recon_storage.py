import contextlib
import ctypes
import mmap
import os
import torch


def memfd(name):
    if hasattr(os,'memfd_create'):
        fd=os.memfd_create(name,0)
    else:
        libc=ctypes.CDLL(None,use_errno=True)
        call=libc.memfd_create
        call.argtypes=[ctypes.c_char_p,ctypes.c_uint];call.restype=ctypes.c_int
        fd=call(name.encode(),0)
        if fd < 0:
            error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    os.set_inheritable(fd,False)
    return fd


class SharedBuffer:
    def __init__(self,name,shape,dtype):
        self.shape=tuple(shape);self.dtype=dtype
        count=1
        for n in shape: count*=n
        self.bytes=count*torch.empty((),dtype=dtype).element_size()
        self.fd=memfd(name)
        os.ftruncate(self.fd,self.bytes)
        self.mapping=mmap.mmap(self.fd,self.bytes,access=mmap.ACCESS_WRITE)
        self.tensor=torch.frombuffer(self.mapping,dtype=dtype,count=count).reshape(shape)
        self.path=f'/proc/{os.getpid()}/fd/{self.fd}'

    def descriptor(self):
        return dict(path=self.path,shape=list(self.shape),dtype=str(self.dtype),bytes=self.bytes)

    def close(self):
        del self.tensor
        self.mapping.close();os.close(self.fd)


BUFFERS={}


def mapped(directory,name,shape,dtype):
    count=1
    for n in shape:count*=n
    assert sum(h.bytes for h in BUFFERS.values())+count*torch.empty((),dtype=dtype).element_size() <= 48*1024**3
    handle=SharedBuffer(name,shape,dtype)
    BUFFERS[handle.tensor.data_ptr()]=handle
    return handle,handle.tensor


def describe(values):
    handles=[BUFFERS[v.data_ptr()] for v in values]
    assert all(h.tensor.shape==v.shape and h.tensor.dtype==v.dtype for h,v in zip(handles,values))
    return [h.descriptor() for h in handles]


@contextlib.contextmanager
def open_shared(descriptors):
    mappings=[];files=[];values=[]
    try:
        for d in descriptors:
            assert d['path'].startswith('/proc/') and '/fd/' in d['path']
            dtype={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32,'torch.int64':torch.int64}[d['dtype']]
            f=open(d['path'],'rb');files.append(f)
            assert os.fstat(f.fileno()).st_size==d['bytes']
            mapping=mmap.mmap(f.fileno(),d['bytes'],access=mmap.ACCESS_COPY);mappings.append(mapping)
            value=torch.frombuffer(mapping,dtype=dtype).reshape(d['shape']);values.append(value)
        yield tuple(values)
    finally:
        values.clear()
        if 'value' in locals(): del value
        for mapping in mappings: mapping.close()
        for f in files:f.close()


class RAMRecords:
    def __init__(self):
        self.entries = []
        self.tokens = 0
        self.raw_bytes = 0

    def append(self, record):
        self.entries.append(record)
        self.tokens += int(record['mask'].sum())
        self.raw_bytes += sum(record[k].numel()*record[k].element_size() for k in ('teacher', 'student'))

    def __iter__(self):
        yield from self.entries

    def close(self):
        self.entries.clear()
