import struct, sys

path = r'C:\Projects\llm-npu\models\qwen2.5-0.5b-instruct-q4_k_m.gguf'
f = open(path, 'rb')

GGUF_TYPES = {0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),10:('Q',8),11:('q',8),6:('f',4),12:('d',8),7:('?',1)}

def rd_u32(): return struct.unpack('<I', f.read(4))[0]
def rd_u64(): return struct.unpack('<Q', f.read(8))[0]
def rd_str():
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')

def skip_value(t):
    if t == 8:
        rd_str()
    elif t == 9:
        et = rd_u32()
        n = rd_u64()
        if et == 8:
            for _ in range(n):
                skip_value(8)
        else:
            f.read(GGUF_TYPES[et][1] * n)
    elif t in GGUF_TYPES:
        f.read(GGUF_TYPES[t][1])
    else:
        raise Exception(f'unknown kv type {t}')

f.read(4)  # magic
ver = rd_u32()
tc = rd_u64()
kc = rd_u64()

# skip kv pairs, but print general.name
for _ in range(kc):
    k = rd_str()
    t = rd_u32()
    off = f.tell()
    skip_value(t)

print(f'GGUF v{ver} tensors={tc}')
for i in range(tc):
    name = rd_str()
    nd = rd_u32()
    ne = struct.unpack(f'<{nd}Q', f.read(8*nd))
    ttype = rd_u32()
    off = rd_u64()
    if 'output' in name or 'lm_head' in name or 'embed' in name:
        print(f'{name}: type={ttype} ne={ne}')
print('GGML type names: 0=F32 1=F16 2=Q4_0 3=Q4_1 6=Q5_0 7=Q5_1 8=Q8_0 12=Q2_K 13=Q3_K 14=Q4_K 15=Q5_K 16=Q6_K')
