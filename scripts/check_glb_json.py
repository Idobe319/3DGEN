import struct,sys
p='workspace/clean_quads_uv.glb'
with open(p,'rb') as f:
    b=f.read()
if len(b)<20:
    print('file too small'); sys.exit(1)
chunk_len=struct.unpack_from('<I',b,12)[0]
chunk_type=b[16:20].decode('ascii')
json_bytes=b[20:20+chunk_len]
s=json_bytes.decode('utf-8',errors='replace')
print('chunk_type',chunk_type)
print('NaN in JSON?', 'NaN' in s or 'nan' in s)
print('JSON size', len(s))
# Optionally, write JSON to a temp file for inspection
with open('workspace/clean_quads_uv.json','w',encoding='utf-8') as out:
    out.write(s)
print('Wrote workspace/clean_quads_uv.json for inspection')