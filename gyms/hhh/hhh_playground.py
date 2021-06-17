from gyms.hhh.cpp.hhhmodule import SketchHHH as HHHAlgo
from ipaddress import IPv4Address

hhh = HHHAlgo(0.1)
hhh.update(0xFFFFFFFF, 5)
hhh.update(0x00000000, 10)
hhh.update(0x00000001, 1)
hhh.update(0xFFFFFFFF, 20)
hhh.update(0x00000001, 3)
hhh.update(0x00000002, 2)
hhh.update(0x00000002, 2)
hhh.update(0xF0000000, 10)
hhh.update(0xA0000000, 15)
res = hhh.query(0.2, 0)

print('=== BLOCKLIST ===')
for r in res:
    padded_ip = str(IPv4Address(r.id)).rjust(15, ' ')
    print(f'{padded_ip}/{r.len} {r.lo, r.hi}')
