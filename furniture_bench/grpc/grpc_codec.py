"""简单的 gRPC bytes 序列化/反序列化工具。"""


from __future__ import annotations

import zlib
import cloudpickle


def dumps(obj, compress: bool = True) -> bytes:
    data = cloudpickle.dumps(obj)
    return zlib.compress(data) if compress else data


def loads(data: bytes, compress: bool = True):
    raw = zlib.decompress(data) if compress else data
    return cloudpickle.loads(raw)
