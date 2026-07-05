import zlib

def hash_episode(ep=None, eid=None):
    if (eid is None):
        eid = ep.id_
    return zlib.crc32(eid.encode('utf-8'))