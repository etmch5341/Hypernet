from itertools import accumulate
from bisect import bisect

from esy.osm.pbf.file import (
    File, read_blob, parse_header_block, decode_strmap, iter_blocks, iter_nodes,
    iter_ways, iter_relations, Node, Way, Relation
)
from esy.osm.pbf.osmformat_pb2 import PrimitiveBlock


def index(file):
    osmheader, indexmap = None, ([], [], [])
    block = PrimitiveBlock()
    for offset, header in iter_blocks(file):
        if header.type == 'OSMHeader':
            osmheader = parse_header_block(file, offset, header)
        elif header.type != 'OSMData':
            raise ValueError(f'Unsupported OSM header type "{header.type}"')

        block.ParseFromString(read_blob(file, offset, header.datasize))

        for group in block.primitivegroup:
            # TODO Check that ids are sorted. Or also fetch minid and do
            # two overlapping queries?
            # TODO If ids are sorted, last element is max_id
            if group.nodes:
                raise ValueError('Plain nodes not supported')
            elif group.dense.id:
                index = indexmap[0]
                max_id = sum(group.dense.id)
            elif group.ways:
                index = indexmap[1]
                max_id = max(w.id for w in group.ways)
            elif group.relations:
                index = indexmap[2]
                max_id = max(r.id for r in group.relations)
            else:
                continue

            index.append((max_id, offset, header.datasize))
    return osmheader, indexmap


def index_block(file, index, query):
    block = PrimitiveBlock()
    start = 0
    for max_id, offset, size in index:
        stop = bisect(query, max_id, lo=start)
        if start == stop:
            continue

        block.ParseFromString(read_blob(file, offset, size))
        yield block, query[start:stop]
        start = stop


iter_funcs = (iter_nodes, iter_ways, iter_relations)


class Index(object):
    def __init__(self, filename):
        self.pbf = File(filename)
        self.header, self.index = index(self.pbf.file)
        self.bounds = (
            self.header.bbox.left / 1e9, self.header.bbox.bottom / 1e9,
            self.header.bbox.right / 1e9, self.header.bbox.top / 1e9
        )

    def __getstate__(self):
        return (self.pbf.file.name, self.header, self.index, self.bounds)

    def __setstate__(self, state):
        filename, self.header, self.index, self.bounds = state
        self.pbf = File(filename)

    @property
    def nblocks(self):
        return sum(len(i) for i in self.index)

    def __call__(self, queries):
        results = ({}, {}, {})
        for iter_func, index, query, result in zip(
            iter_funcs, self.index, queries, results
        ):
            for block, ids in index_block(self.pbf.file, index, sorted(query)):
                ids = set(ids)
                strmap = decode_strmap(block)
                for group in block.primitivegroup:
                    for entry in iter_func(block, strmap, group):
                        if entry.id in ids:
                            result[entry.id] = entry
        return results

    def __iter__(self):
        return self.pbf.__iter__()
