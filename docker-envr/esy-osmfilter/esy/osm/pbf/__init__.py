'''
`esy.osm.pbf` is a low-level Python library to interact with
[OpenStreetMap](https://www.openstreetmap.org) data files in the [Protocol
Buffers (PBF)](https://developers.google.com/protocol-buffers/) format.

# Usage

To count the amount of parks in the OpenStreetMap Andorra `.pbf` file (at least
according to a copy from [geofabrik](https://www.geofabrik.de/)), do this:

First download a copy of the andorra dataset:

```python
>>> import os, urllib.request
>>> if not os.path.exists('andorra.osm.pbf'):
...     filename, headers = urllib.request.urlretrieve(
...         'https://download.geofabrik.de/europe/andorra-190101.osm.pbf',
...         filename='andorra.osm.pbf'
...     )

```

Open the file and iterate over all entry and count those with a tag `leisure`
having a value of `park`.

```python
>>> import esy.osm.pbf
>>> osm = esy.osm.pbf.File('andorra.osm.pbf')
>>> len([entry for entry in osm if entry.tags.get('leisure') == 'park'])
21

```

# Design, Development & Contributing

Design and development notes are available in `esy.osm.pbf.test`.

We would be happy to accept contributions via merge requests, but due to
corporate policy we can only accept contributions if you have send us the signed
[contributor license agreement](CLA.md).

# Contact

Please use the projects issue tracker to get in touch.

# Team

`esy.osm.pbf` is developed by the
[DLR](https://www.dlr.de/EN/Home/home_node.html) Institute of
[Networked Energy Systems](https://www.dlr.de/ve/en/desktopdefault.aspx/tabid-12472/21440_read-49440/)
in the departement for
[Energy Systems Analysis (ESY)](https://www.dlr.de/ve/en/desktopdefault.aspx/tabid-12471/21741_read-49802/).

# Acknowledgements

The authors would like to thank the Federal Government and the Heads of
Government of the Länder, as well as the Joint Science Conference (GWK), for
their funding and support within the framework of the NFDI4Ing consortium.
Funded by the German Research Foundation (DFG) - project number 442146713.
'''

import importlib.metadata

from esy.osm.pbf.file import (
    File, Node, Way, Relation, iter_blocks, read_blob, iter_primitive_block,
)
from esy.osm.pbf.index import Index


__version__ = importlib.metadata.version('esy-osm-pbf')
__all__ = ['file', 'index']
