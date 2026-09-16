#!/usr/bin/env python
"""Hash the CONTENT of a COCO file, so two builds can be compared.

A checksum over the raw file answers the wrong question: any provenance field
that varies between runs -- a timestamp, a path, a host name -- changes the
digest while the annotations are identical. This hashes only what a model would
ever see, under one canonical form, so two machines can settle whether their
rebuilds agree.

Both sides must run THIS script. Two reasonable canonicalisations disagree with
each other, so a mismatch between independently written recipes means nothing.

    python scripts/coco_content_hash.py data/chactun/coco/fold9_val.json
"""
import argparse
import hashlib
import json

FIELDS = {
    "images": ("id", "file_name", "width", "height"),
    "annotations": ("id", "image_id", "category_id", "bbox", "area",
                    "iscrowd", "edge_touching"),
    "categories": ("id", "name"),
}


def canonical(doc):
    """Records reduced to declared fields, sorted by id, floats rounded.

    Rounding matters: the same polygon area can differ in the last bit between
    numpy versions, which would break the comparison for no reason worth caring
    about.
    """
    out = {}
    for section, keys in FIELDS.items():
        rows = []
        for r in doc.get(section, []):
            row = {}
            for k in keys:
                if k not in r:
                    continue
                v = r[k]
                if isinstance(v, float):
                    v = round(v, 4)
                elif isinstance(v, list):
                    v = [round(x, 4) if isinstance(x, float) else x for x in v]
                row[k] = v
            rows.append(row)
        out[section] = sorted(rows, key=lambda x: x.get("id", 0))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    a = p.parse_args()
    for path in a.paths:
        doc = json.load(open(path))
        core = canonical(doc)
        blob = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        print("%s  %-42s images %5d  anns %6d"
              % (hashlib.md5(blob).hexdigest(), path.split("/")[-1],
                 len(core["images"]), len(core["annotations"])))


if __name__ == "__main__":
    main()
