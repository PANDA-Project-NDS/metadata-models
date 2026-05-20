#!/usr/bin/env python3
"""Extract leaf node labels from the COARS Resource Types XML.

Reads example_schema/resource_types_for_dspace_en.xml and outputs the leaf
labels (nodes without children) under a given parent node by COARS id.

Usage:
    python scripts/extract_coars.py                  # defaults to text (c_18cf)
    python scripts/extract_coars.py --id c_0640      # journal subtree
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parent.parent
XML_PATH = REPO_ROOT / "example_schema" / "resource_types_for_dspace_en.xml"


def find_node(element: ET.Element, target_id: str) -> ET.Element | None:
    if element.tag == "node" and element.attrib.get("id") == target_id:
        return element
    for child in element:
        result = find_node(child, target_id)
        if result is not None:
            return result
    return None


def get_leaf_nodes(node: ET.Element) -> list[tuple[str, str]]:
    """Return (id, label) for every leaf node under *node*."""
    leaves: list[tuple[str, str]] = []
    composed = node.find("isComposedBy")
    if composed is None:
        leaves.append((node.attrib["id"], node.attrib["label"]))
    else:
        for child in composed:
            leaves.extend(get_leaf_nodes(child))
    return leaves


def main(target_id: str = "c_18cf") -> None:
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    target = find_node(root, target_id)
    if target is None:
        print(f"ERROR: node with id '{target_id}' not found", file=sys.stderr)
        sys.exit(1)

    leaves = get_leaf_nodes(target)
    leaves.sort(key=lambda x: x[1])

    print(f"# {target.attrib['label']} — {len(leaves)} leaf nodes")
    print()

    # Literal-friendly listing
    for _id, label in leaves:
        print(f'    "{label}",')

    print()
    print(f'# COARS id mapping for "{target.attrib["label"]}"')
    for _id, label in leaves:
        print(f'    "{label}": "{_id}",')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract COARS leaf nodes")
    parser.add_argument(
        "--id",
        default="c_18cf",
        help="COARS node id to extract leaves from (default: c_18cf = text)",
    )
    args = parser.parse_args()
    main(args.id)
