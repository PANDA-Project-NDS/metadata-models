#!/usr/bin/env python3
"""Compare APC fees stored in MongoDB against an external JSON dataset."""

import argparse
import csv
import json
import logging
import sys

from dotenv import load_dotenv

from db import MetadataStore, mongo_connection

logger = logging.getLogger(__name__)


def load_file_journals(filepath: str, publisher: str) -> dict[str, float]:
    """Load journals from JSON file, filtered by publisher. Returns {name: fee}."""
    with open(filepath) as f:
        data = json.load(f)

    entries = [j for j in data["journals"] if j["publisher"] == publisher]
    logger.info(f"Loaded {len(entries)} entries for publisher '{publisher}' from file")

    result: dict[str, float] = {
        entry["name"]: entry["fee"] for entry in entries if entry["currency"] == "USD"
    }

    logger.info(f"{len(result)} unique journals")
    return result


def load_db_journals(meta: MetadataStore, publisher_id: str) -> dict[str, dict]:
    """Load journal metadata from MongoDB, filtered by publisher_id."""
    coll = meta.get_collection(meta.metadata_collection)
    docs = list(coll.find({"publisher_id": publisher_id}))
    logger.info(
        f"Loaded {len(docs)} journals for publisher_id '{publisher_id}' from DB"
    )

    return {doc["journal_id"]: doc for doc in docs if "journal_id" in doc}


def find_closest_usd_apc(db_doc: dict, target_fee: float) -> dict | None:
    """Find the USD APC closest to target_fee. If target_fee is 0, return the highest USD APC."""
    apcs = (db_doc.get("pricing") or {}).get("article_processing_charges", [])
    if not apcs:
        return None

    usd_apcs = []
    for apc in apcs:
        fee = apc.get("fee", {})
        if fee.get("currency") != "USD":
            continue
        fee_val = fee.get("value")
        if fee_val is None:
            continue
        usd_apcs.append((fee_val, apc))

    if not usd_apcs:
        return None

    if target_fee == 0:
        return max(usd_apcs, key=lambda x: x[0])[1]

    best = None
    best_diff = float("inf")
    for fee_val, apc in usd_apcs:
        diff = abs(fee_val - target_fee)
        if diff < best_diff:
            best_diff = diff
            best = apc

    return best


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Compare APC fees between a JSON file and MongoDB."
    )
    parser.add_argument("--file", required=True, help="Path to external JSON file")
    parser.add_argument(
        "--file-publisher", required=True, help="Publisher filter for file entries"
    )
    parser.add_argument(
        "--db-publisher", required=True, help="publisher_id filter for DB documents"
    )
    args = parser.parse_args()

    file_journals = load_file_journals(args.file, args.file_publisher)

    with mongo_connection() as client:
        meta = MetadataStore(client)
        db_journals = load_db_journals(meta, args.db_publisher)

        writer = csv.writer(sys.stdout)
        writer.writerow(["journal", "panter", "db"])

        for name, f_fee in sorted(file_journals.items()):
            db_fee = "0.0"
            db_doc = db_journals.get(name)
            if db_doc:
                closest = find_closest_usd_apc(db_doc, f_fee)
                if closest:
                    db_fee = float(closest["fee"]["value"])
                writer.writerow([name, f_fee, db_fee])


if __name__ == "__main__":
    main()
