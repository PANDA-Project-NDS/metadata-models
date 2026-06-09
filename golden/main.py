import asyncio
import itertools
import json
import logging
import os
import argparse
import contextlib
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from agents.base import langfuse, _langfuse_available
from llama_index.core import Settings
from db import get_embed_model, make_sentence_splitter
from llama_index.core import VectorStoreIndex

from agents import PASSES
from golden.models import JournalIdentity
from golden.pipeline import (
    Chunk,
    chunk_files,
    run_extraction_pipeline,
)
from golden.agents.judge import judge_journal, load_coverage_sections
from golden.lib.flatten import strip_evidence
from models.journal import JournalMetadata


# MUST set WITH_EVIDENCE before any model imports
os.environ.setdefault("WITH_EVIDENCE", "1")


# --- Constants ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_ROOT = PROJECT_ROOT / "journal-samples"
GOLDEN_OUT = SAMPLES_ROOT / "golden"


# --- Field-to-Pass Mapping ---

PASS_FIELDS = {
    0: {"title", "publisher", "issn", "scope", "facts", "metrics"},
    1: {"publication_frequency", "submissions", "policies", "languages", "diamond_open_access"},
    2: {"pricing", "membership"},
    3: {"editors"},
}

# Verify all JournalMetadata top-level fields are assigned to exactly one pass
_all_pass_fields = set()
for fields in PASS_FIELDS.values():
    _all_pass_fields.update(fields)
for fname in JournalMetadata.model_fields:
    if fname not in ("journal_id", "uri"):  # structural fields, no pass
        assert fname in _all_pass_fields, f"Field '{fname}' not in any pass"


# --- Discovery and Loading ---

def discover_journals(publisher_filter=None, journal_filter=None) -> Generator[JournalIdentity, None, None]:
    """Walk journal-samples/{publisher}/extracted/{journal}/ for directories with .md files."""
    for pub_dir in sorted(SAMPLES_ROOT.iterdir()):
        if not pub_dir.is_dir() or pub_dir.name == "golden":
            continue
        if publisher_filter and pub_dir.name != publisher_filter:
            continue
        ext_dir = pub_dir / "extracted"
        if not ext_dir.is_dir():
            continue
        for jdir in sorted(ext_dir.iterdir()):
            if not jdir.is_dir():
                continue
            if journal_filter and jdir.name != journal_filter:
                continue
            if list(jdir.glob("*.md")):
                yield JournalIdentity(pub_dir.name, jdir.name)


def build_index_from_chunks(
    chunks: list[Chunk],
) -> VectorStoreIndex:
    """Build per-journal VectorStoreIndex from chunks.

    Uses make_node_parser() from db/indexer.py to split chunks into
    embedding-sized pieces (450 tokens). Passes chunk.header_path as
    metadata for structural context in search results.
    """
    from llama_index.core import Document

    splitter = make_sentence_splitter()

    documents = [
        Document(text=c.text, metadata={"source_uri": c.filename, "header_path": c.header_path})
        for c in chunks
    ]

    nodes = []
    for doc in documents:
        nodes.extend(splitter.get_nodes_from_documents([doc]))

    return VectorStoreIndex(nodes=nodes)


# --- Output Writers ---

def write_outputs(publisher, journal, metadata, grading):
    """Write golden JSON and grading sidecar."""
    out_dir = GOLDEN_OUT / publisher
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = strip_evidence(metadata.model_dump(mode="json", exclude_none=True))
    (out_dir / f"{journal}.json").write_text(
        json.dumps(clean, indent=2, default=str), encoding="utf-8"
    )

    (out_dir / f"{journal}.grading.json").write_text(
        json.dumps(grading, indent=2, default=str), encoding="utf-8"
    )


# --- Main ---

async def main():
    parser = argparse.ArgumentParser(description="Generate golden JournalMetadata JSON")
    parser.add_argument("--publisher", default=None)
    parser.add_argument("--journal", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of journals processed (for incremental runs).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--map-parallel", type=int, default=4,
                        help="Max concurrent map-phase LLM calls (default: 4).")
    args = parser.parse_args()

    load_dotenv()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    Settings.embed_model = get_embed_model()
    Settings.llm = None

    # Load coverage.md sections once, cached by publisher name
    coverage_sections = load_coverage_sections()
    logger.info(f"Loaded coverage for {len(coverage_sections)} publishers")

    journals = discover_journals(args.publisher, args.journal)
    if args.limit is not None:
        journals = itertools.islice(journals, args.limit)

    stats = {"ok": 0, "fail": 0, "skip": 0}
    sem = asyncio.Semaphore(args.map_parallel)

    for identity in journals:
        publisher, journal = identity.publisher, identity.journal
        out_path = GOLDEN_OUT / publisher / f"{journal}.json"
        if out_path.exists() and not args.force:
            stats["skip"] += 1
            continue

        if _langfuse_available:
            from langfuse import propagate_attributes

            trace_ctx = langfuse.start_as_current_observation(
                as_type="trace",
                name=f"Journal Extraction: {publisher}/{journal}",
            )
            attr_ctx = propagate_attributes(
                metadata={"publisher": publisher, "journal": journal},
                tags=["golden-metadata", publisher],
            )
        else:
            trace_ctx = contextlib.nullcontext()
            attr_ctx = contextlib.nullcontext()

        with trace_ctx, attr_ctx:
            journal_dir = SAMPLES_ROOT / publisher / "extracted" / journal

            # Chunk files and build index once — shared across all passes
            chunks_gen = chunk_files(journal_dir)
            # Convert to list here because chunks are used multiple times across passes
            chunks = list(chunks_gen)
            if not chunks:
                logger.warning(f"No content for {publisher}/{journal}")
                stats["fail"] += 1
                continue
            journal_id = f"{publisher}/{journal}"
            logger.info(f"Processing {journal_id} ({len(chunks)} chunks) ...")

            # Build index in thread to avoid blocking the event loop during embedding
            index = await asyncio.to_thread(build_index_from_chunks, chunks)

            # Phase 1: Map-Reduce Extraction (semaphore-gated, concurrent within journal)
            async def _run_pass(pc):
                async with sem:
                    return await run_extraction_pipeline(pc, journal_id, journal_dir, index, chunks)

            pass_results = await asyncio.gather(*[_run_pass(pc) for pc in PASSES])

            # Merge pass results into JournalMetadata
            merged = {}
            for result, _ in pass_results:
                merged.update(result.model_dump(mode="json"))
            metadata = JournalMetadata.model_validate(merged)
            metadata.journal_id = journal_id

            # Phase 2: Two-Judge Evaluation (coverage + evidence)
            coverage_text = coverage_sections.get(publisher, "")
            grading = await judge_journal(metadata, publisher, coverage_text)

            # Write
            if not args.dry_run:
                write_outputs(publisher, journal, metadata, grading)
            stats["ok"] += 1
            logger.info(f"Done {journal_id}. OK: {stats['ok']}, Failed: {stats['fail']}, Skipped: {stats['skip']}")

    if _langfuse_available:
        langfuse.flush()

    logger.info(f"Finished. OK: {stats['ok']}, Failed: {stats['fail']}, Skipped: {stats['skip']}")


if __name__ == "__main__":
    asyncio.run(main())