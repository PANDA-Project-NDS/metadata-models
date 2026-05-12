import os
from contextlib import contextmanager

from pymongo import MongoClient

from .embed import get_embed_model
from .indexer import Indexer
from .metadata import MetadataStore
from .documents import DocumentStore


@contextmanager
def mongo_connection():
    uri = os.environ["MONGODB_URI"]
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        yield client
    finally:
        client.close()


__all__ = [
    "mongo_connection",
    "DocumentStore",
    "MetadataStore",
    "Indexer",
    "get_embed_model",
]
