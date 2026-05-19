from llama_index.core.schema import NodeWithScore, TextNode


def make_node(
    text: str,
    node_id: str = "test-node",
    source_uri: str = "test.html",
    score: float = 0.9,
) -> NodeWithScore:
    """Create a NodeWithScore for testing."""
    node = TextNode(
        text=text,
        id_=node_id,
        metadata={"source_uri": source_uri, "journal_id": "test-journal"},
    )
    return NodeWithScore(node=node, score=score)
