from contextbuddy.mcp.kb import matches_to_context, search_codebase


def test_search_codebase_finds_match_in_repo() -> None:
    matches = search_codebase("ContextEngine", root=".", max_matches=5, max_files=2000)
    assert len(matches) > 0
    assert any("contextengine" in m.preview.lower() for m in matches)


def test_matches_to_context_format() -> None:
    matches = search_codebase("ContextEngineConfig", root=".", max_matches=2, max_files=2000)
    chunks = matches_to_context(matches)
    assert len(chunks) == len(matches)
    if chunks:
        assert chunks[0].startswith("Source: ")

