from contextbuddy.entities import EntityExtractor


def test_extracts_emails() -> None:
    text = "Contact alice@example.com or bob@test.org for details."
    entities = EntityExtractor().extract(text)
    assert "alice@example.com" in entities
    assert "bob@test.org" in entities


def test_extracts_urls() -> None:
    text = "Visit https://example.com/path?q=1 for more."
    entities = EntityExtractor().extract(text)
    assert any("https://example.com" in e for e in entities)


def test_extracts_iso_dates() -> None:
    text = "Deadline is 2026-04-13 and meeting on 2026-05-01."
    entities = EntityExtractor().extract(text)
    assert "2026-04-13" in entities
    assert "2026-05-01" in entities


def test_extracts_uuids() -> None:
    text = "Record id: 550e8400-e29b-41d4-a716-446655440000 found."
    entities = EntityExtractor().extract(text)
    assert "550e8400-e29b-41d4-a716-446655440000" in entities


def test_extracts_ticket_ids() -> None:
    text = "See JIRA-1234 and ACME-99 for context."
    entities = EntityExtractor().extract(text)
    assert "JIRA-1234" in entities
    assert "ACME-99" in entities


def test_extracts_phone_numbers() -> None:
    text = "Call +1-555-867-5309 or (212) 555-1234."
    entities = EntityExtractor().extract(text)
    assert any("555-867-5309" in e for e in entities)


def test_extracts_money() -> None:
    text = "Total is $1,234.56 and €99.99 was refunded."
    entities = EntityExtractor().extract(text)
    assert any("$1,234.56" in e for e in entities)
    assert any("€99.99" in e for e in entities)


def test_extracts_ip_addresses() -> None:
    text = "Server at 192.168.1.100 and 10.0.0.1."
    entities = EntityExtractor().extract(text)
    assert "192.168.1.100" in entities
    assert "10.0.0.1" in entities


def test_extracts_id_like_values() -> None:
    text = "The account_id=acct_12345 and session_id: sess-abc-def."
    entities = EntityExtractor().extract(text)
    assert "acct_12345" in entities
    assert "sess-abc-def" in entities


def test_id_regex_does_not_match_words_like_identify() -> None:
    text = "Micro-expression analysis enables the system to identify subtle patterns. Identification matters."
    entities = EntityExtractor().extract(text)
    assert "entify" not in entities
    assert "entification" not in entities


def test_max_entities_respected() -> None:
    text = " ".join(f"user{i}@example.com" for i in range(100))
    entities = EntityExtractor(max_entities=5).extract(text)
    assert len(entities) <= 5


def test_empty_text() -> None:
    assert EntityExtractor().extract("") == []
