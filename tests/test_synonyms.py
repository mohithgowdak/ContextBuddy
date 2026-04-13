from contextbuddy.synonyms import expand_synonyms, expand_query_terms


class TestExpandSynonyms:
    def test_known_synonym(self):
        group = expand_synonyms("car")
        assert "automobile" in group
        assert "vehicle" in group
        assert "car" in group

    def test_unknown_word(self):
        group = expand_synonyms("xylophone")
        assert group == frozenset({"xylophone"})

    def test_case_insensitive(self):
        group = expand_synonyms("Car")
        assert "automobile" in group

    def test_business_synonyms(self):
        group = expand_synonyms("payment")
        assert "remittance" in group or "disbursement" in group

    def test_tech_synonyms(self):
        group = expand_synonyms("error")
        assert "bug" in group

    def test_verb_synonyms(self):
        group = expand_synonyms("buy")
        assert "purchase" in group

    def test_adjective_synonyms(self):
        group = expand_synonyms("fast")
        assert "quick" in group
        assert "rapid" in group


class TestExpandQueryTerms:
    def test_expands_all_terms(self):
        expanded = expand_query_terms(["car", "fast"])
        assert "automobile" in expanded
        assert "quick" in expanded
        assert "car" in expanded
        assert "fast" in expanded

    def test_unknown_terms_pass_through(self):
        expanded = expand_query_terms(["xyzzy"])
        assert "xyzzy" in expanded

    def test_empty_input(self):
        expanded = expand_query_terms([])
        assert len(expanded) == 0
