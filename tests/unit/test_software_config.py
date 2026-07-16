from config import Settings


def test_software_defaults():
    s = Settings()
    assert s.software_lookup_enabled is True
    assert s.software_collection == "software_registry"
    assert s.software_docs_folder == "./software"
    assert s.software_registry_path == "./software/software_registry.json"
    assert s.software_fuzzy_threshold == 85.0
    assert s.software_min_semantic_score == 0.5
