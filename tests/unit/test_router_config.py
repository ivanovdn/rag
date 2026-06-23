from config import Settings


def test_router_defaults():
    s = Settings(_env_file=None)  # ignore local .env; assert the code defaults
    assert s.router_enabled is True
    assert s.router_llm_model == ""
    assert s.router_confidence_floor == 0.6
