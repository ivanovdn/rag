from channels.teams.utils import safe_get_nested, strip_html


def test_safe_get_nested_returns_value():
    data = {"a": {"b": {"c": 42}}}
    assert safe_get_nested(data, "a", "b", "c") == 42


def test_safe_get_nested_missing_key_returns_default():
    data = {"a": {"b": {}}}
    assert safe_get_nested(data, "a", "b", "c") == ""
    assert safe_get_nested(data, "a", "x", "c", default="NA") == "NA"


def test_safe_get_nested_non_dict_intermediate_returns_default():
    data = {"a": "not-a-dict"}
    assert safe_get_nested(data, "a", "b", default=None) is None


def test_strip_html_removes_tags_and_collapses_whitespace():
    assert strip_html("<p>Hello   <b>world</b></p>") == "Hello world"


def test_strip_html_decodes_entities():
    assert strip_html("Tom &amp; Jerry &lt;3") == "Tom & Jerry <3"


def test_strip_html_empty_input():
    assert strip_html("") == ""
