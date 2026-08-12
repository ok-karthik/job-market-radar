"""Config-driven search-pool construction.

These assert the SHAPE of the rendered pools, never the user's own config —
a fresh clone runs on config.example.json and must still pass.
"""
from scraper import user_config as uc


_GEOS = [
    {"geo_id": "1", "name": "Germany (remote)", "remote_only": True},
    {"geo_id": "2", "name": "Berlin", "remote_only": False},
    {"geo_id": "3", "name": "Munich", "remote_only": False},
]
_GROUPS = {"A-core": ["Platform Engineer"], "C-cloud": ["Cloud Engineer"]}


def _patch(monkeypatch, skip):
    monkeypatch.setattr(uc, "GEOS", _GEOS)
    monkeypatch.setattr(uc, "KEYWORD_GROUPS", _GROUPS)
    monkeypatch.setattr(uc, "SKIP_GEOS", skip)


def test_no_skip_yields_full_cross_product(monkeypatch):
    _patch(monkeypatch, {})
    assert len(uc.build_search_urls()) == 6  # 2 groups x 3 geos


def test_skip_geos_removes_only_the_named_pool(monkeypatch):
    """Suppressing Munich for one group must not touch the other group."""
    _patch(monkeypatch, {"A-core": ["Munich"]})
    urls = uc.build_search_urls()
    assert len(urls) == 5
    assert ("Munich" in [g for _, p, g in urls if p == "A-core"]) is False
    assert "Munich" in [g for _, p, g in urls if p == "C-cloud"]


def test_skip_geos_is_case_insensitive(monkeypatch):
    _patch(monkeypatch, {"A-core": ["munich"]})
    assert len(uc.build_search_urls()) == 5


def test_c_cloud_germany_pool_survives_a_city_wide_skip(monkeypatch):
    """Regression: the C-cloud CITY pools were cut on 2026-08-12, but its
    Germany pool carries every high scorer (incl. the corpus best at 100.0).
    Cutting it would silently lose the single most valuable pool."""
    _patch(monkeypatch, {"C-cloud": ["Berlin", "Munich"]})
    kept = [g for _, p, g in uc.build_search_urls() if p == "C-cloud"]
    assert kept == ["Germany (remote)"]


def test_remote_only_geo_sets_work_type_filter(monkeypatch):
    _patch(monkeypatch, {})
    urls = {g: u for u, _p, g in uc.build_search_urls()}
    assert "f_WT=2" in urls["Germany (remote)"]
    assert "f_WT=2" not in urls["Berlin"]        # city pools omit f_WT by design
    assert "f_JT=F" in urls["Berlin"]


def test_keywords_are_unquoted_and_or_joined(monkeypatch):
    """Quoting drops compound titles and NOT crashes the guest API (rule 3)."""
    monkeypatch.setattr(uc, "GEOS", _GEOS[:1])
    monkeypatch.setattr(uc, "KEYWORD_GROUPS", {"A": ["Platform Engineer", "SRE"]})
    monkeypatch.setattr(uc, "SKIP_GEOS", {})
    url = uc.build_search_urls()[0][0]
    assert "Platform+Engineer+OR+SRE" in url
    assert "%22" not in url and "NOT" not in url
