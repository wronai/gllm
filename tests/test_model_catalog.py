"""Tests for model_catalog — pure data + pure functions extracted from cli.py."""

from __future__ import annotations

from prellm.model_catalog import (
    MODEL_PAIRS,
    OPENROUTER_MODELS,
    list_model_pairs,
    list_openrouter_models,
)


def test_model_catalog_all_providers():
    """list_model_pairs() returns all pairs when no filter is applied."""
    results = list_model_pairs()
    assert len(results) == len(MODEL_PAIRS)
    assert all("name" in r and "small" in r and "large" in r and "cost" in r for r in results)


def test_model_catalog_filter_by_provider():
    """list_model_pairs() filters by provider name."""
    results = list_model_pairs(provider="ollama")
    assert len(results) >= 1
    assert all("ollama" in r["name"].lower() or "ollama" in r["small"].lower() or "ollama" in r["large"].lower() for r in results)


def test_model_catalog_filter_by_provider_case_insensitive():
    """Provider filter is case-insensitive."""
    results_lower = list_model_pairs(provider="ollama")
    results_upper = list_model_pairs(provider="OLLAMA")
    assert results_lower == results_upper


def test_model_catalog_search():
    """list_model_pairs() filters by search term."""
    results = list_model_pairs(search="kimi")
    assert len(results) >= 1
    assert any("kimi" in r["large"].lower() for r in results)


def test_model_catalog_search_no_match():
    """list_model_pairs() returns empty list for non-matching search."""
    results = list_model_pairs(search="nonexistent_model_xyz")
    assert results == []


def test_model_catalog_combined_filter():
    """list_model_pairs() applies both provider and search filters."""
    results = list_model_pairs(provider="ollama", search="free")
    assert len(results) >= 1
    assert all("ollama" in f"{r['name']} {r['small']} {r['large']}".lower() for r in results)


def test_openrouter_models_all():
    """list_openrouter_models() returns all models when no filter."""
    results = list_openrouter_models()
    assert len(results) == len(OPENROUTER_MODELS)
    assert all("model_id" in r and "description" in r for r in results)


def test_openrouter_models_search():
    """list_openrouter_models() filters by search term."""
    results = list_openrouter_models(search="kimi")
    assert len(results) >= 1
    assert any("kimi" in r["model_id"].lower() for r in results)


def test_openrouter_models_provider_filter():
    """list_openrouter_models() filters by provider."""
    results = list_openrouter_models(provider="openrouter")
    assert len(results) == len(OPENROUTER_MODELS)  # all are openrouter


def test_model_pairs_data_integrity():
    """All MODEL_PAIRS entries have 4 elements."""
    for pair in MODEL_PAIRS:
        assert len(pair) == 4, f"Invalid pair: {pair}"
        name, small, large, cost = pair
        assert isinstance(name, str) and name
        assert isinstance(small, str) and small
        assert isinstance(large, str) and large
        assert isinstance(cost, str) and cost
