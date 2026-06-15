"""Tests for parse_configurations() cascade logic."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from msi_picasso.config_parser import parse_configurations


def test_defaults_returned_when_no_config():
    config = parse_configurations()["MSI-PICASSO"]
    assert config["model"] == "lda"
    assert config["decoy_method"] == "balanced_shuffle"
    assert config["features_exclude"] == []
    assert config["im2deep_calibration"] == "finetune"
    assert config["pseudo_label_fdr"] == pytest.approx(0.10)


def test_toml_overrides_defaults(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[MSI-PICASSO]\nmodel = "qda"\nfeatures_exclude = ["peptide_length"]\n')
    config = parse_configurations([str(toml)])["MSI-PICASSO"]
    assert config["model"] == "qda"
    assert config["features_exclude"] == ["peptide_length"]
    assert config["decoy_method"] == "balanced_shuffle"


def test_json_overrides_defaults(tmp_path):
    j = tmp_path / "cfg.json"
    j.write_text(json.dumps({"MSI-PICASSO": {"train_fdr": 0.05}}))
    config = parse_configurations([str(j)])["MSI-PICASSO"]
    assert config["train_fdr"] == pytest.approx(0.05)
    assert config["model"] == "lda"


def test_cli_namespace_overrides_file(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[MSI-PICASSO]\nmodel = "qda"\n')
    ns = Namespace(model="lda", train_fdr=None)
    config = parse_configurations([str(toml), ns])["MSI-PICASSO"]
    assert config["model"] == "lda"
    assert config["train_fdr"] == pytest.approx(0.05)


def test_none_does_not_override(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[MSI-PICASSO]\npseudo_label_fdr = 0.05\n')
    ns = Namespace(pseudo_label_fdr=None)
    config = parse_configurations([str(toml), ns])["MSI-PICASSO"]
    assert config["pseudo_label_fdr"] == pytest.approx(0.05)


def test_schema_rejects_invalid_model(tmp_path):
    j = tmp_path / "bad.json"
    j.write_text(json.dumps({"MSI-PICASSO": {"model": "xgboost"}}))
    with pytest.raises(Exception):
        parse_configurations([str(j)])


def test_features_exclude_unknown_names_allowed():
    ns = Namespace(features_exclude=["nonexistent_feature"])
    config = parse_configurations([ns])["MSI-PICASSO"]
    assert "nonexistent_feature" in config["features_exclude"]


def test_explicit_falsy_zero_is_honored():
    """A legitimate explicit 0 (param with minimum 0) must survive the cascade,
    not be silently dropped to the default (cascade_config falsy-drop bug)."""
    # n_interaction_features default is 0 already; use a non-default falsy case:
    # set it explicitly and confirm it round-trips.
    config = parse_configurations([Namespace(n_interaction_features=0)])["MSI-PICASSO"]
    assert config["n_interaction_features"] == 0


def test_explicit_matching_ppm_zero_honored():
    """matching_ppm=0 means exact matching / no collision tolerance — a valid choice
    (esp. in raw-query). It must be honored, not silently masked by the default 20.0."""
    config = parse_configurations([Namespace(matching_ppm=0.0)])["MSI-PICASSO"]
    assert config["matching_ppm"] == 0.0
    # a valid positive value is also honored
    config = parse_configurations([Namespace(matching_ppm=8.0)])["MSI-PICASSO"]
    assert config["matching_ppm"] == 8.0


def test_negative_matching_ppm_rejected():
    """A negative matching_ppm is invalid (schema minimum 0) and must raise."""
    with pytest.raises(Exception):
        parse_configurations([Namespace(matching_ppm=-1.0)])


def test_maldi_extraction_section_preserved(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[MSI-PICASSO.maldi_extraction]\nmatching_ppm = 15.0\n')
    config = parse_configurations([str(toml)])["MSI-PICASSO"]
    assert config["maldi_extraction"]["matching_ppm"] == pytest.approx(15.0)
    assert config["maldi_extraction"]["ppm_bin"] == pytest.approx(5.0)


def test_im2deep_section_preserved(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[MSI-PICASSO.im2deep]\nfinetune_epochs = 20\n')
    config = parse_configurations([str(toml)])["MSI-PICASSO"]
    assert config["im2deep"]["finetune_epochs"] == 20
    assert config["im2deep"]["finetune_batch_size"] == 64


def test_dict_source_overrides(tmp_path):
    override = {"MSI-PICASSO": {"n_interaction_features": 3}}
    config = parse_configurations([override])["MSI-PICASSO"]
    assert config["n_interaction_features"] == 3
    assert config["model"] == "lda"
