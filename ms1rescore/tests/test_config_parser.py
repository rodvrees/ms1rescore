"""Tests for parse_configurations() cascade logic."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from ms1rescore.config_parser import parse_configurations


def test_defaults_returned_when_no_config():
    config = parse_configurations()["ms1rescore"]
    assert config["model"] == "lda"
    assert config["decoy_method"] == "balanced_shuffle"
    assert config["features_exclude"] == []
    assert config["im2deep_calibration"] == "finetune"
    assert config["pseudo_label_fdr"] == pytest.approx(0.10)


def test_toml_overrides_defaults(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[ms1rescore]\nmodel = "catboost"\nfeatures_exclude = ["peptide_length"]\n')
    config = parse_configurations([str(toml)])["ms1rescore"]
    assert config["model"] == "catboost"
    assert config["features_exclude"] == ["peptide_length"]
    assert config["decoy_method"] == "balanced_shuffle"


def test_json_overrides_defaults(tmp_path):
    j = tmp_path / "cfg.json"
    j.write_text(json.dumps({"ms1rescore": {"train_fdr": 0.05}}))
    config = parse_configurations([str(j)])["ms1rescore"]
    assert config["train_fdr"] == pytest.approx(0.05)
    assert config["model"] == "lda"


def test_cli_namespace_overrides_file(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[ms1rescore]\nmodel = "catboost"\n')
    ns = Namespace(model="lda", train_fdr=None)
    config = parse_configurations([str(toml), ns])["ms1rescore"]
    assert config["model"] == "lda"
    assert config["train_fdr"] == pytest.approx(0.01)


def test_none_does_not_override(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[ms1rescore]\npseudo_label_fdr = 0.05\n')
    ns = Namespace(pseudo_label_fdr=None)
    config = parse_configurations([str(toml), ns])["ms1rescore"]
    assert config["pseudo_label_fdr"] == pytest.approx(0.05)


def test_schema_rejects_invalid_model(tmp_path):
    j = tmp_path / "bad.json"
    j.write_text(json.dumps({"ms1rescore": {"model": "xgboost"}}))
    with pytest.raises(Exception):
        parse_configurations([str(j)])


def test_features_exclude_unknown_names_allowed():
    ns = Namespace(features_exclude=["nonexistent_feature"])
    config = parse_configurations([ns])["ms1rescore"]
    assert "nonexistent_feature" in config["features_exclude"]


def test_maldi_extraction_section_preserved(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[ms1rescore.maldi_extraction]\nmatching_ppm = 15.0\n')
    config = parse_configurations([str(toml)])["ms1rescore"]
    assert config["maldi_extraction"]["matching_ppm"] == pytest.approx(15.0)
    assert config["maldi_extraction"]["ppm_bin"] == pytest.approx(5.0)


def test_im2deep_section_preserved(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('[ms1rescore.im2deep]\nfinetune_epochs = 20\n')
    config = parse_configurations([str(toml)])["ms1rescore"]
    assert config["im2deep"]["finetune_epochs"] == 20
    assert config["im2deep"]["finetune_batch_size"] == 64


def test_dict_source_overrides(tmp_path):
    override = {"ms1rescore": {"n_interaction_features": 3}}
    config = parse_configurations([override])["ms1rescore"]
    assert config["n_interaction_features"] == 3
    assert config["model"] == "lda"
