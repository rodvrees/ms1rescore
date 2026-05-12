"""Unit tests for MALDI data extraction."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))
from maldi_data_extraction import (
    _fragpipe_to_proforma,
    parse_maldi_mgf,
    parse_maldi_mgf_title,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestTitleParsing:
    def test_full_title(self):
        title = (
            "+prm-PASEF(887.5042, 1033.5240, 1198.7000), "
            "72.1-138.9eV, 1/K0=1.471, X=252-477 Y=92-194, #1-6257"
        )
        result = parse_maldi_mgf_title(title)
        assert result["inv_k0"] == pytest.approx(1.471)
        assert result["collision_energy"] == "72.1-138.9"
        assert result["pixel_range_x"] == (252, 477)
        assert result["pixel_range_y"] == (92, 194)
        assert result["scan_range"] == (1, 6257)
        assert len(result["prm_targets"]) == 3
        assert result["prm_targets"][0] == pytest.approx(887.5042)

    def test_minimal_title(self):
        title = "1/K0=1.630"
        result = parse_maldi_mgf_title(title)
        assert result["inv_k0"] == pytest.approx(1.630)
        assert result["pixel_range_x"] is None

    def test_no_metadata(self):
        title = "some random title"
        result = parse_maldi_mgf_title(title)
        assert result["inv_k0"] is None
        assert result["prm_targets"] == []


class TestProFormaConversion:
    def test_hydroxyproline(self):
        assert _fragpipe_to_proforma("GSP[113]GPAGPK") == "GSP[UNIMOD:35]GPAGPK"

    def test_nterm_mod(self):
        assert _fragpipe_to_proforma("n[+43]PEPTIDEK") == "[UNIMOD:5]-PEPTIDEK"

    def test_no_mods(self):
        assert _fragpipe_to_proforma("PEPTIDEK") == "PEPTIDEK"

    def test_multiple_mods(self):
        result = _fragpipe_to_proforma("GGP[113]GGP[113]GPQGP[113]PGK")
        assert result == "GGP[UNIMOD:35]GGP[UNIMOD:35]GPQGP[UNIMOD:35]PGK"

    def test_unknown_mass(self):
        result = _fragpipe_to_proforma("PEPT[999]IDEK")
        assert result == "PEPT[+999]IDEK"

    def test_nan(self):
        import pandas as pd
        assert _fragpipe_to_proforma(pd.NA) is pd.NA


class TestMGFParsing:
    def test_parse_fixture(self):
        spectra = parse_maldi_mgf(FIXTURE_DIR / "test_maldi.mgf")
        assert len(spectra) == 2

    def test_spectrum_metadata(self):
        spectra = parse_maldi_mgf(FIXTURE_DIR / "test_maldi.mgf")
        s = spectra[0]
        assert s.precursor_mz == pytest.approx(887.50815)
        assert s.inv_k0 == pytest.approx(1.471)
        assert s.acquisition_mode == "iprm-PASEF"
        assert len(s.mz_array) == 10
        assert len(s.intensity_array) == 10

    def test_second_spectrum(self):
        spectra = parse_maldi_mgf(FIXTURE_DIR / "test_maldi.mgf")
        s = spectra[1]
        assert s.precursor_mz == pytest.approx(1198.69998)
        assert s.inv_k0 == pytest.approx(1.630)
