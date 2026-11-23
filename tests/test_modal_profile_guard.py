"""Modal runs must use the faheemgurkani profile."""

import pytest

from modal_app import settings


def _toml(tmp_path, active):
    path = tmp_path / "modal.toml"
    path.write_text(
        f'[faheemgurkani]\nactive = {str(active == "faheemgurkani").lower()}\n'
        f'[muhammadfaheem-mansha]\nactive = {str(active == "muhammadfaheem-mansha").lower()}\n'
    )
    return path


def test_guard_accepts_faheemgurkani(tmp_path, monkeypatch):
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    monkeypatch.delenv("KV_MODAL_PROFILE", raising=False)
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(_toml(tmp_path, "faheemgurkani")))
    settings.assert_expected_modal_profile()


def test_guard_rejects_other_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    monkeypatch.delenv("KV_MODAL_PROFILE", raising=False)
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(_toml(tmp_path, "muhammadfaheem-mansha")))
    with pytest.raises(RuntimeError, match="faheemgurkani"):
        settings.assert_expected_modal_profile()


def test_env_profile_overrides_toml(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(_toml(tmp_path, "muhammadfaheem-mansha")))
    monkeypatch.setenv("MODAL_PROFILE", "faheemgurkani")
    monkeypatch.delenv("KV_MODAL_PROFILE", raising=False)
    settings.assert_expected_modal_profile()
