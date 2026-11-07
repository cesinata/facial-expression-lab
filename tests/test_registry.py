import pytest

from fel.data import (
    download,
    get_dataset,
    licence_notice,
    load_registry,
    validate_registry,
)


def test_real_registry_is_consistent():
    assert validate_registry(load_registry()) == []


def test_private_datasets_are_never_fetchable():
    reg = load_registry()
    for name in ("ckplus", "kdef", "hss_learning"):
        assert reg[name]["public"] is False
        assert reg[name]["fetch"] == "never"


def test_download_refuses_private(capsys):
    with pytest.raises(PermissionError):
        download("ckplus")
    assert "CK+" in capsys.readouterr().out  # licence printed before refusing


def test_download_refuses_user_supplied_dataset():
    with pytest.raises(PermissionError):
        download("jaffe")


def test_download_refuses_not_used_dataset():
    with pytest.raises(PermissionError):
        download("affectnet")


def test_download_unknown_name():
    with pytest.raises(KeyError):
        download("not_a_dataset")


def test_allowed_dataset_prints_licence_then_waits_for_fetcher(capsys):
    with pytest.raises(NotImplementedError):
        download("uibvfedplus_light")
    out = capsys.readouterr().out
    assert "CC BY 4.0" in out


def test_unconfirmed_licence_warns(capsys):
    with pytest.raises(NotImplementedError):
        download("fer2013")
    assert "unconfirmed" in capsys.readouterr().out


def test_jaffe_is_not_labelled_cc_by():
    assert "CC BY" not in get_dataset("jaffe")["licence"].replace("Not CC BY 4.0", "")


def test_validate_catches_public_false_with_fetch():
    bad = {
        "x": {
            "name": "x", "title": "X", "what": "w", "access": "a", "licence": "l",
            "licence_status": "unconfirmed", "public": False, "fetch": "runtime",
            "role": "used", "status": "planned", "last_checked": None,
        }
    }
    problems = validate_registry(bad)
    assert any("fetch must be 'never'" in p for p in problems)


def test_validate_catches_confirmed_without_date():
    bad = {
        "x": {
            "name": "x", "title": "X", "what": "w", "access": "a", "licence": "l",
            "licence_status": "confirmed", "public": True, "fetch": "runtime",
            "role": "used", "status": "planned", "last_checked": None,
        }
    }
    assert any("needs a last_checked" in p for p in validate_registry(bad))


def test_licence_notice_contains_status():
    assert "unconfirmed" in licence_notice("kdef")
