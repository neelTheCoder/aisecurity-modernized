# Ported from millburnai/aisecurity.
"""Tests for the AES-256-GCM gallery encryption."""

import pytest

from aisecurity import crypto


def test_roundtrip():
    secret = b"\x00\x01\x02 some embedding bytes \xff"
    blob = crypto.encrypt(secret, "correct horse battery staple")
    assert crypto.decrypt(blob, "correct horse battery staple") == secret


def test_wrong_passphrase_fails():
    blob = crypto.encrypt(b"top secret", "right-pass")
    with pytest.raises(Exception):
        crypto.decrypt(blob, "wrong-pass")


def test_tamper_detection():
    blob = bytearray(crypto.encrypt(b"top secret", "pw"))
    blob[-1] ^= 0x01  # flip a ciphertext bit
    with pytest.raises(Exception):
        crypto.decrypt(bytes(blob), "pw")


def test_rejects_foreign_blob():
    with pytest.raises(ValueError):
        crypto.decrypt(b"not-an-aisec-blob", "pw")


def test_salt_and_nonce_are_random():
    a = crypto.encrypt(b"same input", "pw")
    b = crypto.encrypt(b"same input", "pw")
    assert a != b  # different salt+nonce => different ciphertext
