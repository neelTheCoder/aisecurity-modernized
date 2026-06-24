"""Authenticated encryption for the embedding gallery.

The original project's strongest idea was its privacy model: store only
irreversible face *embeddings* (never raw images) and encrypt them at rest. We
keep that property and modernise the primitives:

* **AES-256-GCM** (authenticated encryption) instead of unauthenticated AES —
  tampering with the ciphertext is now detected on decrypt.
* **scrypt** key derivation from a passphrase, with a per-file random salt,
  instead of a raw key blob committed next to the data.
* Implemented with `cryptography` (actively maintained) rather than the older
  `pycryptodome` bindings.

Threat model: an attacker who copies the gallery file off the device learns
nothing without the passphrase, and cannot forge or silently edit entries.
Face embeddings are one-way — the original photo cannot be reconstructed from
them — so even a decrypted gallery never exposes a face image.
"""

from __future__ import annotations

import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

# scrypt cost parameters (OWASP-recommended interactive defaults).
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_KEY_LEN = 32  # AES-256
_SALT_LEN = 16
_NONCE_LEN = 12

MAGIC = b"AISEC2"  # format/version marker prepended to every blob


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=_KEY_LEN, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt(plaintext: bytes, passphrase: str) -> bytes:
    """Encrypt ``plaintext`` and return a self-describing blob.

    Layout: ``MAGIC || salt(16) || nonce(12) || ciphertext+tag``.
    """
    salt = os.urandom(_SALT_LEN)
    nonce = os.urandom(_NONCE_LEN)
    key = _derive_key(passphrase, salt)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, MAGIC)
    return MAGIC + salt + nonce + ciphertext


def decrypt(blob: bytes, passphrase: str) -> bytes:
    """Reverse :func:`encrypt`. Raises on a wrong passphrase or tampering."""
    if not blob.startswith(MAGIC):
        raise ValueError("not an AISEC gallery blob (bad magic)")
    body = blob[len(MAGIC) :]
    salt, nonce, ciphertext = (
        body[:_SALT_LEN],
        body[_SALT_LEN : _SALT_LEN + _NONCE_LEN],
        body[_SALT_LEN + _NONCE_LEN :],
    )
    key = _derive_key(passphrase, salt)
    return AESGCM(key).decrypt(nonce, ciphertext, MAGIC)
