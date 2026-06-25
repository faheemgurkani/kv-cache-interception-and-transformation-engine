"""Bit-accurate storage accounting for compressed KV payloads."""

from __future__ import annotations


def bits_to_bytes(num_bits: int) -> int:
    """Convert a bit count to packed storage bytes (rounded up)."""
    return (num_bits + 7) // 8


def index_storage_bits(num_indices: int, bitwidth: int) -> int:
    """Bits to store Lloyd-Max cluster indices (bitwidth bits each, not container dtype)."""
    return num_indices * bitwidth


def sign_storage_bits(num_signs: int) -> int:
    """Bits to store QJL sign codes (1 bit per sign, not int8 tensor bytes)."""
    return num_signs


def float32_storage_bits(num_values: int) -> int:
    return num_values * 32


def effective_bits_per_element(total_storage_bits: int, num_kv_elements: int) -> float:
    if num_kv_elements <= 0:
        return float("inf")
    return total_storage_bits / num_kv_elements
