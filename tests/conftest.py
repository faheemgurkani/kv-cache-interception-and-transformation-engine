"""Test-suite defaults: keep the real-model CPU smoke tests fast.

Production defaults (20 retrieval trials, throughput warm-up + 3 repeats; audit R1/S2) are
deliberately expensive. The smoke tests only need the code paths to execute, so shrink them
here. Tests that assert on the defaults pass explicit arguments instead.
"""

import os

os.environ.setdefault("KV_RETRIEVAL_TRIALS", "2")
os.environ.setdefault("KV_THROUGHPUT_WARMUP_TOKENS", "0")
os.environ.setdefault("KV_THROUGHPUT_REPEATS", "1")
os.environ.setdefault("KV_PPL_DECODE_TOKENS", "8")
