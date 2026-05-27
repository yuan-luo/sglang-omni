# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KV pack/unpack helpers + a mocked KVExporter gather.

Covers:
  - pack_per_layer_kv_for_relay / unpack_per_layer_kv_from_relay round-trip
  - KVExporter.gather_request via a fake pool/req_to_token
  - Layer-range subset gather
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sglang_omni.model_runner.kv_exporter import (
    KVExporter,
    PerLayerKV,
    pack_per_layer_kv_for_relay,
    unpack_per_layer_kv_from_relay,
)


# ---------------------------------------------------------------------------
# pack / unpack round-trip
# ---------------------------------------------------------------------------
class TestPackUnpack:
    def test_round_trip_keys(self):
        per_layer = {
            0: PerLayerKV(k=torch.ones(3, 4, 8), v=torch.zeros(3, 4, 8)),
            5: PerLayerKV(k=torch.full((3, 4, 8), 2.0), v=torch.full((3, 4, 8), 3.0)),
        }
        flat = pack_per_layer_kv_for_relay(per_layer)
        assert set(flat.keys()) == {"L0.k", "L0.v", "L5.k", "L5.v"}

        recovered = unpack_per_layer_kv_from_relay(flat)
        assert set(recovered.keys()) == {0, 5}
        for layer_id, original in per_layer.items():
            assert torch.equal(recovered[layer_id].k, original.k)
            assert torch.equal(recovered[layer_id].v, original.v)

    def test_unpack_ignores_unknown_keys(self):
        flat = {
            "L0.k": torch.ones(2, 1, 1),
            "L0.v": torch.zeros(2, 1, 1),
            "garbage_key": torch.ones(1),
            "L1.x": torch.ones(2, 1, 1),  # wrong side
            "Labc.k": torch.ones(2, 1, 1),  # not an int
        }
        out = unpack_per_layer_kv_from_relay(flat)
        assert set(out.keys()) == {0}

    def test_unpack_skips_orphan_sides(self):
        # Layer 7 has k but no v -> dropped.
        flat = {
            "L7.k": torch.ones(1),
            "L8.k": torch.ones(1),
            "L8.v": torch.zeros(1),
        }
        out = unpack_per_layer_kv_from_relay(flat)
        assert set(out.keys()) == {8}


# ---------------------------------------------------------------------------
# KVExporter.gather_request — backed by fake pool/req_to_token
# ---------------------------------------------------------------------------
class _FakeReqPool:
    def __init__(self, slot_table: torch.Tensor):
        self.req_to_token = slot_table


class _FakeTokenPool:
    """Mimics MHATokenToKVPool's public read API.

    `k_buffers` / `v_buffers` are lists indexed by layer_id (absolute).
    """

    def __init__(self, k_buffers, v_buffers, start_layer: int = 0):
        self.k_buffers = k_buffers
        self.v_buffers = v_buffers
        self.start_layer = start_layer
        self.end_layer = start_layer + len(k_buffers)
        self.layer_num = len(k_buffers)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffers[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffers[layer_id - self.start_layer]


class _FakeModelRunner:
    def __init__(self, token_pool, req_pool):
        self.token_to_kv_pool = token_pool
        self.req_to_token_pool = req_pool


class _FakeTPWorker:
    def __init__(self, runner):
        self.model_runner = runner


def _make_exporter(num_layers: int, pool_size: int, max_seq: int, num_reqs: int):
    """Build a fake exporter with deterministic content per (layer, slot)."""
    k_buffers, v_buffers = [], []
    for layer_id in range(num_layers):
        # Distinctive per-(layer, slot) signature: layer_id * 1000 + slot.
        k = torch.arange(pool_size, dtype=torch.float32).view(pool_size, 1, 1).expand(
            pool_size, 2, 4
        ).contiguous() + layer_id * 1000
        v = -k
        k_buffers.append(k)
        v_buffers.append(v)
    token_pool = _FakeTokenPool(k_buffers, v_buffers, start_layer=0)

    # Assign slots: request `r` gets slots [r * 100 + 0, r * 100 + 1, ...].
    slot_table = torch.zeros(num_reqs, max_seq, dtype=torch.long)
    for r in range(num_reqs):
        for t in range(max_seq):
            slot_table[r, t] = r * 100 + t
    req_pool = _FakeReqPool(slot_table)

    runner = _FakeModelRunner(token_pool, req_pool)
    worker = _FakeTPWorker(runner)
    return KVExporter(worker)


class TestKVExporterGather:
    def test_gather_full_prefix(self):
        exporter = _make_exporter(
            num_layers=3, pool_size=512, max_seq=64, num_reqs=2
        )
        per_layer = exporter.gather_request(rid=1, seq_len=10)
        assert set(per_layer.keys()) == {0, 1, 2}
        for layer_id, kv in per_layer.items():
            assert kv.k.shape == (10, 2, 4)
            assert kv.v.shape == (10, 2, 4)
            # Slot indices for rid=1 are 100..109. Pool[slot] = slot + layer*1000.
            expected_first = 100 + layer_id * 1000
            assert kv.k[0, 0, 0].item() == pytest.approx(expected_first)
            assert kv.v[0, 0, 0].item() == pytest.approx(-expected_first)

    def test_gather_is_a_copy_not_alias(self):
        exporter = _make_exporter(
            num_layers=2, pool_size=256, max_seq=32, num_reqs=1
        )
        per_layer = exporter.gather_request(rid=0, seq_len=5, clone=True)
        # Mutating output should not affect the pool.
        per_layer[0].k.fill_(-999.0)
        # Re-gather and confirm the pool still has original content.
        per_layer2 = exporter.gather_request(rid=0, seq_len=5, clone=True)
        assert not torch.equal(per_layer[0].k, per_layer2[0].k)

    def test_invalid_seq_len_raises(self):
        exporter = _make_exporter(
            num_layers=1, pool_size=64, max_seq=8, num_reqs=1
        )
        with pytest.raises(ValueError, match="seq_len must be positive"):
            exporter.gather_request(rid=0, seq_len=0)

    def test_gather_layer_range(self):
        exporter = _make_exporter(
            num_layers=4, pool_size=256, max_seq=32, num_reqs=1
        )
        out = exporter.gather_request_layer_range(rid=0, seq_len=3, layer_start=1, layer_stop=3)
        assert set(out.keys()) == {1, 2}
        # Layer 0 not included.
        assert 0 not in out

    def test_layer_range_out_of_bounds_raises(self):
        exporter = _make_exporter(
            num_layers=4, pool_size=256, max_seq=32, num_reqs=1
        )
        with pytest.raises(ValueError, match="not within owned"):
            exporter.gather_request_layer_range(
                rid=0, seq_len=3, layer_start=2, layer_stop=10
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
