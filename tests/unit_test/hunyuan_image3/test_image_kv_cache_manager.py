# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ImageKVCacheManager.

Pure-tensor logic — runs on CPU. Exercises:
  - set_prompt_kv (single + bulk)
  - inject_into_layer (rank-3 and rank-4 step shapes, positive/negative branch)
  - build_neg_ar_kv (shared-only and divergent-tail cases, validation errors)
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sglang_omni.models.hunyuan_image3.image_kv_cache_manager import (
    ImageKVCacheManager,
    PromptKVSlice,
    _prepend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _kv(seq_len: int, heads: int = 4, dim: int = 8, vdim: int | None = None):
    """Make a deterministic per-layer (K, V) pair."""
    g = torch.Generator().manual_seed(seq_len * 1000 + heads * 10 + dim)
    k = torch.randn(seq_len, heads, dim, generator=g)
    v = torch.randn(seq_len, heads, vdim or dim, generator=g)
    return k, v


# ---------------------------------------------------------------------------
# PromptKVSlice
# ---------------------------------------------------------------------------
class TestPromptKVSlice:
    def test_auto_seq_len(self):
        k, v = _kv(17)
        slice_ = PromptKVSlice(k=k, v=v)
        assert slice_.seq_len == 17

    def test_kv_mismatch_raises(self):
        k = torch.randn(5, 4, 8)
        v = torch.randn(7, 4, 8)
        with pytest.raises(ValueError, match="seq_len mismatch"):
            PromptKVSlice(k=k, v=v)


# ---------------------------------------------------------------------------
# set_prompt_kv / set_prompt_kv_bulk
# ---------------------------------------------------------------------------
class TestSetPromptKV:
    def test_single_layer(self):
        mgr = ImageKVCacheManager(num_layers=2)
        k, v = _kv(10)
        mgr.set_prompt_kv(0, k, v)
        assert mgr.positive_prefix_len(0) == 10

    def test_bulk(self):
        mgr = ImageKVCacheManager(num_layers=3)
        kv = {i: _kv(5 + i) for i in range(3)}
        mgr.set_prompt_kv_bulk(kv)
        for i in range(3):
            assert mgr.positive_prefix_len(i) == 5 + i

    def test_missing_layer_raises_on_access(self):
        mgr = ImageKVCacheManager(num_layers=2)
        with pytest.raises(KeyError, match="no installed AR prefix"):
            mgr.positive_prefix_len(0)


# ---------------------------------------------------------------------------
# inject_into_layer
# ---------------------------------------------------------------------------
class TestInjectIntoLayer:
    def test_rank3_step(self):
        mgr = ImageKVCacheManager(num_layers=1)
        prefix_k, prefix_v = _kv(6)
        mgr.set_prompt_kv(0, prefix_k, prefix_v)
        step_k, step_v = _kv(4)
        k_full, v_full = mgr.inject_into_layer(0, step_k, step_v)
        assert k_full.shape == (10, 4, 8)
        assert v_full.shape == (10, 4, 8)
        assert torch.allclose(k_full[:6], prefix_k)
        assert torch.allclose(k_full[6:], step_k)

    def test_rank4_step_broadcasts_batch(self):
        mgr = ImageKVCacheManager(num_layers=1)
        prefix_k, prefix_v = _kv(5)
        mgr.set_prompt_kv(0, prefix_k, prefix_v)
        batch = 3
        step_k = torch.randn(batch, 7, 4, 8)
        step_v = torch.randn(batch, 7, 4, 8)
        k_full, v_full = mgr.inject_into_layer(0, step_k, step_v)
        assert k_full.shape == (batch, 12, 4, 8)
        # Prefix broadcast across batch.
        for b in range(batch):
            assert torch.allclose(k_full[b, :5], prefix_k)
        assert torch.allclose(k_full[:, 5:], step_k)

    def test_negative_branch_unpopulated_raises(self):
        mgr = ImageKVCacheManager(num_layers=1)
        k, v = _kv(6)
        mgr.set_prompt_kv(0, k, v)
        step_k, step_v = _kv(2)
        with pytest.raises(RuntimeError, match="branch='negative'"):
            mgr.inject_into_layer(0, step_k, step_v, branch="negative")

    def test_invalid_rank_raises(self):
        mgr = ImageKVCacheManager(num_layers=1)
        k, v = _kv(3)
        mgr.set_prompt_kv(0, k, v)
        bad = torch.randn(2)  # rank 1
        with pytest.raises(ValueError, match="rank 3 or 4"):
            _prepend(k, v, bad, bad)


# ---------------------------------------------------------------------------
# build_neg_ar_kv
# ---------------------------------------------------------------------------
class TestBuildNegArKV:
    def test_shared_only(self):
        """neg_reuse_len == shared_prefix_len — no neg-only tail needed."""
        mgr = ImageKVCacheManager(num_layers=1)
        prefix_k, prefix_v = _kv(20)
        mgr.set_prompt_kv(0, prefix_k, prefix_v)

        mgr.build_neg_ar_kv(
            0,
            pos_reuse_len=20,
            neg_reuse_len=12,
            shared_prefix_len=12,
        )
        assert mgr.negative_prefix_len(0) == 12
        assert mgr.has_negative_branch(0)

        # Injecting on negative branch uses the shared slice.
        step_k, step_v = _kv(3)
        k_full, _ = mgr.inject_into_layer(0, step_k, step_v, branch="negative")
        assert k_full.shape == (15, 4, 8)
        assert torch.allclose(k_full[:12], prefix_k[:12])

    def test_divergent_tail(self):
        """neg_reuse_len > shared_prefix_len — caller supplies neg-only suffix."""
        mgr = ImageKVCacheManager(num_layers=1)
        prefix_k, prefix_v = _kv(20)
        mgr.set_prompt_kv(0, prefix_k, prefix_v)
        tail_k = torch.full((5, 4, 8), 7.0)
        tail_v = torch.full((5, 4, 8), -3.0)
        mgr.build_neg_ar_kv(
            0,
            pos_reuse_len=20,
            neg_reuse_len=15,
            shared_prefix_len=10,
            neg_only_k=tail_k,
            neg_only_v=tail_v,
        )
        assert mgr.negative_prefix_len(0) == 15
        step_k, step_v = _kv(2)
        k_full, v_full = mgr.inject_into_layer(0, step_k, step_v, branch="negative")
        assert k_full.shape == (17, 4, 8)
        assert torch.allclose(k_full[:10], prefix_k[:10])
        assert torch.allclose(k_full[10:15], tail_k)
        assert torch.allclose(v_full[10:15], tail_v)

    def test_tail_length_mismatch_raises(self):
        mgr = ImageKVCacheManager(num_layers=1)
        prefix_k, prefix_v = _kv(20)
        mgr.set_prompt_kv(0, prefix_k, prefix_v)
        bad_tail_k = torch.zeros(3, 4, 8)  # expected 5
        bad_tail_v = torch.zeros(3, 4, 8)
        with pytest.raises(ValueError, match="tail length must be 5"):
            mgr.build_neg_ar_kv(
                0,
                pos_reuse_len=20,
                neg_reuse_len=15,
                shared_prefix_len=10,
                neg_only_k=bad_tail_k,
                neg_only_v=bad_tail_v,
            )

    def test_missing_tail_raises_when_needed(self):
        mgr = ImageKVCacheManager(num_layers=1)
        prefix_k, prefix_v = _kv(20)
        mgr.set_prompt_kv(0, prefix_k, prefix_v)
        with pytest.raises(ValueError, match="requires neg_only_k/neg_only_v"):
            mgr.build_neg_ar_kv(
                0,
                pos_reuse_len=20,
                neg_reuse_len=15,
                shared_prefix_len=10,
            )

    def test_shared_gt_pos_reuse_raises(self):
        mgr = ImageKVCacheManager(num_layers=1)
        prefix_k, prefix_v = _kv(10)
        mgr.set_prompt_kv(0, prefix_k, prefix_v)
        with pytest.raises(ValueError, match="shared_prefix_len="):
            mgr.build_neg_ar_kv(
                0,
                pos_reuse_len=10,
                neg_reuse_len=10,
                shared_prefix_len=11,
            )

    def test_neg_less_than_shared_raises(self):
        mgr = ImageKVCacheManager(num_layers=1)
        prefix_k, prefix_v = _kv(10)
        mgr.set_prompt_kv(0, prefix_k, prefix_v)
        with pytest.raises(ValueError, match="neg_reuse_len="):
            mgr.build_neg_ar_kv(
                0,
                pos_reuse_len=10,
                neg_reuse_len=3,
                shared_prefix_len=5,
            )


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------
class TestClear:
    def test_clear_drops_all_layers(self):
        mgr = ImageKVCacheManager(num_layers=2)
        for i in range(2):
            k, v = _kv(4)
            mgr.set_prompt_kv(i, k, v)
        mgr.clear()
        with pytest.raises(KeyError):
            mgr.positive_prefix_len(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
