# OmniEngine Implementation Progress

## Overview

Tracking progress for implementing the OmniEngine framework as designed in [omni_engine_design.md](./omni_engine_design.md).

---

## Milestones

### Milestone 1: Core Types & Protocols
**Status**: ✅ Complete

| File | Status | Notes |
|------|--------|-------|
| `engines/base.py` | ✅ Exists | Already has Engine ABC + EchoEngine |
| `engines/omni/types.py` | ✅ Done | Request, RequestStatus, SchedulerOutput, RequestOutput, ModelRunnerOutput |
| `engines/omni/policy/base.py` | ✅ Done | SchedulingPolicy, InputPreparer, OutputProcessor protocols |
| `engines/omni/__init__.py` | ✅ Done | Exports types |
| `engines/omni/policy/__init__.py` | ✅ Done | Exports protocols |

---

### Milestone 2: Generic Scheduler
**Status**: ✅ Complete

| File | Status | Notes |
|------|--------|-------|
| `engines/omni/scheduler.py` | ✅ Done | Generic Scheduler class |

**Test**: Unit test with mock Policy

---

### Milestone 3: Generic ModelRunner
**Status**: ✅ Complete

| File | Status | Notes |
|------|--------|-------|
| `engines/omni/model_runner.py` | ✅ Done | Generic ModelRunner class |

**Test**: Unit test with mock components

---

### Milestone 4: OmniEngine
**Status**: ✅ Complete

| File | Status | Notes |
|------|--------|-------|
| `engines/omni/engine.py` | ✅ Done | OmniEngine class |
| `engines/omni/__init__.py` | ✅ Done | Public exports |

**Test**: Integration test with mock components

---

### Milestone 5: Encoder Support
**Status**: ✅ Complete

| File | Status | Notes |
|------|--------|-------|
| `engines/omni/policy/encoder.py` | ✅ Done | EncoderPolicy + data types + preparer + processor |
| `engines/omni/factory.py` | ✅ Done | `create_encoder_engine()` |

**Test**: Run with real BERT model

---

### Milestone 6: Simple AR Support
**Status**: ✅ Complete

| File | Status | Notes |
|------|--------|-------|
| `engines/omni/policy/ar.py` | ✅ Done | SimpleARPolicy + data types + preparer + processor |
| `engines/omni/factory.py` | ✅ Done | `create_simple_ar_engine()` |

**Test**: Run with small LLaMA/GPT-2

---

## File Structure (Target)

```
sglang_omni/engines/
├── base.py                    # ✅ Engine ABC (exists)
├── __init__.py                # Update exports
│
└── omni/
    ├── __init__.py
    ├── types.py               # M1
    ├── scheduler.py           # M2
    ├── model_runner.py        # M3
    ├── engine.py              # M4
    ├── factory.py             # M5, M6
    │
    └── policy/
        ├── __init__.py
        ├── base.py            # M1
        ├── encoder.py         # M5
        └── ar.py              # M6
```

---

## Cleanup Plan

After OmniEngine is working:
- [ ] Remove `engines/encoder/` (replaced by `engines/omni/policy/encoder.py`)
- [ ] Update `engines/__init__.py` to export OmniEngine

---

## Log

| Date | Milestone | Action |
|------|-----------|--------|
| 2026-01-19 | M1 | Core types & protocols implemented |
| 2026-01-19 | M2 | Generic Scheduler implemented |
| 2026-01-19 | M3 | Generic ModelRunner implemented |
| 2026-01-19 | M4 | OmniEngine implemented - Framework complete! |
| 2026-01-19 | M5 | Encoder support implemented |
| 2026-01-19 | M6 | Simple AR support implemented - All milestones complete! |
