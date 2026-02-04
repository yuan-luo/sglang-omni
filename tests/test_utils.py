from sglang_omni.utils import import_string


def test_import_string():
    sglang_omni = import_string("sglang_omni")
    schema = import_string("sglang_omni.config.schema")

    assert sglang_omni.__name__ == "sglang_omni"
    assert schema.__name__ == "sglang_omni.config.schema"
