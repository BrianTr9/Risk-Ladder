import importlib

import riskladder

SUBPACKAGES = ["env", "oracle", "agents", "safety", "harness"]


def test_version_is_set():
    assert riskladder.__version__ != "0.0.0"


def test_subpackages_import():
    for name in SUBPACKAGES:
        module = importlib.import_module(f"riskladder.{name}")
        assert module.__doc__, f"riskladder.{name} should describe its role"
