import importlib.util
import os
import sys

import pytest

MOCK = os.path.join(os.path.dirname(__file__), "mock_flamby")
SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "methods", "FedISIC2019.py")


@pytest.fixture()
def mock_flamby():
    sys.path.insert(0, MOCK)
    yield
    sys.path.remove(MOCK)
    for name in [m for m in sys.modules if m.startswith("flamby")]:
        del sys.modules[name]


@pytest.mark.parametrize("method", ["fedavg", "fedgela", "fedmcd"])
def test_isic_runs_on_mock_data(mock_flamby, tmp_path, method):
    spec = importlib.util.spec_from_file_location("fed_isic2019", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = module.main
    init = str(tmp_path / "init.pt")
    main(["--prepare-only", "--init-state", init])
    main(["--method", method, "--rounds", "2", "--updates", "3", "--init-state", init,
          "--output-dir", str(tmp_path)])
    lines = open(tmp_path / f"{method}_s0" / "metrics.csv").read().splitlines()
    assert len(lines) == 3
