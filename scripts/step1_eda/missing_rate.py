# Bridge: missing_rate -> 1. 缺失率统计.py
import importlib, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "missing_rate", str(Path(__file__).parent / "1. 缺失率统计.py")
)
mod = importlib.util.module_from_spec(spec)
sys.modules["missing_rate"] = mod
spec.loader.exec_module(mod)
compute_missing_rate = mod.compute_missing_rate
