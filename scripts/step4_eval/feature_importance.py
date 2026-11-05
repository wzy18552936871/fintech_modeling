# Bridge: feature_importance -> 10. 特征重要性.py
import importlib, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "feature_importance", str(Path(__file__).parent / "10. 特征重要性.py")
)
mod = importlib.util.module_from_spec(spec)
sys.modules["feature_importance"] = mod
spec.loader.exec_module(mod)
extract_model_importance = mod.extract_model_importance
