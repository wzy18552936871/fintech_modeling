# Bridge: data_preprocess -> 0. 数据预处理.py
import importlib, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "data_preprocess", str(Path(__file__).parent / "0. 数据预处理.py")
)
mod = importlib.util.module_from_spec(spec)
sys.modules["data_preprocess"] = mod
spec.loader.exec_module(mod)
explore_and_convert = mod.explore_and_convert
calc_monthly_label_stats = mod.calc_monthly_label_stats
