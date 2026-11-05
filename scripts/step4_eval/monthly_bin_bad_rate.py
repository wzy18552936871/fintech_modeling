# Bridge: monthly_bin_bad_rate -> 变量每月分箱占比坏率.py
import importlib, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "monthly_bin_bad_rate", str(Path(__file__).parent / "变量每月分箱占比坏率.py")
)
mod = importlib.util.module_from_spec(spec)
sys.modules["monthly_bin_bad_rate"] = mod
spec.loader.exec_module(mod)
draw_graph_batch = mod.draw_graph_batch
