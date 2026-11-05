"""
Python 环境自动切换模块

确保风控建模脚本运行在正确的 Anaconda 环境中。
如果当前不是 pypc 环境（或缺少 pyarrow），会自动 os.execl() 切换到 pypc。

用法：在每个入口脚本的最前面加上
    import ensure_env
即可自动完成环境切换。
"""

import os
import sys


PYPY_EXECUTABLE = r"C:\Users\Admin\.conda\envs\pypc\python.exe"


def _is_correct_env() -> bool:
    """检查当前是否已经在 pypc 环境中运行。"""
    exe = sys.executable
    # 如果已经是目标环境，直接返回 True
    if PYPY_EXECUTABLE in exe:
        return True
    # 如果 pyarrow 可用，也认为是正确的环境
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def ensure_env() -> None:
    """
    确保运行在 pypc 环境中。如果不在，自动切换到 pypc 重新执行当前脚本。
    """
    if _is_correct_env():
        return

    if not os.path.exists(PYPY_EXECUTABLE):
        print(f"[WARN] pypc Python 不存在: {PYPY_EXECUTABLE}")
        print("      请手动安装或使用正确的环境运行。")
        return

    # 切换到 pypc 环境重新执行当前脚本
    script = sys.argv[0]
    args = sys.argv[1:]

    print(f"[Env] Switching to pypc environment: {PYPY_EXECUTABLE}")

    # 用 os.execl 替换当前进程，不返回
    os.execl(PYPY_EXECUTABLE, PYPY_EXECUTABLE, script, *args)


# 模块导入时自动执行检查
ensure_env()
