"""
LightGBM fit() 兼容性工具模块

适配 LightGBM 4.x API 变化：
  - LightGBM >= 3.3.0: fit() 支持 callbacks= 替代 early_stopping_rounds / verbose
  - LightGBM >= 4.0.0: early_stopping_rounds / verbose 已完全移除

用法（所有脚本统一替换）：
    from lgb_fit_compat import safe_lgbm_fit, LGBM_API_V3_PLUS

    # 旧写法（不再使用）：
    # model.fit(X, y, eval_set=[...], early_stopping_rounds=100, verbose=False)

    # 新写法（推荐）：
    callbacks = safe_lgbm_fit.make_callbacks(early_stopping_rounds=100, verbose_interval=0)
    model.fit(X, y, eval_set=[...], callbacks=callbacks)

    # 或者直接调用封装函数：
    safe_lgbm_fit(model, X, y, eval_set=[...], early_stopping_rounds=100, verbose=False)
"""
from __future__ import print_function

import lightgbm as lgb


def _parse_lgb_version():
    """解析 LightGBM 版本元组，返回 (major, minor, patch)。"""
    v = lgb.__version__
    parts = v.split("-", 1)[0].split(".")
    return tuple(int(p) for p in parts[:3])


_LGB_VERSION = _parse_lgb_version()

# ----------------------------------------------------------
# 版本常量（方便 if 分支判断）
# ----------------------------------------------------------
LGBM_IS_V4_PLUS   = _LGB_VERSION >= (4, 0, 0)
LGBM_IS_V33_PLUS  = _LGB_VERSION >= (3, 3, 0)

# 旧 API: early_stopping_rounds / verbose 作为关键字参数传入 fit()
LGBM_USE_LEGACY_KW = not LGBM_IS_V33_PLUS

# ----------------------------------------------------------
# safe_lgbm_fit(model, X, y, ...)  — 一行替换 model.fit(...)
# ----------------------------------------------------------


def safe_lgbm_fit(
    model,
    X_train,
    y_train,
    eval_set=None,
    early_stopping_rounds=None,
    verbose=True,
    verbose_interval=0,
    feature_name=None,
    callbacks=None,
    **extra_kwargs,
):
    """
    兼容各版本 LightGBM 的 fit() 调用。

    Parameters
    ----------
    model : LGBMClassifier / LGBMRegressor
    early_stopping_rounds : int or None
        早停轮数。None = 不开早停。
    verbose : bool
        是否打印训练日志（仅 LightGBM < 3.3 有效，新版本被忽略）。
    verbose_interval : int
        多少轮打印一次日志。0 = 每轮都打印；-1 = 不打印（>= 3.3）。
    callbacks : list or None
        直接传入回调列表（>= 3.3 必选）。
    **extra_kwargs : 透传给原生 fit() 的额外参数。
    """
    if LGBM_USE_LEGACY_KW:
        # --- LightGBM < 3.3：用旧关键字参数 ---
        kw = dict(eval_set=eval_set)
        if early_stopping_rounds is not None:
            kw["early_stopping_rounds"] = early_stopping_rounds
        kw["verbose"] = verbose
        if feature_name is not None:
            kw["feature_name"] = feature_name
        kw.update(extra_kwargs)
        model.fit(X_train, y_train, **kw)
    else:
        # --- LightGBM >= 3.3：用 callbacks ---
        cb_list = []
        if callbacks is not None:
            cb_list.extend(callbacks)

        if early_stopping_rounds is not None:
            cb_list.append(lgb.callback.early_stopping(early_stopping_rounds))

        # verbose=False  → log_evaluation(-1)
        # verbose=True  → log_evaluation(verbose_interval or 1)
        if isinstance(verbose, bool):
            eval_freq = -1 if not verbose else max(verbose_interval, 1)
        else:
            eval_freq = int(verbose)

        cb_list.append(lgb.callback.log_evaluation(eval_freq))

        kw = dict(eval_set=eval_set) if eval_set else {}
        if feature_name is not None:
            kw["feature_name"] = feature_name
        kw["callbacks"] = cb_list if cb_list else None
        kw.update(extra_kwargs)
        model.fit(X_train, y_train, **kw)


# ----------------------------------------------------------
# 便捷回调构造器（供手写 callbacks 的场景使用）
# ----------------------------------------------------------

class make_callbacks:
    """
    factory object — call() returns a list of LightGBM callbacks.

    Examples
    --------
        from lgb_fit_compat import make_callbacks

        cbs = make_callbacks(early_stopping_rounds=100, verbose_interval=-1)
        model.fit(X, y, eval_set=[...], callbacks=cbs)
    """

    def __new__(cls, early_stopping_rounds=None, verbose_interval=0):
        cb_list = []
        if early_stopping_rounds is not None:
            cb_list.append(lgb.callback.early_stopping(early_stopping_rounds))
        cb_list.append(lgb.callback.log_evaluation(verbose_interval))
        return cb_list
