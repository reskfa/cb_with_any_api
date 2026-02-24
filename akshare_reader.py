import akshare as ak
import pandas as pd
import numpy as np
import time
import os
import pickle
import re

from tushare_reader import _newton_ytm, _bs_cb, _implied_vol_batch


# ── 代码转换 ──────────────────────────────────────────────

def _to_ak_symbol(code):
    """'110092.SH' → 'sh110092', '128039.SZ' → 'sz128039'"""
    num, market = code.split('.')
    return market.lower() + num


def _to_ak_numeric(code):
    """'110092.SH' → '110092'"""
    return code.split('.')[0]


# ── 字段映射 ──────────────────────────────────────────────
# bond_zh_cov_value_analysis 返回列: 日期, 收盘价, 纯债价值, 转股价值, 纯债溢价率, 转股溢价率
# Wind 内部字段名 → (akshare 列名关键字, 缩放系数)
FIELD_MAP_AK = {
    'close':            ('收盘价', 1),
    'convvalue':        ('转股价值', 1),
    'convpremiumratio': ('转股溢价率', 1),
    'strbvalue':        ('纯债价值', 1),
    'strbpremiumratio': ('纯债溢价率', 1),
}


# ── 逐券历史数据缓存 ─────────────────────────────────────

_value_cache = {}   # {code: DataFrame} bond_zh_cov_value_analysis 缓存
_ohlcv_cache = {}   # {code: DataFrame} bond_zh_hs_cov_daily 缓存（用于 volume）


def _fetch_value_analysis(code, start, end):
    """单券查 bond_zh_cov_value_analysis，缓存结果。
    返回 DataFrame 含: 收盘价, 纯债价值, 转股价值, 纯债溢价率, 转股溢价率。"""
    if code in _value_cache:
        df = _value_cache[code]
        if df is not None:
            s, e = pd.to_datetime(start), pd.to_datetime(end)
            return df.loc[s:e]
        return None

    numeric = _to_ak_numeric(code)
    try:
        df = ak.bond_zh_cov_value_analysis(symbol=numeric)
    except Exception as e:
        print(f"bond_zh_cov_value_analysis({numeric}) 失败: {e}")
        _value_cache[code] = None
        return None

    if df is None or df.empty:
        _value_cache[code] = None
        return None

    df['date'] = pd.to_datetime(df['日期'])
    df = df.set_index('date').sort_index()
    _value_cache[code] = df

    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return df.loc[s:e]


def _fetch_ohlcv(code, start, end):
    """单券查 bond_zh_hs_cov_daily（仅用于获取 volume）。"""
    if code in _ohlcv_cache:
        df = _ohlcv_cache[code]
        if df is not None:
            s, e = pd.to_datetime(start), pd.to_datetime(end)
            return df.loc[s:e]
        return None

    symbol = _to_ak_symbol(code)
    try:
        df = ak.bond_zh_hs_cov_daily(symbol=symbol)
    except Exception as e:
        print(f"bond_zh_hs_cov_daily({symbol}) 失败: {e}")
        _ohlcv_cache[code] = None
        return None

    if df is None or df.empty:
        _ohlcv_cache[code] = None
        return None

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    _ohlcv_cache[code] = df

    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return df.loc[s:e]


def fetch_akshare(codes, field, start, end):
    """
    从 akshare 获取可转债历史时间序列数据。
    - close/convvalue/convpremiumratio/strbvalue/strbpremiumratio 来自 bond_zh_cov_value_analysis
    - amt 来自 bond_zh_hs_cov_daily 的 volume × close × 10

    Parameters
    ----------
    codes : list
        债券代码列表，如 ['110092.SH', '128039.SZ']
    field : str
        Wind 字段名
    start, end : str or datetime
        日期范围

    Returns
    -------
    DataFrame(index=dates, columns=codes)
    """
    is_amt = (field == 'amt')

    if not is_amt:
        mapping = FIELD_MAP_AK.get(field)
        if mapping is None:
            print(f"警告: 字段 '{field}' 在 akshare 中不可用，跳过")
            return None
        ak_col, scale = mapping

    all_series = {}
    for i, code in enumerate(codes):
        if is_amt:
            # Amt ≈ volume × close × 10（手→张→元）
            df = _fetch_ohlcv(code, start, end)
            if df is not None and 'volume' in df.columns and 'close' in df.columns:
                all_series[code] = df['volume'] * df['close'] * 10
            else:
                all_series[code] = pd.Series(dtype=float)
        else:
            df = _fetch_value_analysis(code, start, end)
            if df is not None:
                matched_col = None
                for c in df.columns:
                    if ak_col in c:
                        matched_col = c
                        break
                if matched_col is not None:
                    all_series[code] = df[matched_col].astype(float) * scale
                else:
                    all_series[code] = pd.Series(dtype=float)
            else:
                all_series[code] = pd.Series(dtype=float)

        if (i + 1) % 50 == 0:
            print(f"  akshare 进度: {i + 1}/{len(codes)}")

        time.sleep(0.3)

    if not all_series:
        return None

    result = pd.DataFrame(all_series)
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    result = result.reindex(columns=codes)
    return result


# ── 新浪静态信息缓存 ─────────────────────────────────────

_sina_cache = {}  # {code: dict}


def _load_sina_profile(code):
    """单券查 bond_cb_profile_sina，解析并缓存。"""
    if code in _sina_cache:
        return _sina_cache[code]

    symbol = _to_ak_symbol(code)
    try:
        df = ak.bond_cb_profile_sina(symbol=symbol)
    except Exception as e:
        print(f"bond_cb_profile_sina({symbol}) 失败: {e}")
        _sina_cache[code] = None
        return None

    if df is None or df.empty:
        _sina_cache[code] = None
        return None

    # df 通常是两列: item/value 或类似结构
    info = {}
    try:
        data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    except Exception:
        _sina_cache[code] = None
        return None

    # 解析到期日 — 匹配含"到期"的 key
    for key in data:
        if '到期' in key and '日' in key:
            try:
                info['maturity_date'] = pd.to_datetime(data[key])
            except Exception:
                pass
            break

    # 解析面值 — 匹配含"面值"的 key
    for key in data:
        if '面值' in key:
            try:
                num = re.search(r'[\d.]+', str(data[key]))
                if num:
                    info['par'] = float(num.group())
            except Exception:
                info['par'] = 100.0
            break
    if 'par' not in info:
        info['par'] = 100.0

    # 解析利率说明 — 匹配含"利率说明"的 key
    for key in data:
        if '利率说明' in key:
            info['coupon_desc'] = str(data[key])
            break

    # 解析发行规模 — 匹配含"发行规模"或"发行总额"的 key
    for key in data:
        if '发行规模' in key or '发行总额' in key:
            try:
                num = re.search(r'[\d.]+', str(data[key]))
                if num:
                    info['issue_size'] = float(num.group())
            except Exception:
                pass
            break

    _sina_cache[code] = info
    return info


def _parse_coupon_schedule(coupon_desc, par=100.0):
    """从利率说明文本解析出逐年票息金额列表。

    例: "第一年0.30%，第二年0.50%，...第六年2.00%"
    返回: [0.30, 0.50, 0.80, 1.00, 1.50, 2.00]  (par=100 时)
    """
    if not coupon_desc:
        return []

    rates = re.findall(r'(\d+\.?\d*)%', coupon_desc)
    if not rates:
        return []

    return [par * float(r) / 100 for r in rates]


# ── 东方财富对比表缓存 ───────────────────────────────────

_comparison_cache = None  # DataFrame, 全市场


def _load_comparison_cache():
    """调 bond_cov_comparison() 一次，缓存全市场到期赎回价。"""
    global _comparison_cache
    if _comparison_cache is not None:
        return _comparison_cache

    try:
        df = ak.bond_cov_comparison()
    except Exception as e:
        print(f"bond_cov_comparison() 失败: {e}")
        _comparison_cache = pd.DataFrame()
        return _comparison_cache

    _comparison_cache = df
    return _comparison_cache


def _get_maturity_call_price_ak(code):
    """从东方财富对比表获取到期赎回价。"""
    df = _load_comparison_cache()
    if df is None or df.empty:
        return np.nan

    numeric = _to_ak_numeric(code)

    # 尝试在代码列中匹配
    code_col = None
    price_col = None
    for c in df.columns:
        if c == '转债代码':
            code_col = c
        if c == '到期赎回价':
            price_col = c

    if code_col is None or price_col is None:
        return np.nan

    match = df[df[code_col].astype(str).str.contains(numeric)]
    if match.empty:
        return np.nan

    try:
        return float(match.iloc[0][price_col])
    except Exception:
        return np.nan


# ── bond_zh_cov 缓存（发行规模、信用评级等） ──────────────

_cov_cache = None


def _load_cov_cache():
    """调 bond_zh_cov() 一次，缓存全市场基本信息。"""
    global _cov_cache
    if _cov_cache is not None:
        return _cov_cache

    try:
        df = ak.bond_zh_cov()
    except Exception as e:
        print(f"bond_zh_cov() 失败: {e}")
        _cov_cache = pd.DataFrame()
        return _cov_cache

    _cov_cache = df
    return _cov_cache


# ── 现金流日历 ───────────────────────────────────────────

def build_cashflow_calendar_ak(codes, cache_path='cb_cashflow_calendar_ak.pkl'):
    """
    构建并缓存现金流日历（akshare 版本）。

    Returns
    -------
    dict : {code: {'dates': [datetime,...], 'amounts': [float,...]}}
    """
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    new_codes = [c for c in codes if c not in cache]
    if not new_codes:
        return {c: cache[c] for c in codes if c in cache}

    for i, code in enumerate(new_codes):
        profile = _load_sina_profile(code)
        if profile is None or 'maturity_date' not in profile:
            continue

        par = profile.get('par', 100.0)
        coupon_desc = profile.get('coupon_desc', '')
        maturity = profile['maturity_date']

        # 解析票息
        coupons = _parse_coupon_schedule(coupon_desc, par)
        if not coupons:
            continue

        # 获取到期赎回价
        mcp = _get_maturity_call_price_ak(code)
        if np.isnan(mcp):
            mcp = par  # 降级为面值

        # 构建现金流日期（假设年付息，从到期日往前推）
        n_years = len(coupons)
        cf_dates = []
        cf_amounts = []
        for j in range(n_years):
            # 从到期日往前推 (n_years - 1 - j) 年
            years_before = n_years - 1 - j
            try:
                dt = maturity - pd.DateOffset(years=years_before)
            except Exception:
                dt = maturity - pd.Timedelta(days=365 * years_before)

            cf_dates.append(dt)
            if j == n_years - 1:
                # 最后一期：票息 + 到期赎回价
                cf_amounts.append(coupons[j] + mcp)
            else:
                cf_amounts.append(coupons[j])

        cache[code] = {'dates': cf_dates, 'amounts': cf_amounts}

        if (i + 1) % 50 == 0:
            print(f"  现金流日历进度: {i + 1}/{len(new_codes)}")

        time.sleep(0.3)

    # 保存缓存
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

    return {c: cache[c] for c in codes if c in cache}


# ── 剩余期限 ─────────────────────────────────────────────

def fetch_ptm_akshare(codes, trade_dates):
    """
    计算可转债剩余期限（年）。

    Parameters
    ----------
    codes : list
        债券代码列表
    trade_dates : list
        交易日列表

    Returns
    -------
    DataFrame(index=dates, columns=codes)
    """
    # 获取到期日
    mat_dates = {}
    for code in codes:
        profile = _load_sina_profile(code)
        if profile and 'maturity_date' in profile:
            mat_dates[code] = profile['maturity_date']
        else:
            mat_dates[code] = None
        time.sleep(0.3)

    idx = pd.to_datetime(trade_dates)
    val_dates = idx.values.astype('datetime64[D]')
    result = pd.DataFrame(index=idx, columns=codes, dtype=float)

    for code in codes:
        md = mat_dates.get(code)
        if md is None:
            result[code] = np.nan
            continue
        maturity = np.datetime64(md, 'D')
        ptm = (maturity - val_dates).astype(np.float64) / 365.0
        ptm[ptm <= 0] = np.nan
        result[code] = ptm

    return result


# ── YTM ──────────────────────────────────────────────────

def fetch_ytm_akshare(codes, start, end, cache_path='cb_cashflow_calendar_ak.pkl'):
    """
    计算可转债 YTM，返回 DataFrame(index=dates, columns=codes)，单位%。
    复用 tushare_reader 的 _newton_ytm。
    """
    df_close = fetch_akshare(codes, 'close', start, end)
    if df_close is None or df_close.empty:
        return None

    cf_cal = build_cashflow_calendar_ak(codes, cache_path)

    val_dates = df_close.index.values.astype('datetime64[D]')
    result = pd.DataFrame(index=df_close.index, columns=codes, dtype=float)

    for code in codes:
        if code not in cf_cal:
            result[code] = np.nan
            continue

        cf = cf_cal[code]
        cf_dates = np.array(cf['dates'], dtype='datetime64[D]')
        cf_amounts = np.array(cf['amounts'], dtype=np.float64)

        prices = df_close[code].values.astype(np.float64)

        bad_price = np.isnan(prices) | (prices <= 0)
        prices_safe = np.where(bad_price, 100.0, prices)

        ytm = _newton_ytm(cf_dates, cf_amounts, prices_safe, val_dates)
        ytm[bad_price] = np.nan
        result[code] = ytm

    return result


# ── 隐含波动率 ───────────────────────────────────────────

def fetch_impliedvol_akshare(codes, start, end):
    """
    计算可转债隐含波动率，返回 DataFrame(index=dates, columns=codes)，单位%。
    复用 tushare_reader 的 _bs_cb + _implied_vol_batch。
    """
    df_close = fetch_akshare(codes, 'close', start, end)
    df_conv = fetch_akshare(codes, 'convvalue', start, end)
    if df_close is None or df_conv is None:
        return None

    trade_dates_str = [d.strftime('%Y%m%d') for d in df_close.index]
    df_ptm = fetch_ptm_akshare(codes, trade_dates_str)

    # 到期赎回价
    mcp_dict = {}
    for c in codes:
        mcp_dict[c] = _get_maturity_call_price_ak(c)
        if np.isnan(mcp_dict[c]):
            # 降级：从 sina 拿面值
            profile = _load_sina_profile(c)
            mcp_dict[c] = profile.get('par', 100.0) if profile else 100.0
    mcp_arr = np.array([mcp_dict[c] for c in codes], dtype=float)

    T, N = len(df_close), len(codes)

    close_2d = df_close.reindex(columns=codes).values.astype(float)
    conv_2d = df_conv.reindex(index=df_close.index, columns=codes).values.astype(float)
    t_2d = df_ptm.reindex(index=df_close.index, columns=codes).values.astype(float)
    mcp_2d = np.tile(mcp_arr, (T, 1))

    valid = ~(np.isnan(close_2d) | np.isnan(conv_2d) | np.isnan(t_2d) | (t_2d <= 0))

    vol_2d = np.full((T, N), np.nan)
    if valid.any():
        vol_flat = _implied_vol_batch(
            close_2d[valid], conv_2d[valid], t_2d[valid], mcp_2d[valid]
        )
        vol_2d[valid] = vol_flat * 100  # 转为百分比

    return pd.DataFrame(vol_2d, index=df_close.index, columns=codes)


# ── 剩余规模 ─────────────────────────────────────────────

def fetch_outstanding_akshare(codes, trade_dates):
    """
    获取可转债剩余规模（元），返回 DataFrame(index=dates, columns=codes)。
    从 bond_zh_cov() 拿发行规模 × 1e8 作为初始值，ffill 到 trade_dates。
    已知限制：不反映转股后的规模减少。
    """
    idx = pd.to_datetime(trade_dates)
    result = pd.DataFrame(index=idx, columns=codes, dtype=float)

    df_cov = _load_cov_cache()
    if df_cov is None or df_cov.empty:
        return result

    # 找发行规模列和代码列
    size_col = None
    code_col = None
    for c in df_cov.columns:
        if '发行规模' in c or '发行总额' in c:
            size_col = c
        if c == '债券代码':
            code_col = c

    if size_col is None or code_col is None:
        print(f"bond_zh_cov 列名不匹配，可用列: {list(df_cov.columns)}")
        return result

    for code in codes:
        numeric = _to_ak_numeric(code)
        match = df_cov[df_cov[code_col].astype(str).str.contains(numeric)]
        if not match.empty:
            try:
                size = float(match.iloc[0][size_col])
                # 发行规模通常单位为亿元，转换为元
                result[code] = size * 1e8
            except Exception:
                result[code] = np.nan

    return result


# ── 面板数据 ─────────────────────────────────────────────

def fetch_panel_from_akshare(codes):
    """
    从 akshare 获取面板（静态）数据，返回 DataFrame(index=codes)。
    列与 Wind 版 panel 对齐，不可获取的字段留 NaN。
    """
    panel_cols = [
        'name', 'creditrating', 'industry',
        'redeem_start', 'redeem_span', 'redeem_maxspan', 'redeem_trigger',
        'putback_start', 'putback_span', 'putback_maxspan', 'putback_trigger',
        'reset_span', 'reset_maxspan', 'reset_trigger',
        'maturity_price', 'underlyingcode', 'stock_code'
    ]
    result = pd.DataFrame(index=codes, columns=panel_cols)

    # 从 bond_zh_cov 拿名称、评级等
    df_cov = _load_cov_cache()
    code_col = None
    name_col = None
    rating_col = None
    if df_cov is not None and not df_cov.empty:
        for c in df_cov.columns:
            if c == '债券代码':
                code_col = c
            if c == '债券简称':
                name_col = c
            if c == '信用评级':
                rating_col = c

    for code in codes:
        numeric = _to_ak_numeric(code)

        # 从 bond_zh_cov 提取信息
        if code_col and df_cov is not None and not df_cov.empty:
            match = df_cov[df_cov[code_col].astype(str).str.contains(numeric)]
            if not match.empty:
                row = match.iloc[0]
                if name_col:
                    result.loc[code, 'name'] = row.get(name_col)
                if rating_col:
                    result.loc[code, 'creditrating'] = row.get(rating_col)

        # 从 comparison 表拿到期赎回价
        mcp = _get_maturity_call_price_ak(code)
        if not np.isnan(mcp):
            result.loc[code, 'maturity_price'] = mcp

    return result


# ── 交易日判断 ───────────────────────────────────────────

def _get_trade_dates_ak(start, end):
    """用 pandas bdate_range 近似获取交易日列表。"""
    return pd.bdate_range(start, end).strftime('%Y%m%d').tolist()


# ── 增量更新入口 ─────────────────────────────────────────

def update_from_df_akshare(df, end, field):
    """
    从 akshare 更新数据到现有 DataFrame，逻辑与 tushare_reader.update_from_df_tushare 一致。
    """
    codes = list(df.columns)
    last_date = pd.to_datetime(df.index[-1])
    end_date = pd.to_datetime(end)

    # 用 bdate_range 近似交易日
    trade_dates = pd.bdate_range(last_date, end_date).strftime('%Y%m%d').tolist()

    if len(trade_dates) > 1:
        new_start = trade_dates[1]
        new_end = trade_dates[-1]

        if field == 'ytm_cb':
            df_new = fetch_ytm_akshare(codes, new_start, new_end)
        elif field == 'ptmyear':
            df_new = fetch_ptm_akshare(codes, trade_dates[1:])
        elif field == 'impliedvol':
            df_new = fetch_impliedvol_akshare(codes, new_start, new_end)
        elif field == 'clause_conversion2_bondlot':
            df_new = fetch_outstanding_akshare(codes, trade_dates[1:])
        else:
            df_new = fetch_akshare(codes, field, new_start, new_end)

        if df_new is not None and not df_new.empty:
            df_new.index = pd.to_datetime(df_new.index)
            df = pd.concat([df, df_new])
            return df
        else:
            print(f"{field} 从 akshare 获取新数据为空")
            return df
    else:
        print(f"{field} 不用更新")
        return df
