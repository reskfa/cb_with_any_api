"""
iFind (同花顺) HTTP API 数据源适配器
为 cb_data 类提供 iFind 数据接口，与 wind/tushare/jqdata/akshare 平行

指标映射 (来自 参数.xlsx "Ths" 列，经验证修正):
  close              -> cmd_history_quotation (PriceType=2 净价)
  amt                -> cmd_history_quotation (amount)
  convvalue          -> date_sequence (ths_transfer_value_cbond)
  strbvalue          -> date_sequence (ths_pure_bond_value_cbond)
  convpremiumratio   -> date_sequence (ths_conversion_premium_rate_cbond)  # 注意: _cbond 非 _bond
  strbpremiumratio   -> date_sequence (ths_pure_bond_premium_rate_cbond)
  clause_conversion2_bondlot -> date_sequence (ths_un_conversion_balance_cbond)
  ytm_cb             -> cmd_history_quotation (yieldMaturity)
  ptmyear            -> cmd_history_quotation (remainingTerm)
  impliedvol         -> 本地计算 (BS定价 + 二分法，不依赖其他数据源)
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import requests
import json
import time
import os
import pickle

import const

# ── 常量 ──────────────────────────────────────────────────
BASE_URL = "https://quantapi.51ifind.com/api/v1"
BATCH_SIZE = 50  # 每批查询代码数
REQUEST_INTERVAL = 0.15  # 请求间隔(秒), 600/min 限制

# Wind 字段名 -> date_sequence 指标名
FIELD_MAP_DS = {
    'convvalue':          'ths_transfer_value_cbond',
    'strbvalue':          'ths_pure_bond_value_cbond',
    'convpremiumratio':   'ths_conversion_premium_rate_cbond',  # _cbond 非 _bond
    'strbpremiumratio':   'ths_pure_bond_premium_rate_cbond',
    'clause_conversion2_bondlot': 'ths_un_conversion_balance_cbond',
    'close':              'ths_bond_close_cbond',
    'amt':                'ths_bond_amt_cbond',
}

# cmd_history_quotation 支持的字段 (标准指标名)
FIELD_MAP_HQ = {
    'close':  'close',
    'amt':    'amount',
    'ytm_cb': 'yieldMaturity',
    'ptmyear': 'remainingTerm',
}


# ── Token 管理 ────────────────────────────────────────────
_token_cache = {'token': None, 'expires_at': None}
_TOKEN_CACHE_PATH = 'ifind_token_cache.pkl'


def _get_access_token():
    """
    获取有效的 access_token，自动刷新。
    - 缓存在内存和 pkl 文件中
    - access_token 有效期 7 天，提前 1 天刷新
    """
    global _token_cache

    # 检查内存缓存
    if _token_cache['token'] and _token_cache['expires_at']:
        if pd.Timestamp.now() < _token_cache['expires_at']:
            return _token_cache['token']

    # 尝试从磁盘加载
    if os.path.exists(_TOKEN_CACHE_PATH):
        try:
            with open(_TOKEN_CACHE_PATH, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('token') and cached.get('expires_at'):
                if pd.Timestamp.now() < cached['expires_at']:
                    _token_cache = cached
                    return _token_cache['token']
        except Exception:
            pass

    # 请求新的 access_token
    if not const.ifind_refresh_token:
        raise RuntimeError("ifind_refresh_token 未配置，请在 const.py 中设置")

    resp = requests.post(
        f'{BASE_URL}/get_access_token',
        headers={"Content-Type": "application/json", "refresh_token": const.ifind_refresh_token}
    )
    data = json.loads(resp.content)

    if data.get('errorcode') != 0:
        raise RuntimeError(f"获取 iFind access_token 失败: {data.get('errmsg', 'unknown error')}")

    token = data['data']['access_token']
    # expire_time 格式待确认，默认 7 天有效期，提前 1 天刷新
    expires_at = pd.Timestamp.now() + pd.Timedelta(days=6)

    _token_cache = {'token': token, 'expires_at': expires_at}

    # 持久化到磁盘
    try:
        with open(_TOKEN_CACHE_PATH, 'wb') as f:
            pickle.dump(_token_cache, f)
    except Exception:
        pass

    return token


# ── HTTP 请求封装 ─────────────────────────────────────────

def _ifind_request(endpoint, payload, max_retries=2):
    """
    通用 iFind HTTP POST 请求封装。
    - 自动附加 access_token
    - 处理 token 过期 (-1302) 自动重试
    - 返回解析后的 JSON 或抛出异常
    """
    url = f'{BASE_URL}/{endpoint}'
    access_token = _get_access_token()
    headers = {"Content-Type": "application/json", "access_token": access_token}

    for attempt in range(max_retries):
        resp = requests.post(url, json=payload, headers=headers)
        data = json.loads(resp.content)
        ec = data.get('errorcode')

        if ec == 0:
            return data

        # token 过期，刷新后重试
        if ec == -1302 and attempt < max_retries - 1:
            global _token_cache
            _token_cache = {'token': None, 'expires_at': None}
            if os.path.exists(_TOKEN_CACHE_PATH):
                os.remove(_TOKEN_CACHE_PATH)
            access_token = _get_access_token()
            headers = {"Content-Type": "application/json", "access_token": access_token}
            continue

        # 其他错误
        errmsg = data.get('errmsg', 'unknown')
        if ec == -4001:
            # 数据为空，返回 None
            return None
        raise RuntimeError(f"iFind API 错误: errorcode={ec}, errmsg={errmsg}, endpoint={endpoint}")

    return None


# ── 响应解析 ──────────────────────────────────────────────

def _parse_tables(tables, codes, indicator_name=None):
    """
    解析 iFind date_sequence / cmd_history_quotation 的 tables 结构。
    返回 DataFrame(index=DatetimeIndex, columns=codes)

    响应格式:
    {"tables": [{"thscode": "113050.SH", "time": [...], "table": {"indicator": [...]}}]}
    """
    if not tables:
        return None

    all_series = {}
    for item in tables:
        thscode = item.get('thscode', '')
        times = item.get('time', [])
        table = item.get('table', {})

        if not times or not table:
            continue

        # 获取第一个指标值列表
        if indicator_name and indicator_name in table:
            values = table[indicator_name]
        else:
            # 取 table 中第一个 key 的值
            values = list(table.values())[0] if table else []

        # 替换 None 为 NaN
        values = [np.nan if v is None else v for v in values]

        dates = pd.to_datetime(times)
        all_series[thscode] = pd.Series(values, index=dates)

    if not all_series:
        return None

    result = pd.DataFrame(all_series)
    result = result.sort_index()
    result = result.reindex(columns=codes)
    return result


# ── 交易日历 ──────────────────────────────────────────────

def _fetch_trade_dates(start, end):
    """
    获取交易日列表。
    POST /api/v1/get_trade_dates
    返回: list of date strings (YYYY-MM-DD)
    """
    para = {
        "marketcode": "212001",  # 上交所
        "startdate": pd.to_datetime(start).strftime('%Y%m%d'),
        "enddate": pd.to_datetime(end).strftime('%Y%m%d'),
        "functionpara": {"mode": "1", "dateType": "0", "dateFormat": "0", "period": "D"}
    }

    data = _ifind_request('get_trade_dates', para)
    if data is None:
        return []

    return data.get('tables', {}).get('time', [])


# ── 数据获取: date_sequence ───────────────────────────────

def _fetch_date_sequence(codes, indicator, start, end):
    """
    调用 date_sequence 端点获取时间序列数据。
    返回 DataFrame(index=DatetimeIndex, columns=codes)
    """
    start_str = pd.to_datetime(start).strftime('%Y-%m-%d')
    end_str = pd.to_datetime(end).strftime('%Y-%m-%d')

    para = {
        "codes": ",".join(codes),
        "startdate": start_str,
        "enddate": end_str,
        "indipara": [{"indicator": indicator, "indiparams": [""]}]
    }

    data = _ifind_request('date_sequence', para)
    if data is None:
        return None

    tables = data.get('tables', [])
    return _parse_tables(tables, codes, indicator)


# ── 数据获取: cmd_history_quotation ───────────────────────

def _fetch_historical_quotation(codes, indicators, start, end, price_type=None):
    """
    调用 cmd_history_quotation 端点获取历史行情。
    返回 dict: {indicator_name: DataFrame(index=DatetimeIndex, columns=codes)}

    Parameters
    ----------
    codes : list
    indicators : list of str, e.g. ['close', 'amount']
    start, end : str or datetime
    price_type : str, '1' for 全价, '2' for 净价 (债券专用)
    """
    start_str = pd.to_datetime(start).strftime('%Y-%m-%d')
    end_str = pd.to_datetime(end).strftime('%Y-%m-%d')

    para = {
        "codes": ",".join(codes),
        "indicators": ",".join(indicators),
        "startdate": start_str,
        "enddate": end_str,
    }
    if price_type:
        para["functionpara"] = {"PriceType": price_type}

    data = _ifind_request('cmd_history_quotation', para)
    if data is None:
        return None

    tables = data.get('tables', [])

    # 为每个指标分别构建 DataFrame
    result = {}
    for indicator in indicators:
        all_series = {}
        for item in tables:
            thscode = item.get('thscode', '')
            times = item.get('time', [])
            table = item.get('table', {})
            values = table.get(indicator, [])

            if not times or not values:
                continue

            values = [np.nan if v is None else v for v in values]
            dates = pd.to_datetime(times)
            all_series[thscode] = pd.Series(values, index=dates)

        if all_series:
            df = pd.DataFrame(all_series)
            df = df.sort_index()
            df = df.reindex(columns=codes)
            result[indicator] = df

    return result


# ── 批量查询 ──────────────────────────────────────────────

def _fetch_ds_in_batches(codes, indicator, start, end):
    """批量调用 date_sequence，每批 BATCH_SIZE 个代码。"""
    pieces = []
    for i in range(0, len(codes), BATCH_SIZE):
        batch = codes[i:i + BATCH_SIZE]
        df = _fetch_date_sequence(batch, indicator, start, end)
        if df is not None and not df.empty:
            pieces.append(df)
        if i + BATCH_SIZE < len(codes):
            time.sleep(REQUEST_INTERVAL)

    if not pieces:
        return None

    result = pd.concat(pieces, axis=1)
    result = result.reindex(columns=codes)
    return result


def _fetch_hq_in_batches(codes, indicators, start, end, price_type=None):
    """批量调用 cmd_history_quotation。"""
    pieces = {ind: [] for ind in indicators}

    for i in range(0, len(codes), BATCH_SIZE):
        batch = codes[i:i + BATCH_SIZE]
        result = _fetch_historical_quotation(batch, indicators, start, end, price_type)
        if result:
            for ind in indicators:
                if ind in result and result[ind] is not None and not result[ind].empty:
                    pieces[ind].append(result[ind])
        if i + BATCH_SIZE < len(codes):
            time.sleep(REQUEST_INTERVAL)

    final = {}
    for ind in indicators:
        if pieces[ind]:
            df = pd.concat(pieces[ind], axis=1)
            df = df.reindex(columns=codes)
            final[ind] = df

    return final if final else None


# ── 核心取数函数 ──────────────────────────────────────────

def fetch_ifind(codes, field, start, end):
    """
    从 iFind 获取可转债时间序列数据。
    返回 DataFrame(index=DatetimeIndex, columns=codes)，与 fetch_wind 格式一致。

    Parameters
    ----------
    codes : list
        债券代码列表，如 ['113050.SH', '128039.SZ']
    field : str
        Wind 字段名
    start, end : str or datetime
        日期范围
    """
    # 特殊字段路由
    if field == 'ytm_cb':
        return fetch_ytm_ifind(codes, start, end)
    elif field == 'ptmyear':
        trade_dates = _fetch_trade_dates(start, end)
        if not trade_dates:
            return None
        return fetch_ptm_ifind(codes, trade_dates)
    elif field == 'impliedvol':
        return fetch_impliedvol_ifind(codes, start, end)
    elif field == 'clause_conversion2_bondlot':
        trade_dates = _fetch_trade_dates(start, end)
        return fetch_outstanding_ifind(codes, trade_dates)

    # 优先使用 cmd_history_quotation (close, amt)
    if field in FIELD_MAP_HQ:
        hq_indicator = FIELD_MAP_HQ[field]
        price_type = '2' if field == 'close' else None  # 债券用净价
        result = _fetch_hq_in_batches(codes, [hq_indicator], start, end, price_type)
        if result and hq_indicator in result:
            return result[hq_indicator]

    # 使用 date_sequence (CB 专有指标)
    if field in FIELD_MAP_DS:
        indicator = FIELD_MAP_DS[field]
        return _fetch_ds_in_batches(codes, indicator, start, end)

    print(f"警告: 字段 '{field}' 在 iFind 中不可用")
    return None


# ── 特殊字段函数 ──────────────────────────────────────────

def fetch_ytm_ifind(codes, start, end):
    """获取到期收益率 (%)，使用 cmd_history_quotation yieldMaturity。"""
    result = _fetch_hq_in_batches(codes, ['yieldMaturity'], start, end)
    if result and 'yieldMaturity' in result:
        return result['yieldMaturity']
    return None


def fetch_ptm_ifind(codes, trade_dates):
    """获取剩余期限（年），使用 cmd_history_quotation remainingTerm。"""
    if not trade_dates:
        return None
    start = pd.to_datetime(trade_dates[0]).strftime('%Y-%m-%d')
    end = pd.to_datetime(trade_dates[-1]).strftime('%Y-%m-%d')
    result = _fetch_hq_in_batches(codes, ['remainingTerm'], start, end)
    if result and 'remainingTerm' in result:
        return result['remainingTerm']
    return None


def fetch_outstanding_ifind(codes, trade_dates):
    """获取剩余规模（元），使用 date_sequence ths_un_conversion_balance_cbond。"""
    if not trade_dates:
        return None
    start = pd.to_datetime(trade_dates[0]).strftime('%Y-%m-%d')
    end = pd.to_datetime(trade_dates[-1]).strftime('%Y-%m-%d')
    return _fetch_ds_in_batches(codes, 'ths_un_conversion_balance_cbond', start, end)


# ── 隐含波动率 (本地计算，不依赖其他数据源) ────────────────

_basic_cache_ifind = None


def _bs_cb(s, x, t, vol, r):
    """转债 BS 定价"""
    d1 = (np.log(s / x) + (r + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    ret = s * norm.cdf(d1) + x * np.exp(-r * t) * (1 - norm.cdf(d2))
    expired = t <= 0
    if np.any(expired):
        ret[expired] = np.maximum(s, x)[expired]
    return ret


def _implied_vol_batch(close, conv, t, maturitycallprice, rf=0.03, iterMax=500):
    """二分法批量计算隐含波动率。返回绝对数字。"""
    vol = np.full_like(close, 0.4, dtype=float)
    vol[close < np.maximum(conv, maturitycallprice * np.exp(-rf * t))] = 0.001

    vmax = np.full_like(close, 2.0, dtype=float)
    vmin = np.full_like(close, 0.001, dtype=float)

    for _ in range(iterMax):
        closeHat = _bs_cb(conv, maturitycallprice, t, vol, rf)
        diff = close - closeHat
        if np.nanmax(np.abs(diff)) <= 0.1:
            break
        vmax[diff < 0] = vol[diff < 0]
        vmin[diff > 0] = vol[diff > 0]
        vol = 0.5 * (vmax + vmin)

    return vol


def _load_basic_cache_ifind(codes=None):
    """
    从 iFind 获取可转债基础信息并缓存。
    完全基于 iFind API，不依赖 tushare/wind。
    """
    global _basic_cache_ifind
    if _basic_cache_ifind is None:
        _basic_cache_ifind = {}

    if codes is None:
        return _basic_cache_ifind

    # 找出缓存中没有的券
    new_codes = [c for c in codes if c not in _basic_cache_ifind]
    if not new_codes:
        return _basic_cache_ifind

    access_token = _get_access_token()
    headers = {"Content-Type": "application/json", "access_token": access_token}
    today = pd.Timestamp.now().strftime('%Y-%m-%d')

    # 批量获取正股代码 + 到期日
    for i in range(0, len(new_codes), BATCH_SIZE):
        batch = new_codes[i:i + BATCH_SIZE]

        # 初始化缓存条目
        for code in batch:
            if code not in _basic_cache_ifind:
                _basic_cache_ifind[code] = {
                    'stk_code': None,
                    'maturity_date': None,
                    'par': 100.0,
                    'add_rate': 0.0,
                    'maturity_call_price': np.nan,
                }

        # 获取正股代码
        para = {
            "codes": ",".join(batch),
            "indipara": [{"indicator": "ths_stock_code_cbond", "indiparams": [""]}]
        }
        try:
            resp = requests.post(f'{BASE_URL}/basic_data_service', json=para, headers=headers)
            data = json.loads(resp.content)
            if data.get('errorcode') == 0:
                for item in data.get('tables', []):
                    code = item.get('thscode', '')
                    val = item.get('table', {}).get('ths_stock_code_cbond', [None])
                    if val and val[0]:
                        _basic_cache_ifind[code]['stk_code'] = val[0]
        except Exception:
            pass
        time.sleep(REQUEST_INTERVAL)

        # 获取到期日
        df_mat = _fetch_date_sequence(batch, 'ths_maturity_date_cbond', today, today)
        if df_mat is not None:
            for code in batch:
                if code in df_mat.columns:
                    val = df_mat[code].dropna().iloc[-1] if not df_mat[code].dropna().empty else None
                    if pd.notna(val):
                        _basic_cache_ifind[code]['maturity_date'] = val

    # 尝试获取到期赎回价: 通过 iFind 的 date_sequence 获取近期纯债价值
    # 作为近似，如果无法获取 par 和 add_rate，使用默认值 100
    # 对于大部分可转债，maturity_call_price = par * (1 + add_rate/100)
    # 其中 par 通常为 100, add_rate 需要从 iFind 获取
    # 尝试获取 ths_redeem_price_cbond 或类似指标
    for i in range(0, len(new_codes), BATCH_SIZE):
        batch = new_codes[i:i + BATCH_SIZE]

        # 尝试获取面值和补偿利率
        para = {
            "codes": ",".join(batch),
            "indipara": [
                {"indicator": "ths_stock_code_cbond", "indiparams": [""]},
            ]
        }
        # 暂时无法从 iFind 获取 par 和 add_rate
        # 使用默认值: par=100, add_rate=0, maturity_call_price=100
        for code in batch:
            if code in _basic_cache_ifind:
                info = _basic_cache_ifind[code]
                info['maturity_call_price'] = info['par'] * (1 + info['add_rate'] / 100)

    # 尝试从纯债价值 + YTM 反推到期赎回价（更精确的方式）
    # 方法: 用到期日的纯债价值近似 = 到期赎回价 + 最后一期票息
    # 但这需要票息信息，iFind 暂时无法获取
    # 因此: 先用默认值 100，后续可通过超级命令获取更多指标后优化

    return _basic_cache_ifind


def _get_maturity_call_prices_ifind(codes):
    """
    获取到期赎回价 = par * (1 + add_rate/100)。
    完全基于 iFind 数据，不依赖 tushare。
    目前 iFind 无法获取 par 和 add_rate，使用默认值 100。
    """
    cache = _load_basic_cache_ifind(codes)
    result = {}
    for c in codes:
        if c in cache:
            result[c] = cache[c].get('maturity_call_price', 100.0)
        else:
            result[c] = 100.0  # 默认值
    return result


def fetch_impliedvol_ifind(codes, start, end):
    """
    计算可转债隐含波动率，返回 DataFrame(index=dates, columns=codes)，单位%。
    完全基于 iFind 数据，不依赖 tushare/wind。
    使用本地 BS 定价 + 二分法计算。
    """
    df_close = fetch_ifind(codes, 'close', start, end)
    df_conv = fetch_ifind(codes, 'convvalue', start, end)
    if df_close is None or df_conv is None:
        return None

    trade_dates_str = [d.strftime('%Y%m%d') for d in df_close.index]
    df_ptm = fetch_ptm_ifind(codes, trade_dates_str)

    mcps = _get_maturity_call_prices_ifind(codes)
    mcp_arr = np.array([mcps.get(c, 100.0) for c in codes], dtype=float)

    T, N = len(df_close), len(codes)

    close_2d = df_close.reindex(columns=codes).values.astype(float)
    conv_2d = df_conv.reindex(index=df_close.index, columns=codes).values.astype(float)
    t_2d = df_ptm.reindex(index=df_close.index, columns=codes).values.astype(float) if df_ptm is not None else np.full((T, N), np.nan)
    mcp_2d = np.tile(mcp_arr, (T, 1))

    valid = ~(np.isnan(close_2d) | np.isnan(conv_2d) | np.isnan(t_2d) | (t_2d <= 0))

    vol_2d = np.full((T, N), np.nan)
    if valid.any():
        vol_flat = _implied_vol_batch(
            close_2d[valid], conv_2d[valid], t_2d[valid], mcp_2d[valid]
        )
        vol_2d[valid] = vol_flat * 100  # 转为百分比

    return pd.DataFrame(vol_2d, index=df_close.index, columns=codes)


# ── 面板数据 ──────────────────────────────────────────────

def fetch_panel_from_ifind(codes):
    """
    从 iFind 获取面板（静态）数据，返回 DataFrame(index=codes)。
    iFind basic_data_service 对 CB 指标支持有限，缺失字段留 NaN。
    """
    panel_cols = [
        'name', 'creditrating', 'industry',
        'redeem_start', 'redeem_span', 'redeem_maxspan', 'redeem_trigger',
        'putback_start', 'putback_span', 'putback_maxspan', 'putback_trigger',
        'reset_span', 'reset_maxspan', 'reset_trigger',
        'maturity_price', 'underlyingcode', 'stock_code'
    ]
    result = pd.DataFrame(index=codes, columns=panel_cols)

    # 使用 basic_data_service 获取可用的字段
    access_token = _get_access_token()
    headers = {"Content-Type": "application/json", "access_token": access_token}

    # 1. 正股代码
    for i in range(0, len(codes), BATCH_SIZE):
        batch = codes[i:i + BATCH_SIZE]
        para = {
            "codes": ",".join(batch),
            "indipara": [{"indicator": "ths_stock_code_cbond", "indiparams": [""]}]
        }
        try:
            resp = requests.post(f'{BASE_URL}/basic_data_service', json=para, headers=headers)
            data = json.loads(resp.content)
            if data.get('errorcode') == 0:
                tables = data.get('tables', [])
                for item in tables:
                    code = item.get('thscode', '')
                    val = item.get('table', {}).get('ths_stock_code_cbond', [None])
                    if val and val[0]:
                        result.loc[code, 'underlyingcode'] = val[0]
                        result.loc[code, 'stock_code'] = val[0]
        except Exception as e:
            print(f"获取正股代码失败: {e}")
        time.sleep(REQUEST_INTERVAL)

    # 2. 债券名称
    for i in range(0, len(codes), BATCH_SIZE):
        batch = codes[i:i + BATCH_SIZE]
        para = {
            "codes": ",".join(batch),
            "indipara": [{"indicator": "ths_bond_name_bond", "indiparams": [""]}]
        }
        try:
            resp = requests.post(f'{BASE_URL}/basic_data_service', json=para, headers=headers)
            data = json.loads(resp.content)
            if data.get('errorcode') == 0:
                tables = data.get('tables', [])
                for item in tables:
                    code = item.get('thscode', '')
                    val = item.get('table', {}).get('ths_bond_name_bond', [None])
                    if val and val[0]:
                        result.loc[code, 'name'] = val[0]
        except Exception as e:
            print(f"获取债券名称失败: {e}")
        time.sleep(REQUEST_INTERVAL)

    # 3. 到期日 (用于计算 maturity_price)
    start = end = pd.Timestamp.now().strftime('%Y-%m-%d')
    for i in range(0, len(codes), BATCH_SIZE):
        batch = codes[i:i + BATCH_SIZE]
        df_mat = _fetch_date_sequence(batch, 'ths_maturity_date_cbond', start, end)
        if df_mat is not None:
            for code in batch:
                if code in df_mat.columns:
                    val = df_mat[code].iloc[-1] if not df_mat[code].empty else None
                    if pd.notna(val):
                        result.loc[code, 'maturity_date'] = val

    # 4. 到期赎回价: 基于 iFind 缓存
    cache = _load_basic_cache_ifind(codes)
    for code in codes:
        if code in cache and pd.notna(cache[code].get('maturity_call_price')):
            result.loc[code, 'maturity_price'] = cache[code]['maturity_call_price']

    return result


# ── 代码列表 ──────────────────────────────────────────────

def getCodeList_ifind():
    """
    获取全市场可转债代码列表（Wind 格式）。
    使用 data_pool 专题报表 或 basic_data_service 查询。
    """
    access_token = _get_access_token()
    headers = {"Content-Type": "application/json", "access_token": access_token}

    # 尝试使用智能选股查询可转债
    para = {
        "searchstring": "可转债",
        "searchtype": "stock"
    }
    try:
        resp = requests.post(f'{BASE_URL}/smart_stock_picking', json=para, headers=headers)
        data = json.loads(resp.content)
        if data.get('errorcode') == 0:
            tables = data.get('tables', [])
            codes = []
            for item in tables:
                thscode = item.get('thscode', '')
                # 只保留 SH/SZ 后缀的可转债代码
                if thscode and (thscode.endswith('.SH') or thscode.endswith('.SZ')):
                    # 可转债代码通常以 110/113/128/123/127 开头
                    num = thscode.split('.')[0]
                    if num.startswith(('110', '113', '128', '123', '127', '132', '120')):
                        codes.append(thscode)
            if codes:
                return codes
    except Exception as e:
        print(f"智能选股获取可转债列表失败: {e}")

    # 备选: 使用 basic_data_service 的 ths_stock_code_cbond 反查
    # 或者让用户通过其他途径获取代码列表
    print("警告: 无法通过 iFind API 获取可转债代码列表，请使用其他数据源的 getCodeList")
    return []


# ── 增量更新入口 ──────────────────────────────────────────

def update_from_df_ifind(df, end, field):
    """
    从 iFind 更新数据到现有 DataFrame，逻辑与 tushare_reader.update_from_df_tushare 一致。
    """
    codes = list(df.columns)
    last_date = pd.to_datetime(df.index[-1])
    end_date = pd.to_datetime(end)

    # 用 iFind 交易日历判断是否需要更新
    trade_dates = _fetch_trade_dates(last_date, end_date)

    if len(trade_dates) > 1:
        new_start = trade_dates[1]
        new_end = trade_dates[-1]

        if field == 'ytm_cb':
            df_new = fetch_ytm_ifind(codes, new_start, new_end)
        elif field == 'ptmyear':
            df_new = fetch_ptm_ifind(codes, trade_dates[1:])
        elif field == 'impliedvol':
            df_new = fetch_impliedvol_ifind(codes, new_start, new_end)
        elif field == 'clause_conversion2_bondlot':
            df_new = fetch_outstanding_ifind(codes, trade_dates[1:])
        else:
            df_new = fetch_ifind(codes, field, new_start, new_end)

        if df_new is not None and not df_new.empty:
            df_new.index = pd.to_datetime(df_new.index)
            df = pd.concat([df, df_new])
            return df
        else:
            print(f"{field} 从 iFind 获取新数据为空")
            return df
    else:
        print(f"{field} 不用更新")
        return df
