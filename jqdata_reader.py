import pandas as pd
import numpy as np
import time
import os
import pickle
import re

import jqdatasdk as jq
import const

# 确保 jqdata 已认证
if const.jqdata_username and const.jqdata_password:
    if not jq.is_auth():
        jq.auth(const.jqdata_username, const.jqdata_password)

from tushare_reader import _newton_ytm, _bs_cb, _implied_vol_batch


# ── 默认回退链 ──────────────────────────────────────────────
FALLBACK_CHAIN = ['wind', 'tushare', 'akshare']


# ── 代码转换 ────────────────────────────────────────────────

def _to_jq_code(code):
    """Wind 代码 → jqdata 代码
    '110092.SH' → '110092.XSHG'
    '128013.SZ' → '128013.XSHE'
    """
    num, market = code.split('.')
    if market == 'SH':
        return num + '.XSHG'
    elif market == 'SZ':
        return num + '.XSHE'
    return code


def _from_jq_code(jq_code):
    """jqdata 代码 → Wind 代码
    '110092.XSHG' → '110092.SH'
    '128013.XSHE' → '128013.SZ'
    """
    num, market = jq_code.split('.')
    if market == 'XSHG':
        return num + '.SH'
    elif market == 'XSHE':
        return num + '.SZ'
    return jq_code


def _to_numeric_code(code):
    """Wind 代码 → 纯数字 (CONBOND_BASIC_INFO 中的 code 格式)
    '110092.SH' → '110092'
    """
    return code.split('.')[0]


# ── 回退链机制 ──────────────────────────────────────────────

def _fetch_with_fallback(codes, field, start, end, fallback_chain=None):
    """对 jqdata 不支持的字段，按回退链依次尝试其他数据源"""
    chain = fallback_chain or FALLBACK_CHAIN
    from cb_with_any_api import list_available_apis
    for source in chain:
        if source not in list_available_apis:
            continue
        try:
            if source == 'wind':
                import wind_reader
                return wind_reader.fetch_wind(codes, field, start, end)
            elif source == 'tushare':
                import tushare_reader
                return tushare_reader.fetch_tushare(codes, field, start, end)
            elif source == 'akshare':
                import akshare_reader
                return akshare_reader.fetch_akshare(codes, field, start, end)
        except Exception as e:
            print(f"回退到 {source} 获取 {field} 失败: {e}")
            continue
    return None


def _update_with_fallback(df, end, field, fallback_chain=None):
    """对 jqdata 不支持的字段，按回退链依次尝试更新"""
    chain = fallback_chain or FALLBACK_CHAIN
    from cb_with_any_api import list_available_apis
    for source in chain:
        if source not in list_available_apis:
            continue
        try:
            if source == 'wind':
                import wind_reader
                return wind_reader.update_from_df(df, end, field)
            elif source == 'tushare':
                import tushare_reader
                return tushare_reader.update_from_df_tushare(df, end, field)
            elif source == 'akshare':
                import akshare_reader
                return akshare_reader.update_from_df_akshare(df, end, field)
        except Exception as e:
            print(f"回退到 {source} 更新 {field} 失败: {e}")
            continue
    print(f"所有回退源均失败，{field} 未更新")
    return df


# ── 基础缓存 ────────────────────────────────────────────────

_basic_cache_jq = None


def _load_basic_cache_jq():
    """首次调用全量拉取 jqdata 可转债基础信息并缓存。

    一次性批量查询 CONBOND_BASIC_INFO，按纯数字 code 匹配。
    """
    global _basic_cache_jq
    if _basic_cache_jq is not None:
        return _basic_cache_jq

    _basic_cache_jq = {}

    # 获取所有可转债证券信息（jq_code 格式如 110067.XSHG）
    df_sec = jq.get_all_securities('conbond')
    if df_sec is None or df_sec.empty:
        return _basic_cache_jq

    # 建立 纯数字code → jq_code 的映射
    numeric_to_jq = {}
    for jq_code in df_sec.index:
        numeric = jq_code.split('.')[0]
        numeric_to_jq[numeric] = jq_code

    # 全量拉取 CONBOND_BASIC_INFO
    df_basic = jq.bond.run_query(
        jq.query(
            jq.bond.CONBOND_BASIC_INFO.code,
            jq.bond.CONBOND_BASIC_INFO.short_name,
            jq.bond.CONBOND_BASIC_INFO.company_code,
            jq.bond.CONBOND_BASIC_INFO.convert_price,
            jq.bond.CONBOND_BASIC_INFO.maturity_date,
            jq.bond.CONBOND_BASIC_INFO.par,
            jq.bond.CONBOND_BASIC_INFO.list_date,
        ).limit(5000)
    )

    # 建立 纯数字code → basic_info 的映射
    basic_map = {}
    if df_basic is not None and not df_basic.empty:
        for _, row in df_basic.iterrows():
            basic_map[str(row['code'])] = row

    # 尝试获取到期赎回价（从 tushare 缓存）
    mcp_map = _load_maturity_call_prices_from_tushare()

    # 合并
    for jq_code in df_sec.index:
        wind_code = _from_jq_code(jq_code)
        numeric = jq_code.split('.')[0]
        sec_row = df_sec.loc[jq_code]

        basic_row = basic_map.get(numeric)

        if basic_row is not None:
            maturity_date = pd.to_datetime(basic_row['maturity_date']) if pd.notna(basic_row.get('maturity_date')) else None
            par = float(basic_row['par']) if pd.notna(basic_row.get('par')) else 100.0
            convert_price = float(basic_row['convert_price']) if pd.notna(basic_row.get('convert_price')) else np.nan
            stk_code = basic_row.get('company_code') if pd.notna(basic_row.get('company_code')) else None
        else:
            maturity_date = None
            par = 100.0
            convert_price = np.nan
            stk_code = None

        # 到期赎回价：优先 tushare，否则默认 par
        mcp = mcp_map.get(wind_code, par)

        _basic_cache_jq[wind_code] = {
            'jq_code': jq_code,
            'numeric_code': numeric,
            'maturity_date': maturity_date,
            'par': par,
            'convert_price': convert_price,
            'maturity_call_price': mcp,
            'stk_code': stk_code,
            'start_date': sec_row.get('start_date'),
            'name': sec_row.get('display_name'),
        }

    return _basic_cache_jq


def _load_maturity_call_prices_from_tushare():
    """尝试从 tushare 获取到期赎回价映射。jqdata 没有此字段。"""
    try:
        from cb_with_any_api import list_available_apis
        if 'tushare' in list_available_apis:
            from tushare_reader import _get_maturity_call_prices, _load_basic_cache
            _load_basic_cache()
            import tushare_reader
            cache = tushare_reader._basic_cache
            if cache:
                result = {}
                for ts_code, info in cache.items():
                    # ts_code 与 wind_code 相同
                    par = info.get('par', 100) or 100
                    add_rate = info.get('add_rate', 0) or 0
                    result[ts_code] = par * (1 + add_rate / 100)
                return result
    except Exception as e:
        print(f"从 tushare 获取到期赎回价失败: {e}")
    return {}


def _get_maturity_dates_jq(codes):
    """获取到期日"""
    cache = _load_basic_cache_jq()
    return {c: cache[c]['maturity_date'] if c in cache else None for c in codes}


def _get_maturity_call_prices_jq(codes):
    """获取到期赎回价"""
    cache = _load_basic_cache_jq()
    result = {}
    for c in codes:
        if c in cache:
            result[c] = cache[c]['maturity_call_price']
        else:
            result[c] = np.nan
    return result


# ── 转股价 ──────────────────────────────────────────────────

def _get_convert_price(code):
    """获取可转债当前转股价。
    jqdata 没有转股价调整历史表，只有 CONBOND_BASIC_INFO 的初始转股价。
    对于需要精确转股价的场景，使用初始值（大部分券不会调整）。
    """
    cache = _load_basic_cache_jq()
    if code in cache:
        return cache[code].get('convert_price', np.nan)
    return np.nan


# ── 核心取数函数 ────────────────────────────────────────────

def fetch_jqdata(codes, field, start, end):
    """
    从 jqdata 获取可转债时间序列数据。
    返回 DataFrame(index=dates, columns=codes)，与 fetch_wind 格式一致。

    Parameters
    ----------
    codes : list
        债券代码列表，如 ['110092.SH', '128039.SZ']
    field : str
        Wind 字段名
    start, end : str or datetime
        日期范围
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    # 回退字段
    if field in ('strbvalue', 'strbpremiumratio', 'clause_conversion2_bondlot'):
        return _fetch_with_fallback(codes, field, start, end)

    if field == 'close':
        return _fetch_close_jq(codes, start_dt, end_dt)
    elif field == 'amt':
        return _fetch_amt_jq(codes, start_dt, end_dt)
    elif field == 'convvalue':
        return _fetch_convvalue_jq(codes, start_dt, end_dt)
    elif field == 'convpremiumratio':
        return _fetch_convprem_jq(codes, start_dt, end_dt)
    else:
        print(f"警告: 字段 '{field}' 在 jqdata 中不可用，尝试回退链")
        return _fetch_with_fallback(codes, field, start, end)


def _fetch_price_field_jq(codes, jq_field, start, end):
    """从 jqdata 批量获取可转债价格字段。
    处理 get_price 对单券和多券返回格式不同的问题。
    """
    jq_codes = [_to_jq_code(c) for c in codes]

    if len(jq_codes) == 1:
        # 单券: get_price 返回 index=date, columns=[field]
        df = jq.get_price(jq_codes[0], start_date=start, end_date=end,
                          frequency='daily', fields=[jq_field], panel=False)
        if df is None or df.empty:
            return None
        result = pd.DataFrame(index=df.index, columns=codes)
        result[codes[0]] = df[jq_field].values
        result.index = pd.to_datetime(result.index).normalize()
        return result
    else:
        # 多券: get_price 返回长表 with time, code, field columns
        df = jq.get_price(jq_codes, start_date=start, end_date=end,
                          frequency='daily', fields=[jq_field], panel=False)
        if df is None or df.empty:
            return None
        df['date'] = pd.to_datetime(df['time']).dt.normalize()
        pivot = df.pivot_table(index='date', columns='code', values=jq_field)
        pivot.columns = [_from_jq_code(c) for c in pivot.columns]
        pivot = pivot.reindex(columns=codes)
        return pivot


def _fetch_close_jq(codes, start, end):
    """从 jqdata 获取收盘价"""
    return _fetch_price_field_jq(codes, 'close', start, end)


def _fetch_amt_jq(codes, start, end):
    """从 jqdata 获取成交额（元）"""
    return _fetch_price_field_jq(codes, 'money', start, end)


def _fetch_convvalue_jq(codes, start, end):
    """计算转股价值 = stock_close × (100 / conversion_price)
    使用 CONBOND_BASIC_INFO 的 convert_price（初始转股价）。
    """
    cache = _load_basic_cache_jq()

    # 获取交易日
    trade_dates = jq.get_trade_days(start_date=start, end_date=end)
    trade_dates_idx = pd.DatetimeIndex(trade_dates)

    result = pd.DataFrame(index=trade_dates_idx, columns=codes, dtype=float)

    # 收集需要查询的正股代码
    stk_map = {}  # stk_jq_code → [wind_code, ...]
    conv_prices = {}  # wind_code → convert_price
    for code in codes:
        if code not in cache:
            continue
        info = cache[code]
        stk_code = info.get('stk_code')
        cp = info.get('convert_price')
        if not stk_code or '.' not in str(stk_code) or np.isnan(cp) if isinstance(cp, float) else not cp:
            continue
        conv_prices[code] = float(cp)
        stk_map.setdefault(stk_code, []).append(code)

    if not stk_map:
        return result

    # 批量获取正股收盘价
    stk_codes_list = list(stk_map.keys())
    if len(stk_codes_list) == 1:
        df_stk = jq.get_price(stk_codes_list[0], start_date=start, end_date=end,
                               frequency='daily', fields=['close'], panel=False)
        if df_stk is not None and not df_stk.empty:
            df_stk.index = pd.to_datetime(df_stk.index).normalize()
            for code in stk_map[stk_codes_list[0]]:
                cp = conv_prices.get(code)
                if cp:
                    result[code] = df_stk['close'] * (100.0 / cp)
    else:
        # 分批获取正股（避免一次请求太多）
        for i in range(0, len(stk_codes_list), 50):
            batch = stk_codes_list[i:i+50]
            try:
                df_stk = jq.get_price(batch, start_date=start, end_date=end,
                                       frequency='daily', fields=['close'], panel=False)
                if df_stk is None or df_stk.empty:
                    continue
                df_stk['date'] = pd.to_datetime(df_stk['time']).dt.normalize()
                for stk_code in batch:
                    sub = df_stk[df_stk['code'] == stk_code].set_index('date')['close']
                    sub = sub.reindex(trade_dates_idx)
                    for bond_code in stk_map.get(stk_code, []):
                        cp = conv_prices.get(bond_code)
                        if cp:
                            result[bond_code] = sub * (100.0 / cp)
            except Exception as e:
                print(f"获取正股价格失败 (batch {i}): {e}")

    return result


def _fetch_convprem_jq(codes, start, end):
    """计算转股溢价率 = (close - convvalue) / convvalue × 100"""
    df_close = _fetch_close_jq(codes, start, end)
    df_conv = _fetch_convvalue_jq(codes, start, end)

    if df_close is None or df_conv is None:
        return None

    # 对齐 index
    common_idx = df_close.index.intersection(df_conv.index)
    close = df_close.loc[common_idx]
    conv = df_conv.loc[common_idx]

    conv_prem = (close - conv) / conv * 100
    return conv_prem.reindex(columns=codes)


# ── 剩余期限 ────────────────────────────────────────────────

def fetch_ptm_jqdata(codes, trade_dates):
    """
    计算可转债剩余期限（年）。

    Parameters
    ----------
    codes : list
        债券代码列表
    trade_dates : list
        交易日列表
    """
    mat_dates = _get_maturity_dates_jq(codes)

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


# ── 现金流日历 ──────────────────────────────────────────────

def build_cashflow_calendar_jq(codes, cache_path='cb_cashflow_calendar_jq.pkl'):
    """
    构建并缓存现金流日历（jqdata 版本）。
    使用 BOND_COUPON 表获取逐年票息。

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

    basic_cache = _load_basic_cache_jq()

    # 批量查询 BOND_COUPON
    numeric_codes = []
    code_map = {}  # numeric → wind_code
    for code in new_codes:
        if code in basic_cache:
            numeric = basic_cache[code]['numeric_code']
            numeric_codes.append(numeric)
            code_map[numeric] = code

    if not numeric_codes:
        return {c: cache[c] for c in codes if c in cache}

    # 分批查询 BOND_COUPON（limit 5000 应该够）
    coupon_pieces = []
    for i in range(0, len(numeric_codes), 100):
        batch = numeric_codes[i:i+100]
        try:
            df_coupon = jq.bond.run_query(
                jq.query(jq.bond.BOND_COUPON).filter(
                    jq.bond.BOND_COUPON.code.in_(batch)
                ).limit(5000)
            )
            if df_coupon is not None and not df_coupon.empty:
                coupon_pieces.append(df_coupon)
        except Exception as e:
            print(f"查询 BOND_COUPON 失败: {e}")

    if coupon_pieces:
        df_coupon_all = pd.concat(coupon_pieces, ignore_index=True)
    else:
        df_coupon_all = pd.DataFrame()

    for numeric in numeric_codes:
        wind_code = code_map[numeric]
        if wind_code not in basic_cache:
            continue

        info = basic_cache[wind_code]
        par = info.get('par', 100.0)
        mcp = info.get('maturity_call_price', par)
        maturity_date = info.get('maturity_date')

        if maturity_date is None:
            continue

        # 从 BOND_COUPON 获取该券的票息记录
        if df_coupon_all.empty:
            continue

        sub = df_coupon_all[df_coupon_all['code'] == numeric].copy()
        if sub.empty:
            continue

        sub['coupon_end_date'] = pd.to_datetime(sub['coupon_end_date'])
        sub = sub.sort_values('coupon_end_date')

        cf_dates = []
        cf_amounts = []
        n = len(sub)
        for j, (_, row) in enumerate(sub.iterrows()):
            dt = row['coupon_end_date']
            coupon_rate = float(row['coupon']) if pd.notna(row.get('coupon')) else 0.0
            coupon = par * coupon_rate / 100

            if j == n - 1:
                # 最后一期：票息 + 到期赎回价
                amount = coupon + mcp
            else:
                amount = coupon
            cf_dates.append(dt)
            cf_amounts.append(amount)

        if cf_dates:
            cache[wind_code] = {'dates': cf_dates, 'amounts': cf_amounts}

    # 保存缓存
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

    return {c: cache[c] for c in codes if c in cache}


# ── YTM ─────────────────────────────────────────────────────

def fetch_ytm_jqdata(codes, start, end, cache_path='cb_cashflow_calendar_jq.pkl'):
    """
    计算可转债 YTM，返回 DataFrame(index=dates, columns=codes)，单位%。
    """
    df_close = _fetch_close_jq(codes, pd.to_datetime(start), pd.to_datetime(end))
    if df_close is None or df_close.empty:
        return None

    cf_cal = build_cashflow_calendar_jq(codes, cache_path)

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


# ── 隐含波动率 ──────────────────────────────────────────────

def fetch_impliedvol_jqdata(codes, start, end):
    """
    计算可转债隐含波动率，返回 DataFrame(index=dates, columns=codes)，单位%。
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    df_close = _fetch_close_jq(codes, start_dt, end_dt)
    df_conv = _fetch_convvalue_jq(codes, start_dt, end_dt)
    if df_close is None or df_conv is None:
        return None

    trade_dates_str = [d.strftime('%Y%m%d') for d in df_close.index]
    df_ptm = fetch_ptm_jqdata(codes, trade_dates_str)

    mcps = _get_maturity_call_prices_jq(codes)
    mcp_arr = np.array([mcps.get(c, 100.0) for c in codes], dtype=float)

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


# ── 剩余规模 ────────────────────────────────────────────────

def fetch_outstanding_jqdata(codes, trade_dates):
    """
    获取可转债剩余规模。
    jqdata 没有直接的剩余规模字段，走回退链。
    """
    from cb_with_any_api import list_available_apis
    idx = pd.to_datetime(trade_dates)

    for source in FALLBACK_CHAIN:
        if source not in list_available_apis:
            continue
        try:
            if source == 'wind':
                import wind_reader
                return wind_reader.fetch_wind(codes, 'clause_conversion2_bondlot', idx[0], idx[-1])
            elif source == 'tushare':
                import tushare_reader
                td_str = [d.strftime('%Y%m%d') for d in idx]
                return tushare_reader.fetch_outstanding_tushare(codes, td_str)
            elif source == 'akshare':
                import akshare_reader
                td_str = [d.strftime('%Y%m%d') for d in idx]
                return akshare_reader.fetch_outstanding_akshare(codes, td_str)
        except Exception as e:
            print(f"回退到 {source} 获取 Outstanding 失败: {e}")
            continue

    # 全部失败，返回空 DataFrame
    return pd.DataFrame(index=idx, columns=codes, dtype=float)


# ── 面板数据 ────────────────────────────────────────────────

def fetch_panel_from_jqdata(codes):
    """
    从 jqdata 获取面板（静态）数据，返回 DataFrame(index=codes)。
    缺失字段走回退链。
    """
    panel_cols = [
        'name', 'creditrating', 'industry',
        'redeem_start', 'redeem_span', 'redeem_maxspan', 'redeem_trigger',
        'putback_start', 'putback_span', 'putback_maxspan', 'putback_trigger',
        'reset_span', 'reset_maxspan', 'reset_trigger',
        'maturity_price', 'underlyingcode', 'stock_code'
    ]
    result = pd.DataFrame(index=codes, columns=panel_cols)

    cache = _load_basic_cache_jq()

    for code in codes:
        if code not in cache:
            continue

        info = cache[code]
        result.loc[code, 'name'] = info.get('name')
        result.loc[code, 'maturity_price'] = info.get('maturity_call_price')

        stk_code = info.get('stk_code')
        if stk_code:
            result.loc[code, 'underlyingcode'] = stk_code
            result.loc[code, 'stock_code'] = stk_code

            # 尝试获取行业
            try:
                industry = jq.get_industry(stk_code)
                if industry and stk_code in industry:
                    ind_info = industry[stk_code]
                    if 'sw_l1' in ind_info:
                        result.loc[code, 'industry'] = ind_info['sw_l1'].get('industry_name')
            except Exception:
                pass

    # 回退链补充缺失数据
    missing_name = result['name'].isna()
    if missing_name.any():
        missing_codes = list(result.index[missing_name])
        from cb_with_any_api import list_available_apis
        for source in FALLBACK_CHAIN:
            if source not in list_available_apis:
                continue
            try:
                if source == 'tushare':
                    import tushare_reader
                    df_fb = tushare_reader.fetch_panel_from_tushare(missing_codes)
                elif source == 'akshare':
                    import akshare_reader
                    df_fb = akshare_reader.fetch_panel_from_akshare(missing_codes)
                else:
                    continue

                if df_fb is not None:
                    for col in result.columns:
                        if col in df_fb.columns:
                            for c in missing_codes:
                                if c in df_fb.index and pd.notna(df_fb.loc[c, col]) and pd.isna(result.loc[c, col]):
                                    result.loc[c, col] = df_fb.loc[c, col]
                break
            except Exception as e:
                print(f"回退到 {source} 获取面板数据失败: {e}")
                continue

    return result


# ── 增量更新入口 ────────────────────────────────────────────

def update_from_df_jqdata(df, end, field):
    """
    从 jqdata 更新数据到现有 DataFrame，逻辑与 tushare_reader.update_from_df_tushare 一致。
    """
    codes = list(df.columns)
    last_date = pd.to_datetime(df.index[-1])
    end_date = pd.to_datetime(end)

    # 用 jqdata 交易日历判断是否需要更新
    trade_days = jq.get_trade_days(start_date=last_date, end_date=end_date)
    trade_dates = [pd.to_datetime(d).strftime('%Y%m%d') for d in trade_days]

    if len(trade_dates) > 1:
        new_start = trade_dates[1]
        new_end = trade_dates[-1]

        # 回退字段直接走回退链
        if field in ('strbvalue', 'strbpremiumratio'):
            return _update_with_fallback(df, end, field)

        if field == 'ytm_cb':
            df_new = fetch_ytm_jqdata(codes, new_start, new_end)
        elif field == 'ptmyear':
            df_new = fetch_ptm_jqdata(codes, trade_dates[1:])
        elif field == 'impliedvol':
            df_new = fetch_impliedvol_jqdata(codes, new_start, new_end)
        elif field == 'clause_conversion2_bondlot':
            df_new = fetch_outstanding_jqdata(codes, trade_dates[1:])
        else:
            df_new = fetch_jqdata(codes, field, new_start, new_end)

        if df_new is not None and not df_new.empty:
            df_new.index = pd.to_datetime(df_new.index)
            df = pd.concat([df, df_new])
            return df
        else:
            print(f"{field} 从 jqdata 获取新数据为空")
            return df
    else:
        print(f"{field} 不用更新")
        return df


# ── 代码列表 ────────────────────────────────────────────────

def getCodeList_jqdata():
    """获取全市场可转债代码列表（Wind 格式）"""
    df_sec = jq.get_all_securities('conbond')
    if df_sec is None or df_sec.empty:
        return []

    return [_from_jq_code(c) for c in df_sec.index]
