import tushare as ts
import pandas as pd
import numpy as np
from scipy.stats import norm
import time
import os
import pickle
import const

# 初始化 tushare pro API
ts.set_token(const.ts_token)
pro = ts.pro_api()

# Wind 字段名 → (Tushare cb_daily 字段名, 单位缩放系数)
# 缩放系数: Tushare值 * scale = Wind 值
FIELD_MAP = {
    'amt':                ('amount',       10000),  # Tushare万元 → Wind元
    'close':              ('close',        1),
    'convpremiumratio':   ('cb_over_rate', 1),
    'strbvalue':          ('bond_value',   1),
    'convvalue':          ('cb_value',     1),
    'strbpremiumratio':   ('bond_over_rate', 1),
}


# cb_basic 全量内存缓存，首次调用时加载
# {ts_code: {'maturity_date': datetime, 'par': float, 'add_rate': float}}
_basic_cache = None


def _load_basic_cache():
    """首次调用全量拉取 cb_basic 并缓存到内存（maturity_date, par, add_rate）。"""
    global _basic_cache
    if _basic_cache is None:
        df = pro.cb_basic(fields='ts_code,maturity_date,par,add_rate')
        _basic_cache = {}
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                _basic_cache[row['ts_code']] = {
                    'maturity_date': pd.to_datetime(row['maturity_date']) if pd.notna(row['maturity_date']) else None,
                    'par': row['par'] if pd.notna(row.get('par')) else 100.0,
                    'add_rate': row['add_rate'] if pd.notna(row.get('add_rate')) else 0.0,
                }
    return _basic_cache


def _get_maturity_dates(codes):
    """获取到期日，复用 _basic_cache。"""
    cache = _load_basic_cache()
    return {c: cache[c]['maturity_date'] if c in cache else None for c in codes}


def _get_maturity_call_prices(codes):
    """获取到期赎回价 = par * (1 + add_rate/100)，复用 _basic_cache。"""
    cache = _load_basic_cache()
    result = {}
    for c in codes:
        if c in cache:
            result[c] = cache[c]['par'] * (1 + cache[c]['add_rate'] / 100)
        else:
            result[c] = np.nan
    return result


def fetch_ptm_tushare(codes, trade_dates):
    """
    计算可转债剩余期限（年）。

    Parameters
    ----------
    codes : list
        债券代码列表
    trade_dates : list
        交易日列表（由调用方提供，避免重复调用 trade_cal）
    """
    mat_dates = _get_maturity_dates(codes)

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


def build_cashflow_calendar(codes, cache_path='cb_cashflow_calendar.pkl'):
    """
    构建并缓存现金流日历。

    Parameters
    ----------
    codes : list
        债券代码列表，如 ['110092.SH', '123099.SZ']
    cache_path : str
        缓存文件路径

    Returns
    -------
    dict : {ts_code: {'dates': [datetime,...], 'amounts': [float,...]}}
    """
    # 加载已有缓存
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    # 找出缓存中没有的券
    new_codes = [c for c in codes if c not in cache]
    if not new_codes:
        return {c: cache[c] for c in codes if c in cache}

    # 复用全量 basic 缓存（避免逐批调 cb_basic，该接口不支持逗号分隔）
    basic_cache = _load_basic_cache()
    basic_map = {c: basic_cache[c] for c in new_codes if c in basic_cache}

    # 批量获取 cb_rate（每次最多50个）
    rate_pieces = []
    for i in range(0, len(new_codes), 50):
        batch = new_codes[i:i+50]
        ts_codes_str = ','.join(batch)
        df_rate = pro.cb_rate(ts_code=ts_codes_str,
                              fields='ts_code,rate_end_date,coupon_rate')
        if df_rate is not None and not df_rate.empty:
            rate_pieces.append(df_rate)
        if i + 50 < len(new_codes):
            time.sleep(0.3)

    if rate_pieces:
        df_rate_all = pd.concat(rate_pieces, ignore_index=True)
    else:
        df_rate_all = pd.DataFrame(columns=['ts_code', 'rate_end_date', 'coupon_rate'])

    # 对每只新券构建现金流
    for code in new_codes:
        info = basic_map.get(code)
        if info is None:
            continue

        par = info.get('par', 100) or 100
        add_rate = info.get('add_rate', 0) or 0

        # 该券的付息记录，按日期排序
        rate_rows = df_rate_all[df_rate_all['ts_code'] == code].copy()
        if rate_rows.empty:
            continue

        rate_rows['rate_end_date'] = pd.to_datetime(rate_rows['rate_end_date'])
        rate_rows = rate_rows.sort_values('rate_end_date')

        cf_dates = []
        cf_amounts = []
        n = len(rate_rows)
        for j, (_, row) in enumerate(rate_rows.iterrows()):
            dt = row['rate_end_date']
            coupon = par * row['coupon_rate'] / 100
            if j == n - 1:
                # 最后一期：票息 + 本金 + 补偿利息
                amount = coupon + par * (1 + add_rate / 100)
            else:
                amount = coupon
            cf_dates.append(dt)
            cf_amounts.append(amount)

        cache[code] = {'dates': cf_dates, 'amounts': cf_amounts}

    # 保存缓存
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

    return {c: cache[c] for c in codes if c in cache}


def _newton_ytm(cf_dates, cf_amounts, prices, val_dates, max_iter=50, tol=1e-8):
    """
    单券跨日期矢量化 Newton 法计算 YTM。

    Parameters
    ----------
    cf_dates : np.array, shape (K,), datetime64
    cf_amounts : np.array, shape (K,), float
    prices : np.array, shape (T,), float — 收盘价
    val_dates : np.array, shape (T,), datetime64
    max_iter : int
    tol : float

    Returns
    -------
    np.array, shape (T,) — YTM 百分比
    """
    K = len(cf_dates)
    T = len(prices)

    # year_frac: (T, K) 每个交易日到每个现金流日期的年分数
    # datetime64 差值转天数再除365
    delta = (cf_dates[np.newaxis, :] - val_dates[:, np.newaxis]).astype('timedelta64[D]').astype(np.float64)
    year_frac = delta / 365.0  # (T, K)

    # mask: 只保留估值日之后的现金流
    mask = (delta > 0).astype(np.float64)  # (T, K)

    # 对已到期券（没有剩余现金流）直接返回 NaN
    has_cf = mask.sum(axis=1) > 0  # (T,)

    # 初始猜测
    r = np.full(T, 0.03)

    for _ in range(max_iter):
        rr = r[:, np.newaxis]  # (T, 1)
        disc = (1 + rr) ** (-year_frac)  # (T, K)
        f = (cf_amounts * mask * disc).sum(axis=1) - prices  # (T,)
        fp = (-year_frac * cf_amounts * mask * disc / (1 + rr)).sum(axis=1)  # (T,)

        # 避免除以0
        safe = np.abs(fp) > 1e-15
        delta_r = np.where(safe, f / fp, 0.0)
        r -= delta_r

        if np.all(np.abs(delta_r) < tol):
            break

    result = r * 100  # 转为百分比
    # 无剩余现金流 → NaN
    result[~has_cf] = np.nan
    return result


def fetch_ytm_tushare(codes, start, end, cache_path='cb_cashflow_calendar.pkl'):
    """
    计算可转债 YTM，返回 DataFrame(index=dates, columns=codes)，单位%。

    Parameters
    ----------
    codes : list
        债券代码列表
    start, end : str or datetime
        日期范围
    cache_path : str
        现金流日历缓存路径
    """
    # 获取收盘价矩阵
    df_close = fetch_tushare(codes, 'close', start, end)
    if df_close is None or df_close.empty:
        return None

    # 构建/加载现金流日历
    cf_cal = build_cashflow_calendar(codes, cache_path)

    val_dates = df_close.index.values.astype('datetime64[D]')  # (T,)
    result = pd.DataFrame(index=df_close.index, columns=codes, dtype=float)

    for code in codes:
        if code not in cf_cal:
            result[code] = np.nan
            continue

        cf = cf_cal[code]
        cf_dates = np.array(cf['dates'], dtype='datetime64[D]')
        cf_amounts = np.array(cf['amounts'], dtype=np.float64)

        prices = df_close[code].values.astype(np.float64)

        # 价格为 NaN 或 0 的位置
        bad_price = np.isnan(prices) | (prices <= 0)

        # 用占位价格（100）替代 bad_price，计算后再置 NaN
        prices_safe = np.where(bad_price, 100.0, prices)

        ytm = _newton_ytm(cf_dates, cf_amounts, prices_safe, val_dates)
        ytm[bad_price] = np.nan
        result[code] = ytm

    return result


def _bs_cb(s, x, t, vol, r):
    """转债 BS 定价（与 greeks.py 中 bsCB 一致）"""
    d1 = (np.log(s / x) + (r + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    ret = s * norm.cdf(d1) + x * np.exp(-r * t) * (1 - norm.cdf(d2))
    expired = t <= 0
    if np.any(expired):
        ret[expired] = np.maximum(s, x)[expired]
    return ret


def _implied_vol_batch(close, conv, t, maturitycallprice, rf=0.03, iterMax=500):
    """二分法批量计算隐含波动率（与 greeks.py 中 impliedVol_批量 一致）。返回绝对数字。"""
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


def fetch_impliedvol_tushare(codes, start, end):
    """
    计算可转债隐含波动率，返回 DataFrame(index=dates, columns=codes)，单位%。
    需要 close、convvalue（转股价值）、ptm、到期赎回价。
    """
    df_close = fetch_tushare(codes, 'close', start, end)
    df_conv = fetch_tushare(codes, 'convvalue', start, end)
    if df_close is None or df_conv is None:
        return None

    trade_dates_str = [d.strftime('%Y%m%d') for d in df_close.index]
    df_ptm = fetch_ptm_tushare(codes, trade_dates_str)

    mcps = _get_maturity_call_prices(codes)
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


def fetch_panel_from_tushare(codes):
    """
    从 tushare 获取面板（静态）数据，返回 DataFrame(index=codes)。
    列与 Wind 版 panel 对齐，Tushare 无法提供的条款字段留 NaN。

    可获取字段: name, industry, maturity_price, underlyingcode, stock_code
    不可获取字段: creditrating, redeem_*, putback_*, reset_*
    """
    basic_cache = _load_basic_cache()

    # 全量拿 cb_basic 的 bond_short_name, stk_code（不在 _basic_cache 里，单独查一次全量）
    df_basic = pro.cb_basic(fields='ts_code,bond_short_name,stk_code')
    basic_name_map = {}
    basic_stk_map = {}
    if df_basic is not None and not df_basic.empty:
        for _, row in df_basic.iterrows():
            basic_name_map[row['ts_code']] = row['bond_short_name']
            basic_stk_map[row['ts_code']] = row['stk_code']

    # 全量拿 stock_basic 的 industry
    df_stk = pro.stock_basic(fields='ts_code,industry')
    industry_map = {}
    if df_stk is not None and not df_stk.empty:
        industry_map = df_stk.set_index('ts_code')['industry'].to_dict()

    # 构建 panel DataFrame
    panel_cols = [
        'name', 'creditrating', 'industry',
        'redeem_start', 'redeem_span', 'redeem_maxspan', 'redeem_trigger',
        'putback_start', 'putback_span', 'putback_maxspan', 'putback_trigger',
        'reset_span', 'reset_maxspan', 'reset_trigger',
        'maturity_price', 'underlyingcode', 'stock_code'
    ]
    result = pd.DataFrame(index=codes, columns=panel_cols)

    for code in codes:
        result.loc[code, 'name'] = basic_name_map.get(code)

        stk = basic_stk_map.get(code)
        result.loc[code, 'underlyingcode'] = stk
        result.loc[code, 'stock_code'] = stk

        if stk and stk in industry_map:
            result.loc[code, 'industry'] = industry_map[stk]

        if code in basic_cache:
            info = basic_cache[code]
            par = info['par']
            add_rate = info['add_rate']
            result.loc[code, 'maturity_price'] = par * (1 + add_rate / 100)

    return result


def fetch_outstanding_tushare(codes, trade_dates):
    """
    获取可转债剩余规模（元），返回 DataFrame(index=dates, columns=codes)。
    从 cb_share 拿变动记录，reindex 到交易日后 ffill（与 Wind 行为一致）。

    Parameters
    ----------
    codes : list
        债券代码列表
    trade_dates : list
        交易日列表（由调用方提供，避免重复调用 trade_cal）
    """
    idx = pd.to_datetime(trade_dates)
    result = pd.DataFrame(index=idx, columns=codes, dtype=float)

    # 批量查 cb_share（支持逗号分隔）
    share_pieces = []
    for i in range(0, len(codes), 50):
        batch = codes[i:i+50]
        try:
            df = pro.cb_share(ts_code=','.join(batch),
                              fields='ts_code,end_date,remain_size')
            if df is not None and not df.empty:
                share_pieces.append(df)
        except Exception as e:
            print(f"cb_share 查询失败: {e}")
        if i + 50 < len(codes):
            time.sleep(0.3)

    if not share_pieces:
        return result

    df_share = pd.concat(share_pieces, ignore_index=True)
    df_share['end_date'] = pd.to_datetime(df_share['end_date'])
    # remain_size 本身就是元，与 Wind 单位一致，无需转换

    # 从 _basic_cache 拿 issue_size 作为转股期前的初始值
    basic_cache = _load_basic_cache()

    for code in codes:
        sub = df_share[df_share['ts_code'] == code].copy()
        if sub.empty:
            # 没有变动记录，用发行规模填充
            if code in basic_cache:
                # cb_basic 的 issue_size 不在 _basic_cache 里，单独查
                pass
            result[code] = np.nan
            continue

        # 按日期去重（同日取最新），构建稀疏 Series
        sub = sub.sort_values('end_date').drop_duplicates('end_date', keep='last')
        srs = sub.set_index('end_date')['remain_size']

        # reindex 到交易日，ffill
        srs = srs.reindex(idx, method='ffill')
        result[code] = srs

    return result


def fetch_tushare(codes, field, start, end):
    '''
    从 tushare 获取可转债时间序列数据
    返回 DataFrame，index=日期, columns=债券代码，与 fetch_wind 格式一致

    按 trade_date 逐日查询，每次返回当日所有转债，比逐券查询高效得多。

    Parameters
    ----------
    codes : list
        债券代码列表，如 ['113556.SH', '128114.SZ']
    field : str
        Wind 字段名，如 'close', 'amt', 'convpremiumratio'
    start : str or datetime
        开始日期
    end : str or datetime
        结束日期
    '''
    # 查找 tushare 对应字段和缩放系数
    mapping = FIELD_MAP.get(field)
    if mapping is None:
        print(f"警告: 字段 '{field}' 在 tushare cb_daily 中不可用，跳过")
        return None

    ts_field, scale = mapping

    # 获取日期范围内的交易日
    start_str = pd.to_datetime(start).strftime('%Y%m%d')
    end_str = pd.to_datetime(end).strftime('%Y%m%d')
    cal = pro.trade_cal(exchange='SSE', start_date=start_str, end_date=end_str, is_open='1')
    trade_dates = sorted(cal['cal_date'].tolist())

    if not trade_dates:
        return None

    codes_set = set(codes)
    pieces = []
    for i, td in enumerate(trade_dates):
        try:
            df = pro.cb_daily(
                trade_date=td,
                fields=f'ts_code,trade_date,{ts_field}'
            )
            if df is not None and not df.empty:
                # 只保留目标券
                df = df[df['ts_code'].isin(codes_set)]
                if not df.empty:
                    row = df.set_index('ts_code')[ts_field]
                    row.name = td
                    pieces.append(row)
        except Exception as e:
            print(f"获取 {td} 失败: {e}")

        # tushare 频率限制
        if (i + 1) % 50 == 0:
            time.sleep(1)

    if not pieces:
        return None

    result = pd.DataFrame(pieces)
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    # 按传入的 codes 顺序排列列，缺失的券自动为 NaN
    result = result.reindex(columns=codes)

    if scale != 1:
        result = result * scale

    return result


def update_from_df_tushare(df, end, field):
    """
    从 tushare 更新数据到现有 DataFrame，逻辑与 wind_reader.update_from_df 一致
    """
    codes = list(df.columns)
    last_date = pd.to_datetime(df.index[-1])
    end_date = pd.to_datetime(end)

    # 用 tushare 交易日历判断是否需要更新
    cal = pro.trade_cal(
        exchange='SSE',
        start_date=last_date.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d'),
        is_open='1'
    )
    trade_dates = sorted(cal['cal_date'].tolist())

    if len(trade_dates) > 1:
        # 从 last_date 的下一个交易日开始获取
        new_start = trade_dates[1]
        new_end = trade_dates[-1]

        if field == 'ytm_cb':
            df_new = fetch_ytm_tushare(codes, new_start, new_end)
        elif field == 'ptmyear':
            df_new = fetch_ptm_tushare(codes, trade_dates[1:])
        elif field == 'impliedvol':
            df_new = fetch_impliedvol_tushare(codes, new_start, new_end)
        elif field == 'clause_conversion2_bondlot':
            df_new = fetch_outstanding_tushare(codes, trade_dates[1:])
        else:
            df_new = fetch_tushare(codes, field, new_start, new_end)

        if df_new is not None and not df_new.empty:
            df_new.index = pd.to_datetime(df_new.index)
            df = pd.concat([df, df_new])
            return df
        else:
            print(f"{field} 从 tushare 获取新数据为空")
            return df
    else:
        print(f"{field} 不用更新")
        return df
