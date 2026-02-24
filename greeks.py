# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm
import pandas as pd
import cb_with_any_api as cb

def bsCB(s, x, t, vol, r) -> np.array:
    '''转债专用,期间票息不重要直接忽略'''

    d1_ = d1(s, x, t, vol, r)
    d2_ = d2(s, x, t, vol, r)
    ret = s * norm.cdf(d1_) + x * np.exp(-r * t) * (1 - norm.cdf(d2_))
    ret[t <= 0] = np.maximum(s, x)[t <= 0]
    
    return ret

def d1(s, x, t, vol, r):    
    # 辅助函数
    return (np.log(s / x) + (r + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))

def d2(s, x, t, vol, r):
    # 辅助函数
    return (np.log(s / x) + (r - 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))

def d3(s, k, t, vol):
    # 辅助函数
    return (np.log(s) - np.log(k)) / (vol * t ** 0.5)    


def delta(s, x, t, vol, r):
    # 标准版，没有考虑退市线
    d1_ = d1(s, x, t, vol, r)
    
    return norm.cdf(d1_)

def theta(s, x, t, vol, r, option='cb'):
    # 标准版，没有考虑退市线
    '''option如果选择cb，则会考虑票息
    如果选择stock，则按照普通欧式期权走
    '''
    d1_, d2_ = d1(s, x, t, vol, r), d2(s, x, t, vol, r)
    if option == 'cb':
        theta_ = -0.5 * s * vol * norm.pdf(d1_) / (t ** 0.5) + \
                r * x * np.exp(-r * t) * (1 - norm.cdf(d2_))
    elif option == 'stock':
        theta_ = -0.5 * s * vol * norm.pdf(d1_) / (t ** 0.5) + \
                r * x * np.exp(-r * t) * (1 - norm.cdf(d2_))
    else:
        raise ValueError('option must be cb or stock')
    
    return theta_

def gamma(s, x, t, vol, r):
    # 标准版，没有考虑退市线
    d1_ = d1(s, x, t, vol, r)
    gamma_ = norm.pdf(d1_) / (s * vol * t ** 0.5)
    return gamma_

def vega(s, x, t, vol, r):
    # 标准版，没有考虑退市线
    d1_ = d1(s, x, t, vol, r)
    return np.maximum(s * norm.pdf(d1_) * t ** 0.5, .01)

def impliedVol_批量(close: [np.array, pd.Series],
                  conv: [np.array, pd.Series],
                  t: [np.array, pd.Series],
                  maturitycallprice: [np.array, pd.Series],
                  rf: float=0.03, iterMax: int=500):
    '''
    批量计算隐含波动率，返回绝对数字，不是百分比，上限2.0，下限.001        

    Parameters
    ----------
    close : [np.array, pd.Series]
        最好是np.array, pd.Series会被强制转换，但请确保几个参数的codes顺序一致.
    conv : [np.array, pd.Series]
        同上
    t : [np.array, pd.Series]
        
    maturitycallprice : [np.array, pd.Series]
        
    rf : float, optional
        The default is 0.03.
    iterMax : int, optional
        The default is 500.

    Returns
    -------
    vol : np.array
        隐含波动率

    '''
    if isinstance(close, pd.Series):
        close = close.values

    if isinstance(conv, pd.Series):
        conv = conv.values
    
    if isinstance(t, pd.Series):
        t = t.values
    
    if isinstance(maturitycallprice, pd.Series):
        maturitycallprice = maturitycallprice.values
    
    # 给一个初始试探点
    vol = np.array([0.4] * close.shape[0], dtype=float)
    # 负溢价的直接给一个占位符
    vol[close < np.maximum(conv, maturitycallprice * np.exp(-rf * t))] = 0.001
    closeHat = bsCB(conv, maturitycallprice, t, vol, rf)
    # 二分法
    vmax, vmin = np.array([2.] * close.shape[0]), np.array([.001] * close.shape[0])
    
    while (np.abs((close - closeHat)).max() > .1) and (iterMax > 0):
        iterMax -= 1
        vmax[(close - closeHat) < 0] = vol[(close - closeHat) < 0]
        vmin[(close - closeHat) > 0] = vol[(close - closeHat) > 0]
        vol = 0.5 * (vmax + vmin)
        
        closeHat = bsCB(conv, maturitycallprice, t, vol, rf)
    
    return vol

def delta_from_cb(obj: cb.cb_data, date=None, codes=None, rf=0.03, vol=None) -> np.array:
    '''
    从cb.cb_data出发得到delta，标准版，不考虑退市

    Parameters
    ----------
    obj : cb.cb_data
        DESCRIPTION.
    date : TYPE, optional
        The default is None.
    codes : TYPE, optional
        The default is None.
    rf : TYPE, optional
        The default is 0.03.
    vol : TYPE, optional
        The default is None.

    Returns
    -------
    np.array       

    '''
    if date is None:
        date = obj.date
    
    if codes is None:
        codes = obj.selByAmt(date)
        
    conv = obj.ConvV.loc[date, codes]
    mtcp = obj.panel.loc[codes, 'maturity_price']
    t = obj.Ptm.loc[date, codes]
    close = obj.Close.loc[date, codes]
    if vol is None:
        vol = impliedVol_批量(close, conv, t, mtcp)
    
    d1_ = d1(conv, mtcp, t, vol, rf)
    return norm.cdf(d1_)


def theta_from_cb(obj, date=None, codes=None, rf=0.03, vol=None):
    '''
    从cb.cb_data出发得到theta，标准版，不考虑退市

    Parameters
    ----------
    obj : cb.cb_data
        DESCRIPTION.
    date : TYPE, optional
        The default is None.
    codes : TYPE, optional
        The default is None.
    rf : TYPE, optional
        The default is 0.03.
    vol : TYPE, optional
        The default is None.

    Returns
    -------
    np.array       

    '''    
    if date is None:
        date = obj.date
    
    if codes is None:
        codes = obj.selByAmt(date)    
    
    conv = obj.ConvV.loc[date, codes]
    mtcp = obj.panel.loc[codes, 'maturity_price']
    t = obj.Ptm.loc[date, codes]
    close = obj.Close.loc[date, codes]

    if vol is None:
        vol = impliedVol_批量(close, conv, t, mtcp)

    theta_ = theta(conv, mtcp, t, vol, rf)
    
    return theta_

def gamma_from_cb(obj, date=None, codes=None, rf=0.03, vol=None):
    '''
    从cb.cb_data出发得到gamma，标准版，不考虑退市

    Parameters
    ----------
    obj : cb.cb_data
        DESCRIPTION.
    date : TYPE, optional
        The default is None.
    codes : TYPE, optional
        The default is None.
    rf : TYPE, optional
        The default is 0.03.
    vol : TYPE, optional
        The default is None.

    Returns
    -------
    np.array       

    '''    
    if date is None:
        date = obj.date
    
    if codes is None:
        codes = obj.selByAmt(date)    
    
    conv = obj.ConvV.loc[date, codes]
    mtcp = obj.panel.loc[codes, 'maturity_price']
    t = obj.Ptm.loc[date, codes]
    close = obj.Close.loc[date, codes]

    if vol is None:
        vol = impliedVol_批量(close, conv, t, mtcp)
        
    gamma_ = gamma(conv, mtcp, t, vol, rf)
    
    return gamma_

def vega_from_cb(obj, date=None, codes=None, rf=0.03, vol=None):
    '''
    从cb.cb_data出发得到vega，标准版，不考虑退市

    Parameters
    ----------
    obj : cb.cb_data
        DESCRIPTION.
    date : TYPE, optional
        The default is None.
    codes : TYPE, optional
        The default is None.
    rf : TYPE, optional
        The default is 0.03.
    vol : TYPE, optional
        The default is None.

    Returns
    -------
    np.array       

    '''        
    if date is None:
        date = obj.date
    
    if codes is None:
        codes = obj.selByAmt(date)    
    
    conv = obj.ConvV.loc[date, codes]
    mtcp = obj.panel.loc[codes, 'maturity_price']
    t = obj.Ptm.loc[date, codes]
    close = obj.Close.loc[date, codes]

    if vol is None:
        vol = impliedVol_批量(close, conv, t, mtcp)


    vega_ = vega(conv, mtcp, t, vol, rf)
    
    return vega_


def impliedVol_from_cb(obj, date=None, codes=None, rf=.03, iterMax=500):
    '''
    从cb.cb_data出发得到impliedVol，标准版，不考虑退市

    Parameters
    ----------
    obj : cb.cb_data
        DESCRIPTION.
    date : TYPE, optional
        The default is None.
    codes : TYPE, optional
        The default is None.
    rf : TYPE, optional
        The default is 0.03.

    Returns
    -------
    np.array       

    '''
    if date is None:
        date = obj.date
    if codes is None:
        codes = obj.selByAmt(date)
    
    close = obj.Close.loc[date, codes]
    conv = obj.ConvV.loc[date, codes]
    t = obj.Ptm.loc[date, codes]
    maturitycallprice = obj.panel.maturity_price[codes]
    
    return impliedVol_批量(close, conv, t, maturitycallprice, rf, iterMax)


def price_from_obj(obj, date, codes, vol=None, r=None):
    s = obj.ConvV.loc[date, codes]
    x = obj.panel.maturity_price[codes].fillna(100)
    t = obj.Ptm.loc[date, codes]
    if vol is None:
        vol = obj.ImpliedVol.loc[date, codes] / 100.0
    if r is None:
        r = 0.03
    
    return pd.Series(bsCB(s, x, t, vol, r), index=codes)
