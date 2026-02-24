from WindPy import w
import pandas as pd

def _windChecker():
    if not w.isconnected(): w.start()

def update_from_df(df, end, field):
    
    codes = df.columns
    _windChecker()     
    dates = w.tdays(df.index[-1], end).Data[0]
    
    if len(dates) > 1:
        
        dfNew = fetch_wind(codes, field, dates[1], dates[-1])
        dfNew.index = pd.to_datetime(dfNew.index)
        df = pd.concat([df, dfNew])
        
        return df
    else:
        
        print(f"{field}不用更新")
        return df   
    
def getCodeList():
    _windChecker()
    _ , dfIssue = w.wset("cbissue", "startdate=2015-01-01;enddate=2030-12-31", usedf=True)
    
    return dfIssue.loc[dfIssue["issue_type"].apply(lambda x: True if x not in ("私募","定向") else False)].bond_code.to_list()

def fetch_panel_from_wind(codes):
    '''从 wind 获取面板数据'''
    _windChecker()
    error, df = w.wss(",".join(codes), "issueamount,carrydate,underlyingcode,underlyingname,maturitycallprice", "unit=1;", usedf=True)
    
    if error != 0:
        print(f"Wind API 错误: {error}")
        return None
    
    return df


def fetch_wss(codes, fields, date=None):
    '''从 wind 获取静态数据'''
    _windChecker()
    
    # 如果提供了日期，则添加日期参数
    options = f"tradedate={date}" if date else ""
    
    error, df = w.wss(",".join(codes), fields, options, usedf=True)
    
    if error != 0:
        print(f"Wind API 错误: {error}")
        return None
    
    return df
    

def fetch_wind(codes, field, start, end):
    '''从 wind 获取单字段时间序列数据'''
    if len(codes) >= 300:
        # 超过 300 个代码，需要分批次获取
        dfList = []
        for i in range(0, len(codes), 300):
            dfList.append(fetch_wind(codes[i:i+300], field, start, end))
        
        return pd.concat(dfList, axis=1)

    _windChecker()
    others = "rfIndex=1" if field == "impliedvol" else ""
    error_code, df = w.wsd(",".join(codes), field, start, end, others, usedf=True)
    
    if error_code != 0:
        print(f"Wind API 错误: {error_code}")
        return None

    if len(codes) == 1:
        # 因为如果只有一个 code，wind 的 columns 会变成 field
        df.columns = codes
    
    if pd.to_datetime(start) == pd.to_datetime(end):
        # 只有一天时，情况也不对，需要特殊处理
        dfRet = pd.DataFrame(columns=df.index)
        dfRet.loc[start] = df.iloc[:, 0]
        return dfRet
    else:
        return df