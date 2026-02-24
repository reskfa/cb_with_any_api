import pandas as pd
import const

list_available_apis = []

if const.wind_available:
    try:
        import wind_reader
        list_available_apis.append('wind')
    except:
        print('Wind API 未正确安装')

if const.ts_token:
    import tushare as ts
    ts.set_token(const.ts_token)
    pro = ts.pro_api()
    list_available_apis.append('tushare')

if const.jqdata_username and const.jqdata_password:
    import jqdatasdk as jq
    jq.auth(const.jqdata_username, const.jqdata_password)
    list_available_apis.append('jqdata')


class cb_data(object):
    
    lstSpecial=['100016.SH','100096.SH','100567.SH','110009.SH','110010.SH',\
        '110036.SH','110037.SH','110219.SH','110232.SH','110317.SH',\
        '110325.SH','110398.SH','110418.SH','110598.SH','125024.SZ',\
        '125528.SZ','125629.SZ','125630.SZ','125729.SZ','125822.SZ',\
        '125898.SZ','125930.SZ','125937.SZ','125959.SZ','125960.SZ','125302.SZ']

    def __init__(self, prefix="", file_type="pkl"):
        '''
        初始化 cb_data 类
        prefix: 数据文件的前缀，默认空字符串
        file_type: 数据文件的类型，默认 pkl，支持 csv, xlsx 格式
        '''
        self.DB = {} # DB为字典，准备装载各维度的数据
        self.prefix = prefix
        self.file_type = file_type
        
        
    def loadData(self, prefix):
        self.dfParams = pd.read_excel("参数.xlsx", index_col=0)
        for k, v in self.dfParams.iterrows():
            if self.file_type == "pkl":
                df = pd.read_pickle(prefix + k + "." + self.file_type, index_col=0)
            elif self.file_type == "csv":
                df = pd.read_csv(prefix + k + "." + self.file_type, index_col=0)
            elif self.file_type == "xlsx":  
                df = pd.read_excel(prefix + k + "." + self.file_type, index_col=0)
            else:
                raise ValueError(f"不支持的文件类型: {self.file_type}")
            df.index = pd.to_datetime(df.index)
            self.DB[k] = df
        
        self.panel = pd.read_excel("静态数据.xlsx", index_col=0)
    
    def __getitem__(self, key):
        return self.DB[key] if key in self.DB.keys() else None
    
    
    def __getattr__(self, key):
        if key in self.DB.keys():
            return self.DB[key]
        else:
            raise AttributeError

    
    @property
    def date(self):
        '''返回数据的最新日期'''
        return self.DB["Amt"].index[-1]
    
    
    
    @property
    def codes(self):
        '''返回所有代码'''
        return list(self.DB["Amt"].columns)
    
    
    @property
    def codes_active(self):
        '''返回最新交易日有交易的代码'''
        srs = self.DB["Amt"].loc[self.date, self.codes]
        return list(srs[srs > 0].index)    
    
    
    def update(self, end, method=None):

        if method is None:
            method = list_available_apis[0]

        if method == 'wind':    
            for k, v in self.dfParams.iterrows():
                df = self.DB[k]
                df = wind_reader.update_from_df(df, end, v["Wind"])
                self.DB[k] = df
                print(f'{k} 更新已完成')
                
            self.DB["Outstanding"] = self.Outstanding.reindex_like(self.Amt).fillna(method="pad")
            for k, v in self.DB.items():
                self.DB[k] = v.reindex_like(self.Amt)
        elif method == 'tushare':
            # 导入 tushare 相关函数
            import tushare_reader
            
            for k, v in self.dfParams.iterrows():
                df = self.DB[k]
                df = tushare_reader.update_from_df_tushare(df, end, v["Wind"])
                self.DB[k] = df
                print(f'{k} 从 tushare 更新已完成')
                
            self.DB["Outstanding"] = self.Outstanding.reindex_like(self.Amt).fillna(method="pad")
            for k, v in self.DB.items():
                self.DB[k] = v.reindex_like(self.Amt)
        elif method == 'jqdata':
            # jqdata 的实现（占位符）
            print("jqdata 更新功能暂未完全实现")
            # 这里可以添加 jqdata 的具体实现
            for k, v in self.dfParams.iterrows():
                # 占位实现，实际需要根据 jqdata API 来实现
                print(f'{k} 从 jqdata 更新（暂未实现）')
        else:
            raise ValueError(f"不支持的 API 类型: {method}")
    def insertNewKey(self, new_codes, method=None):

        if method is None:
            method = list_available_apis[0]
        
        for key, value in self.DB.items():
           
            diff = list(set(new_codes) - set(value.keys()))
                                                                                                
            if diff:                                        
                field = self.dfParams.loc[key, '字段(Wind)']
        
                start = self.DB[key].index[0] 
                end = self.DB[key].index[-1]
                
                if method == "wind":
                    # 使用 wind_reader.fetch_wind 获取数据
                    df = wind_reader.fetch_wind(diff, field, start, end)
                    
                    # 确保 df 不为 None
                    if df is not None:
                        # 将新数据与现有数据合并
                        value = value.join(df)
                        self.DB[key] = value
                        
        self.updatePanelData(new_codes)
    
    
    def readPanel(self, codes=None, method=None):

        if codes is None : codes = self.codes
        if method is None:
            method = list_available_apis[0]

        if method == 'wind':
            date = pd.to_datetime(self.date).strftime("%Y%m%d")
            dfParams = pd.read_excel("静态参数.xlsx", index_col=0)
            df = wind_reader.fetch_wss(codes, ",".join(dfParams["字段(Wind)"]), date)
            if df is not None:
                df.columns = list(dfParams.index)
            return df
        elif method == 'tushare':
            import tushare_reader
            return tushare_reader.fetch_panel_from_tushare(codes)
        else:
            raise ValueError(f"不支持的 API 类型: {method}")
    
    
    def updatePanelData(self, new_codes=None, method=None):

        if new_codes is None: new_codes = self.codes

        diff = list(set(new_codes) - set(self.panel.index))

        if diff:
            dfNew = self.readPanel(diff, method=method)
            self.panel = pd.concat([self.panel, dfNew])
    
    
    @property
    def matTrading(self):
        return self["Amt"].applymap(lambda x: 1 if x > 0 else np.nan)
    
    
    @property
    def matNormal(self):
        
        matTurn = self.DB["Amt"] * 10000.0 / self.DB["Outstanding"] / self.DB["Close"]
        
        matEx = (matTurn.applymap(lambda x: 1 if x > 100 else np.nan) * \
             self.DB["Close"].applymap(lambda x: 1 if x > 135 else np.nan) * \
             self.DB["ConvPrem"].applymap(lambda x: 1 if x >35 else np.nan)).applymap(lambda x: 1 if x != 1 else np.nan)
        
        return self.matTrading * matEx
    
    
    def output(self, prefix=""):
        for k , v in self.DB.items():
            v.to_csv(prefix + self.dfParams.loc[k, "文件名"])
        self.panel.to_excel("静态数据.xlsx")
    
    def selByAmt(self, date, other=None):
        t = self.DB['Amt'].loc[date] > 0
        
        if other:
            for k, v in other.items():
                t *= self.DB[k].loc[date].between(v[0], v[1])
                
        codes = list(t[t].index)
        return codes
    
    def selByAmtPeriod(self, start, end):
        
        t = self.DB['Amt'].loc[start:end].sum() > 0
        return list(t[t].index)
        

    def _excludeSpecial(self, hasEB=1):
        
        columns =  set(list(self.DB['Amt'].columns))
        columns -= set(cb_data.lstSpecial)
        columns =  list(columns)        
        
        if not hasEB:
            
            for code in columns:
                
                if code[:3] == '132' or code[:3] == '120':
                    columns.remove(code)
        
        return columns

