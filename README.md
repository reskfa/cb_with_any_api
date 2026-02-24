# CB with Any API

A股可转债统一数据接口。对接多种数据源，输出统一的 Pandas DataFrame 格式，用于可转债的统计分析、量化研究和策略开发。

## 为什么需要这个项目

A股可转债研究者面临一个现实问题：**数据源贵且不通用**。Wind 专业但昂贵，Tushare 免费但字段不全，各家 API 的字段名、单位、返回格式都不一样。换一个数据源，分析代码就要重写。

本项目将各数据源的差异封装在 reader 层，上层分析代码只需要面对统一的 `cb_data` 对象。

## 数据源支持

| 数据源 | 状态 | 说明 |
|--------|------|------|
| **Wind** | ✅ 完成 | 全字段支持，需要 Wind 终端 |
| **Tushare** | ✅ 完成 | 10 个核心字段全部对接，YTM/Ptm/ImpliedVol 自行计算 |
| **jqdata** | 🚧 本月 | 聚宽数据 |
| **akshare** | 🚧 本月 | 免费开源数据源 |

## 核心数据结构

```python
from cb_withi_any_api import cb_data

obj = cb_data()
obj.loadData("data/newt")         # 加载历史数据
obj.update("2026-02-13", method="tushare")  # 增量更新

# obj.DB — 字典，每个 key 是一个 DataFrame (index=日期, columns=券代码)
obj.DB['Close']       # 收盘价
obj.DB['Amt']         # 成交额
obj.DB['ConvV']       # 转股价值
obj.DB['ConvPrem']    # 转股溢价率
obj.DB['Strb']        # 纯债价值
obj.DB['StrbPrem']    # 纯债溢价率
obj.DB['Outstanding'] # 剩余规模
obj.DB['YTM']         # 到期收益率
obj.DB['Ptm']         # 剩余期限（年）
obj.DB['ImpliedVol']  # 隐含波动率

# obj.panel — 静态信息 DataFrame (index=券代码)
obj.panel             # 券名、正股代码、行业、赎回价、条款参数等

# 常用属性
obj.date              # 数据最新日期
obj.codes             # 全部券代码
obj.codes_active      # 当日有交易的券
obj.matNormal         # 过滤异常券后的有效矩阵
```

## 快速开始

**1. 安装依赖**

```bash
pip install pandas numpy scipy tushare
# 可选: WindPy (需 Wind 终端), jqdatasdk, akshare
```

**2. 配置 API 凭证**

编辑 `const.py`:

```python
wind_available = False            # 没有 Wind 设为 False
ts_token = 'your_tushare_token'
jqdata_username = ''
jqdata_password = ''
```

**3. 使用**

```python
from cb_withi_any_api import cb_data

obj = cb_data(file_type="csv")
obj.loadData("data/newt")
obj.update("2026-02-13", method="tushare")

# 当日活跃券的转股溢价率
prem = obj.ConvPrem.loc[obj.date, obj.codes_active]
print(prem.describe())
```

## 项目结构

```
cb_withi_any_api.py   # 核心类 cb_data，统一接口层
wind_reader.py        # Wind 数据适配器
tushare_reader.py     # Tushare 数据适配器（含 YTM/Ptm/ImpliedVol 计算引擎）
greeks.py             # 希腊字母计算（Delta/Gamma/Theta/Vega，考虑退市风险）
const.py              # API 凭证配置（不要提交到 git）
参数.xlsx              # 字段映射表（Wind/Tushare/同花顺 字段对照）
```

## Tushare 字段对接细节

直接从 `cb_daily` 获取的：Amt, Close, ConvV, ConvPrem, Strb, StrbPrem

自行计算的：
- **YTM** — 从 `cb_rate` (票息) + `cb_basic` (赎回价) 构建现金流日历，Newton 法求解，结果缓存至 `cb_cashflow_calendar.pkl`
- **Ptm** — 从 `cb_basic.maturity_date` 计算，全量缓存至内存
- **ImpliedVol** — BS 定价模型 + 二分法反推，跨日期x跨券全矢量化
- **Outstanding** — 从 `cb_share.remain_size` 获取变动记录，ffill 到交易日

已知的数据源差异（详见 `test_0224.md`）：
- Tushare `close` 为全价，Wind 为净价，差异 = 应计利息
- `Ptm` 对已公告强赎的券使用原始到期日（Wind 用强赎日）

## Roadmap

- [x] Wind 全字段对接
- [x] Tushare 全字段对接 + 自计算引擎
- [x] 静态面板数据 (panel) Tushare 支持
- [ ] jqdata 适配器
- [ ] akshare 适配器
- [ ] Agent-readable 版本 — 将转债数据库封装为 AI Agent 可调用的 Skill

## License

MIT
