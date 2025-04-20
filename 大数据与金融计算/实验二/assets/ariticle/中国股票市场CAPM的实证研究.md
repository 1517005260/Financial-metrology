# CAPM 模型及其在中国股市的实证研究笔记

## 一、CAPM 模型基本原理

CAPM (Capital Asset Pricing Model, 资本资产定价模型)由 Sharp(1964)、Lintner(1965)和 Black(1972)提出，是现代金融理论的基石之一。该模型建立在有效市场假说(EMH)的基础上，主要用于描述资产的预期收益与市场风险(用 β 系数表示)之间的关系。

### 基本假设

- 投资者是风险厌恶的，追求效用最大化
- 市场是有效的，信息是完全公开的
- 投资者可以以无风险利率进行借贷
- 所有投资者对未来收益有相同预期

### 基本公式

CAPM 模型的标准公式为：

```
E[Ri] = Rf + β(E[Rm] - Rf)
```

其中：

- E[Ri]：资产 i 的预期收益率
- Rf：无风险利率
- β：资产 i 的系统性风险系数
- E[Rm]：市场组合的预期收益率
- (E[Rm] - Rf)：市场风险溢价

β 系数计算公式：

```
β = Cov(Ri, Rm) / Var(Rm)
```

## 二、CAPM 的实证研究与异常现象

随着研究深入，学者们发现了一系列与 CAPM 理论预测不符的市场异常现象：

1. **规模效应(Size Effect)**：Banz(1981)发现，较小市值的公司往往提供更高的平均收益率
2. **价值效应(Value Effect)**：
   - Basu(1983)发现低市盈率(P/E)股票往往有较高回报
   - Stattman(1980)、Rosenberg 等(1985)发现账面市值比(BE/ME)与股票收益呈正相关
3. **动量效应**：过去表现良好的股票在短期内继续表现良好

这些异常现象挑战了 CAPM 只用 β 一个因素解释资产收益率的观点。1992 年，Fama 和 French 提出了著名的三因素模型，加入了规模因子和价值因子。

## 三、中国股市 CAPM 实证研究结果

研究对象：

- 样本：707 只 A 股股票
- 时间：1998 年 1 月-2002 年 12 月

主要发现：

1. 在中国股市，β 与股票收益率呈负相关关系，与 CAPM 预测相反
2. 公司规模(市值)、账面市值比(BE/ME)和市盈率(P/E)对股票收益率均有显著解释力
3. 回归模型显示 R² 高达 0.92，表明这些因素对股票收益率有很强的解释力

横截面回归方程：

```
RETURN = 235.59 - 12.35lnME - 9.42β - 0.03(P/E) - 31.35ln(BE/ME)
```

## 四、Python 代码实现 CAPM 模型

### 1. 基本 CAPM 模型实现

```python
import numpy as np
import pandas as pd

# 模拟数据
np.random.seed(42)
# 生成50个月的市场收益率数据
market_returns = np.random.normal(0.01, 0.05, 50)  # 均值1%，标准差5%
# 生成无风险利率数据
risk_free_rate = np.ones(50) * 0.003  # 0.3%

# 计算某股票的β系数
def calculate_beta(stock_returns, market_returns):
    # 计算协方差
    covariance = np.cov(stock_returns, market_returns)[0, 1]
    # 计算市场收益率方差
    market_variance = np.var(market_returns)
    # β系数等于协方差除以市场收益率方差
    beta = covariance / market_variance
    return beta

# 根据CAPM模型计算预期收益率
def calculate_expected_return(beta, risk_free_rate, market_return):
    risk_premium = market_return - risk_free_rate
    expected_return = risk_free_rate + beta * risk_premium
    return expected_return

# 模拟一只股票的收益率 (假设β=1.2)
stock_beta = 1.2
stock_returns = risk_free_rate + stock_beta * (market_returns - risk_free_rate) + np.random.normal(0, 0.02, 50)

# 估计β值
estimated_beta = calculate_beta(stock_returns, market_returns)
print(f"估计的β值: {estimated_beta:.4f}")

# 计算预期收益率
avg_risk_free = np.mean(risk_free_rate)
avg_market_return = np.mean(market_returns)
expected_return = calculate_expected_return(estimated_beta, avg_risk_free, avg_market_return)
print(f"预期月收益率: {expected_return:.4f}")
```

### 2. 实现多因素模型（类似中国股市研究结论）

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 模拟生成25个股票的数据
np.random.seed(42)
n_stocks = 25

# 生成因子数据
ln_market_equity = np.random.normal(10, 2, n_stocks)  # 市值取对数
beta = np.random.normal(1, 0.3, n_stocks)  # β值
pe_ratio = np.random.normal(15, 5, n_stocks)  # 市盈率
ln_book_to_market = np.random.normal(-0.5, 0.3, n_stocks)  # 账面市值比取对数

# 根据文章中的回归方程生成收益率数据
returns = 235.59 - 12.35 * ln_market_equity - 9.42 * beta - 0.03 * pe_ratio - 31.35 * ln_book_to_market
# 添加一些随机扰动
returns += np.random.normal(0, 3, n_stocks)

# 创建数据框
data = pd.DataFrame({
    'RETURN': returns,
    'lnME': ln_market_equity,
    'BETA': beta,
    'PE': pe_ratio,
    'lnBEME': ln_book_to_market
})

# 展示数据的前几行
print("数据样本:")
print(data.head())

# 计算相关系数矩阵
correlation_matrix = data.corr()
print("\n相关系数矩阵:")
print(correlation_matrix)

# 进行多元线性回归
X = data[['lnME', 'BETA', 'PE', 'lnBEME']]
y = data['RETURN']

model = LinearRegression()
model.fit(X, y)

# 打印回归结果
print("\n回归系数:")
for i, col in enumerate(X.columns):
    print(f"{col}: {model.coef_[i]:.4f}")
print(f"截距: {model.intercept_:.4f}")

# 计算R²
r_squared = model.score(X, y)
print(f"R²: {r_squared:.4f}")

# 预测示例
sample_stock = {
    'lnME': 11.5,  # 较大市值
    'BETA': 0.9,   # 较低风险
    'PE': 12,      # 较低市盈率
    'lnBEME': -0.4 # 中等账面市值比
}

sample_df = pd.DataFrame([sample_stock])
predicted_return = model.predict(sample_df)[0]
print(f"\n预测收益率: {predicted_return:.4f}")
```

## 五、简单数学运算举例

### β 系数计算示例

假设有以下 5 个月的数据：

| 月份 | 股票收益率(%) | 市场收益率(%) |
| ---- | ------------- | ------------- |
| 1    | 2.5           | 1.8           |
| 2    | -1.2          | -0.5          |
| 3    | 3.1           | 2.0           |
| 4    | -0.8          | -1.0          |
| 5    | 2.2           | 1.5           |

计算 β 系数：

1. 计算市场收益率的方差：
   Var(Rm) = Var([1.8, -0.5, 2.0, -1.0, 1.5]) = 1.89

2. 计算股票收益率与市场收益率的协方差：
   Cov(Ri, Rm) = Cov([2.5, -1.2, 3.1, -0.8, 2.2], [1.8, -0.5, 2.0, -1.0, 1.5]) = 1.96

3. 计算 β 值：
   β = Cov(Ri, Rm) / Var(Rm) = 1.96 / 1.89 = 1.04

这个 β 值表明该股票的系统性风险略高于市场，意味着当市场上涨 1%时，该股票预期会上涨 1.04%。

### CAPM 预期收益率计算

假设无风险利率 Rf = 0.3%，市场预期收益率 E[Rm] = 1.0%，股票 β = 1.04，那么：

E[Ri] = Rf + β(E[Rm] - Rf)
E[Ri] = 0.3% + 1.04 × (1.0% - 0.3%)
E[Ri] = 0.3% + 1.04 × 0.7%
E[Ri] = 0.3% + 0.728%
E[Ri] = 1.028%

这表明该股票的预期月收益率为 1.028%。

## 六、结论

1. CAPM 模型理论上预测 β 与收益率存在正相关关系，但在中国股市的实证研究中显示为负相关
2. 除 β 外，公司规模(ME)、市盈率(P/E)和账面市值比(BE/ME)对股票收益率有显著解释力
3. 这些发现挑战了有效市场假说，表明中国股市可能存在明显的市场非效率性
4. 投资者行为可能是非理性的，这暗示在中国股市可能存在利用这些因素的投资策略获取超额收益的机会

这些研究结果与国际市场研究发现类似，支持了 Fama-French 多因素模型的有效性，但中国股市的 β 与收益率负相关的现象则是一个特殊的发现，值得进一步研究。
