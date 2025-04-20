# 金融风险度量：VaR 模型与回测

## 1. 在险价值(Value at Risk, VaR)基本概念

在险价值(VaR)是一种流行的金融风险度量方法，用于估计在给定置信水平下，在一定持有期内可能发生的最大损失。

**定义**：对于给定的置信水平 α，VaR 是满足下面条件的值：

$$VaR_{\alpha}^{t} = -\sup\{r | P[R_t \leq r] \leq \alpha\}$$

其中$R_t$表示 t 时刻的资产回报率，α 通常取 1%或 5%。由于监管惯例，VaR 为正值。

## 2. VaR 估计方法

### 2.1 历史模拟法(Historical Simulation, HS)

最简单的 VaR 估计方法，基于历史收益率数据的经验分布。

**数学表达**：
$$VaR_{HS}^{t} = -R_{(\omega)}$$

其中$R_{(\omega)}$是收益率样本的$\omega$阶顺序统计量，$\omega = [T \times \alpha]$，T 是样本大小。

**Python 实现**：

```python
import numpy as np

def historical_simulation_var(returns, alpha=0.01):
    """
    使用历史模拟法计算VaR

    参数:
    returns: 历史收益率序列
    alpha: 置信水平，默认1%

    返回:
    VaR值
    """
    sorted_returns = np.sort(returns)
    index = int(np.ceil(alpha * len(returns)))
    return -sorted_returns[index]

# 示例
returns = np.random.normal(0.001, 0.02, 1000)  # 生成1000个正态分布的收益率样本
var_1pct = historical_simulation_var(returns, 0.01)
print(f"1% VaR: {var_1pct:.4f}")
```

### 2.2 参数化方法：基于条件波动率的 VaR

这类方法通常采用两步骤：

1. 估计条件均值和条件波动率
2. 估计标准化残差的分布分位数

**数学表达**：
$$R_t = \mu_t + \varepsilon_t\sigma_t$$
$$VaR_{\alpha}^{t} = \mu_t + q_{\varepsilon}^{\alpha}\sigma_t$$

其中$q_{\varepsilon}^{\alpha}$是标准化残差的 α 分位数。

#### GARCH 模型

GARCH(1,1)模型可以用于估计条件波动率：

$$\sigma_t^2 = \beta_0 + \beta_1\varepsilon_{t-1}^2 + \beta_2\sigma_{t-1}^2$$

**Python 实现**：

```python
import numpy as np
from scipy.stats import norm, t

def garch_var(returns, alpha=0.01, dist='normal'):
    """
    使用GARCH模型估计VaR

    参数:
    returns: 收益率序列
    alpha: 置信水平
    dist: 分布假设，'normal'或't'

    返回:
    VaR估计值
    """
    # 假设我们已经有了GARCH参数估计和预测的条件波动率
    mu = np.mean(returns)
    sigma = np.std(returns)  # 简化示例，实际应使用GARCH模型估计的条件波动率

    if dist == 'normal':
        q_alpha = norm.ppf(alpha)
    elif dist == 't':
        # 假设自由度为5的t分布
        q_alpha = t.ppf(alpha, 5)

    var = -(mu + q_alpha * sigma)
    return var

# 示例
returns = np.random.normal(0.001, 0.02, 1000)
var_normal = garch_var(returns, 0.01, 'normal')
var_t = garch_var(returns, 0.01, 't')
print(f"正态分布假设下的1% VaR: {var_normal:.4f}")
print(f"t分布假设下的1% VaR: {var_t:.4f}")
```

### 2.3 极值理论方法(Extreme Value Theory, EVT)

极值理论专注于分布尾部的建模，有两种主要方法：

#### 2.3.1 峰值超限法(Peak Over Threshold, POT)

POT 方法对超过特定阈值的观测值建模，使用广义帕累托分布(GPD)：

$$VaR_{GPD}^t = -\left(R_{(k+1)} + \frac{\hat{\sigma}}{\hat{\xi}}\left[\left(\frac{\alpha}{k/T}\right)^{-\hat{\xi}} - 1\right]\right)$$

其中$R_{(k+1)}$是阈值，k 是超过阈值的观测值数量，$\hat{\xi}$和$\hat{\sigma}$是 GPD 分布的形状和尺度参数估计。

**Python 实现**：

```python
import numpy as np

def gpd_var(returns, alpha=0.01, threshold_method='fixed', threshold_value=None):
    """
    使用GPD方法估计VaR

    参数:
    returns: 收益率序列
    alpha: 置信水平
    threshold_method: 阈值选择方法
    threshold_value: 阈值（如果threshold_method='fixed'）

    返回:
    VaR估计值
    """
    # 简化实现，实际应包括参数估计和阈值选择
    sorted_returns = np.sort(returns)

    # 阈值选择
    if threshold_method == 'fixed':
        threshold = threshold_value if threshold_value else np.percentile(returns, 5)
    else:
        # 其他阈值选择方法
        threshold = np.percentile(returns, 10)

    # 找出超过阈值的观测值
    exceedances = sorted_returns[sorted_returns < threshold]
    k = len(exceedances)

    # 简化的参数估计（实际应使用最大似然估计）
    xi = 0.3  # 形状参数
    sigma = 0.02  # 尺度参数

    # 计算VaR
    var = -(threshold + (sigma/xi) * (((alpha/(k/len(returns)))**(-xi)) - 1))
    return var

# 示例
returns = np.random.normal(0.001, 0.02, 1000)
var_gpd = gpd_var(returns, 0.01)
print(f"GPD方法估计的1% VaR: {var_gpd:.4f}")
```

### 2.4 条件自回归值(CAViaR)模型

CAViaR 直接对分位数建模，不需要对条件波动率进行建模。

**数学表达**：
$$VaR_{CAV}^t = \beta_0 + \beta_1 VaR_{CAV}^{t-1} + \beta_2 \ell(X_{t-1})$$

其中$\ell(\cdot)$是过去观测变量的函数。

几种常见的 CAViaR 模型规范：

1. 自适应（Adaptive）CAViaR
2. 绝对值（Absolute Value）CAViaR
3. 不对称（Asymmetric Slope）CAViaR

**Python 简化示例**：

```python
import numpy as np

def caviar_var_forecast(returns, var_prev, model='adaptive', params=None):
    """
    使用CAViaR模型预测VaR

    参数:
    returns: 历史收益率
    var_prev: 上一期VaR
    model: CAViaR模型类型
    params: 模型参数

    返回:
    下一期VaR预测
    """
    if params is None:
        # 默认参数，实际应通过分位数回归估计
        if model == 'adaptive':
            params = {'beta0': 0, 'beta1': 1, 'c': 10}
        elif model == 'absolute':
            params = {'beta0': 0.01, 'beta1': 0.7, 'beta2': 0.3}
        elif model == 'asymmetric':
            params = {'beta0': 0.01, 'beta1': 0.7, 'beta3': 0.2, 'beta4': 0.1}

    if model == 'adaptive':
        return params['beta1'] * var_prev + (1/(1 + np.exp(params['c'] * (returns[-1] + var_prev))) - 0.01)
    elif model == 'absolute':
        return params['beta0'] + params['beta1'] * var_prev + params['beta2'] * np.abs(returns[-1])
    elif model == 'asymmetric':
        return params['beta0'] + params['beta1'] * var_prev + params['beta3'] * max(returns[-1], 0) + params['beta4'] * min(returns[-1], 0)

# 示例
returns = np.random.normal(0.001, 0.02, 1000)
var_init = 0.05  # 初始VaR
var_adaptive = caviar_var_forecast(returns, var_init, 'adaptive')
var_absolute = caviar_var_forecast(returns, var_init, 'absolute')
var_asymmetric = caviar_var_forecast(returns, var_init, 'asymmetric')

print(f"自适应CAViaR VaR预测: {var_adaptive:.4f}")
print(f"绝对值CAViaR VaR预测: {var_absolute:.4f}")
print(f"不对称CAViaR VaR预测: {var_asymmetric:.4f}")
```

## 3. VaR 模型回测

回测是评估 VaR 模型有效性的重要步骤。常见的回测方法包括：

### 3.1 无条件覆盖检验(Unconditional Coverage Test)

检验 VaR 违约率是否等于目标水平 α。

定义击中指示变量：$I_t^{\alpha} = 1_{(R_t < -VaR_t^{\alpha})}$

无条件覆盖假设：$E[I_t^{\alpha}] = \alpha$

**似然比检验**：
$$LR_{uc} = -2\ln\left[\frac{\alpha^{n_1}(1-\alpha)^{n_0}}{(\frac{n_1}{n_0+n_1})^{n_1}(\frac{n_0}{n_0+n_1})^{n_0}}\right]$$

其中$n_1$是 VaR 违约次数，$n_0$是非违约次数。$LR_{uc}$渐近服从$\chi^2(1)$分布。

```python
import numpy as np
from scipy.stats import chi2

def unconditional_coverage_test(violations, alpha, obs):
    """
    执行无条件覆盖检验

    参数:
    violations: VaR违约次数
    alpha: 目标显著性水平
    obs: 总观测数

    返回:
    LR统计量和p值
    """
    n1 = violations
    n0 = obs - n1
    violation_rate = n1 / obs

    if n1 == 0:
        return np.nan, 1.0

    lr_uc = -2 * (n1 * np.log(alpha) + n0 * np.log(1 - alpha)) + 2 * (n1 * np.log(violation_rate) + n0 * np.log(1 - violation_rate))
    p_value = 1 - chi2.cdf(lr_uc, 1)

    return lr_uc, p_value

# 示例
violations = 12  # 假设有12次VaR违约
total_obs = 1000  # 总观测数
alpha = 0.01  # 目标显著性水平

lr, p_val = unconditional_coverage_test(violations, alpha, total_obs)
print(f"无条件覆盖检验 - LR统计量: {lr:.4f}, p值: {p_val:.4f}")
print(f"实际违约率: {violations/total_obs:.4f}, 目标违约率: {alpha:.4f}")
```

### 3.2 条件覆盖检验(Conditional Coverage Test)

检验 VaR 违约是否呈现聚集特性。

条件覆盖假设：$E[I_t^{\alpha}|I_{t-1}^{\alpha}] = \alpha$

**似然比检验**：
$$LR_{cc} = LR_{uc} + LR_{ind}$$

其中$LR_{ind}$是独立性检验的似然比统计量。$LR_{cc}$渐近服从$\chi^2(2)$分布。

```python
import numpy as np
from scipy.stats import chi2

def conditional_coverage_test(hit_sequence, alpha):
    """
    执行条件覆盖检验

    参数:
    hit_sequence: 击中序列，0表示非违约，1表示违约
    alpha: 目标显著性水平

    返回:
    LR统计量和p值
    """
    T = len(hit_sequence)
    n1 = sum(hit_sequence)
    n0 = T - n1

    # 状态转移计数
    n00 = 0
    n01 = 0
    n10 = 0
    n11 = 0

    for i in range(1, T):
        if hit_sequence[i-1] == 0 and hit_sequence[i] == 0:
            n00 += 1
        elif hit_sequence[i-1] == 0 and hit_sequence[i] == 1:
            n01 += 1
        elif hit_sequence[i-1] == 1 and hit_sequence[i] == 0:
            n10 += 1
        else:  # hit_sequence[i-1] == 1 and hit_sequence[i] == 1
            n11 += 1

    # 无条件覆盖部分
    pi = n1 / T
    lr_uc = -2 * (n1 * np.log(alpha) + n0 * np.log(1 - alpha)) + 2 * (n1 * np.log(pi) + n0 * np.log(1 - pi))

    # 独立性部分
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0

    if pi01 == 0 or pi11 == 0 or n00 == 0 or n10 == 0:
        return np.nan, 1.0

    lr_ind = -2 * (n00 * np.log(1 - pi) + n01 * np.log(pi) + n10 * np.log(1 - pi) + n11 * np.log(pi)) + 2 * (n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11))

    # 条件覆盖检验
    lr_cc = lr_uc + lr_ind
    p_value = 1 - chi2.cdf(lr_cc, 2)

    return lr_cc, p_value

# 示例
np.random.seed(42)
hit_sequence = np.random.binomial(1, 0.01, 1000)  # 生成随机击中序列
lr_cc, p_val = conditional_coverage_test(hit_sequence, 0.01)
print(f"条件覆盖检验 - LR统计量: {lr_cc:.4f}, p值: {p_val:.4f}")
```

### 3.3 动态分位数检验(Dynamic Quantile Test)

检验击中序列是否为鞅差序列。

**数学表达**：
对中心化击中序列$\delta_t^\alpha = I_t^\alpha - \alpha$检验：
$$E[\delta_t^\alpha \otimes X_{t-1}] = 0$$

其中$X_{t-1}$包含$t-1$时刻信息集中的任何向量。

```python
import numpy as np
import statsmodels.api as sm

def dynamic_quantile_test(hit_sequence, var_forecasts, alpha, lags=5):
    """
    执行动态分位数检验

    参数:
    hit_sequence: 击中序列，0表示非违约，1表示违约
    var_forecasts: VaR预测值
    alpha: 目标显著性水平
    lags: 考虑的滞后项数

    返回:
    DQ统计量和p值
    """
    T = len(hit_sequence)

    # 中心化击中序列
    centered_hits = hit_sequence - alpha

    # 构建解释变量矩阵
    X = np.ones((T-lags, lags+1))

    for i in range(lags):
        X[:, i+1] = centered_hits[i:T-lags+i]

    # 被解释变量
    y = centered_hits[lags:]

    # OLS回归
    model = sm.OLS(y, X)
    results = model.fit()

    # 计算DQ统计量
    dq = results.nobs * results.rsquared
    p_value = 1 - chi2.cdf(dq, lags+1)

    return dq, p_value

# 示例
np.random.seed(42)
hit_sequence = np.random.binomial(1, 0.01, 1000)
var_forecasts = np.random.normal(0.05, 0.01, 1000)  # 假设的VaR预测

# 注意：实际使用时需要安装statsmodels库
# dq, p_val = dynamic_quantile_test(hit_sequence, var_forecasts, 0.01)
# print(f"动态分位数检验 - DQ统计量: {dq:.4f}, p值: {p_val:.4f}")
```

## 4. VaR 模型比较方法

当多个 VaR 模型被证明都是有效的时，我们需要比较它们的性能以选择最佳模型。

### 4.1 监管损失函数(Regulatory Loss Function)

$$C(m) = \sum_{t=T+1}^{T+H} C_t^{(m)}$$

其中：

$$
C_t^{(m)} = \begin{cases}
f(R_t, VaR_t^{(m)}) & \text{如果} R_t < VaR_t^{(m)} \\
g(R_t, VaR_t^{(m)}) & \text{如果} R_t \geq VaR_t^{(m)}
\end{cases}
$$

一个常用的损失函数形式是：$f(R_t, VaR_t^{(m)}) = (R_t - VaR_t^{(m)})^2$, $g(R_t, VaR_t^{(m)}) = 0$

```python
import numpy as np

def regulatory_loss_function(returns, var_forecasts):
    """
    计算监管损失函数

    参数:
    returns: 实际收益率
    var_forecasts: VaR预测值

    返回:
    总损失
    """
    losses = np.zeros_like(returns)

    for i in range(len(returns)):
        if returns[i] < -var_forecasts[i]:  # 违约情况
            losses[i] = (returns[i] + var_forecasts[i])**2
        else:  # 非违约情况
            losses[i] = 0

    return np.sum(losses)

# 示例
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, 1000)
var_forecasts1 = np.random.normal(0.05, 0.01, 1000)
var_forecasts2 = np.random.normal(0.04, 0.008, 1000)

loss1 = regulatory_loss_function(returns, var_forecasts1)
loss2 = regulatory_loss_function(returns, var_forecasts2)

print(f"模型1损失: {loss1:.4f}")
print(f"模型2损失: {loss2:.4f}")
print(f"最佳模型: {'模型1' if loss1 < loss2 else '模型2'}")
```

### 4.2 预测分位数损失函数(Predictive Quantile Loss Function)

$$C_t^{(m)} = (\alpha - 1\{R_t < VaR_t^{\alpha}\})(R_t - VaR_t^{\alpha})$$

这个损失函数考虑了违约的严重程度。

```python
import numpy as np

def predictive_quantile_loss(returns, var_forecasts, alpha=0.01):
    """
    计算预测分位数损失函数

    参数:
    returns: 实际收益率
    var_forecasts: VaR预测值
    alpha: 目标显著性水平

    返回:
    总损失
    """
    losses = np.zeros_like(returns)

    for i in range(len(returns)):
        indicator = 1 if returns[i] < -var_forecasts[i] else 0
        losses[i] = (alpha - indicator) * (returns[i] + var_forecasts[i])

    return np.sum(losses)

# 示例
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, 1000)
var_forecasts1 = np.random.normal(0.05, 0.01, 1000)
var_forecasts2 = np.random.normal(0.04, 0.008, 1000)

loss1 = predictive_quantile_loss(returns, var_forecasts1)
loss2 = predictive_quantile_loss(returns, var_forecasts2)

print(f"模型1分位数损失: {loss1:.4f}")
print(f"模型2分位数损失: {loss2:.4f}")
print(f"最佳模型: {'模型1' if loss1 < loss2 else '模型2'}")
```

## 总结

在险价值(VaR)是金融风险管理中的重要工具，有多种估计方法：

1. 历史模拟法 - 简单但无法捕捉条件波动性
2. 参数化方法 - 考虑时变波动性但模型设定风险
3. CAViaR 模型 - 直接对分位数建模
4. 极值理论方法 - 专注于尾部风险建模

有效的 VaR 模型应通过多种回测方法进行验证：

1. 无条件覆盖检验 - 检验总体违约率
2. 条件覆盖检验 - 考虑违约聚集性
3. 动态分位数检验 - 检验击中序列预测性

模型比较通过监管损失函数或预测分位数损失函数进行，以选择最优模型。
