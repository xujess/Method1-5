import random
import statistics
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.title("方法一至五 随机抽取报价")

# 使用st.columns()将输入框排列在一行
col1, col2 = st.columns(2)

# 创建报价范围的文本输入框
with col1:
    lower = st.number_input("最低报价", min_value=0.0, max_value=1.0, value=0.758, step=0.001, format="%.3f")
with col2:
    upper = st.number_input("最高报价", min_value=0.0, max_value=1.0, value=0.898, step=0.001, format="%.3f")

# 使用st.columns()将输入框排列在一行
col3, col4 = st.columns(2)

# 创建报价数量的区间选择器
with col3:
    num_bids_lower = st.number_input("最低报价数量", min_value=1, max_value=50, value=7, step=1)
with col4:
    num_bids_upper = st.number_input("最高报价数量", min_value=1, max_value=50, value=11, step=1)

# 创建循环次数的文本输入框
num_iterations = st.number_input("跑多少次数据", min_value=1, max_value=1000000, value=100000, step=1)


st.title("方法一")

# 假设我们使用1000作为因子来转换0.758和0.898为整数范围
factor = 1000

# 调整lower和upper以便使用randint()
int_lower = int(lower * factor)
int_upper = int(upper * factor)

# 记录所有基准价
results = []

# 循环指定次数实现
for i in range(num_iterations):

    # 随机选择报价数量
    num_bids = random.randint(num_bids_lower, num_bids_upper)

    # 生成报价，这里需要转换回浮点数
    bids = [random.randint(int_lower, int_upper) / factor for i in range(num_bids)]

    # 排序报价
    bids.sort()

    # 根据报价数量计算去除的报价数
    remove_num = int(round(num_bids * 0.2))

    if num_bids >= 7:
        bids = bids[remove_num:-remove_num]
    elif 4 <= num_bids < 7:
        bids = bids[:-1]
    else:
        bids = bids[1]

    # 计算平均价A
    A = statistics.mean(bids)

    # K值随机选择0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01
    K = random.choice([0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01])

    # 计算基准价
    benchmark = A * K

    # 记录基准价
    results.append(benchmark)


# 计算平均值、中位数和标准差
mean = np.mean(results)
median = np.median(results)
std = np.std(results)
low = mean - std
high = mean + std

# 创建一个包含统计数据的DataFrame
stats_data = pd.DataFrame({
    'Stats': ['平均值', '中位数', '标准差', '平均值- 1*标准差', '平均值+ 1*标准差'],
    'Value': [f"{mean:.6f}", f"{median:.6f}", f"{std:.6f}", f"{low:.6f}", f"{high:.6f}"]
})

# 在Streamlit中以表格形式展示统计数据
st.table(stats_data.set_index('Stats'))


# 绘制基准价分布
plt.figure()  # Add this line to create a new figure for method 1
plt.hist(results, bins=100)  # 按0.01的间隔划分bin
mean_line = statistics.mean(results)  # Changed variable name to mean_line
plt.axvline(mean, color='r', linestyle='--', label='Mean')
plt.axvline(low, color='b', linestyle=':', label='Mean - Std')
plt.axvline(high, color='b', linestyle=':', label='Mean + Std')

plt.legend()

plt.xlabel('Benchmark')
plt.ylabel('Count')
plt.xlim(lower, upper)

# 在Streamlit中显示图表
st.pyplot(plt)



########################################

# 方法二
control_price = 1

st.title("方法二")


# 输入自定义数值 K2
K2 = st.number_input('输入自定义数值 K2', value = 0.93, format="%.2f")


# 记录所有基准价
results = []

# 循环指定次数实现
for i in range(num_iterations):

  # 随机选择报价数量
  num_bids = random.randint(num_bids_lower, num_bids_upper)

  # 生成报价，这里需要转换回浮点数
  bids = [random.randint(int_lower, int_upper) / factor for i in range(num_bids)]

  # 排序报价
  bids.sort()

  # 根据报价数量计算去除的报价数
  remove_num = int(round(num_bids * 0.2))

  if num_bids >= 7:
    bids = bids[remove_num:-remove_num]

  elif 4 <= num_bids < 7:
    bids = bids[:-1]

  else:
    bids = bids[1]

  # 计算平均价A
  A = statistics.mean(bids)

  # B为控制价
  B = control_price

  # 随机抽取Q1和K1
  Q1 = random.choice([0.65, 0.7, 0.75, 0.8, 0.85])
  K1 = random.choice([0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01])

  # 计算基准价
  benchmark = A * K1 * Q1 + B * K2 * (1-Q1)

  # 记录基准价
  results.append(benchmark)

# 计算平均值、中位数和标准差
mean_val = np.mean(results)
median_val = np.median(results)
std_val = np.std(results)
low = mean_val - std_val
high = mean_val + std_val

# 创建一个包含统计数据的DataFrame
stats_data = pd.DataFrame({
    'Stats': ['平均值', '中位数', '标准差', '平均值- 1*标准差', '平均值+ 1*标准差'],
    'Value': [f"{mean_val:.6f}", f"{median_val:.6f}", f"{std_val:.6f}", f"{low:.6f}", f"{high:.6f}"]
})

# 在Streamlit中以表格形式展示统计数据
st.table(stats_data.set_index('Stats'))


# 绘制基准价分布
plt.figure()  # Add this line to create a new figure for method 2
plt.hist(results, bins=100)  # 按0.01的间隔划分bin
mean = statistics.mean(results)
plt.axvline(mean, color='r', linestyle='--', label='Mean')
plt.axvline(low, color='b', linestyle=':', label='Mean - Std')
plt.axvline(high, color='b', linestyle=':', label='Mean + Std')

plt.legend()

plt.xlabel('Benchmark')
plt.ylabel('Count')
plt.xlim(lower, upper)

# 在Streamlit中显示图表
st.pyplot(plt)


########################################
# 方法三


st.title("方法三")

# 记录所有基准价
results = []

# 循环指定次数实现
for i in range(num_iterations):

  # 随机选择报价数量
  num_bids = random.randint(num_bids_lower, num_bids_upper)

  # 生成报价，这里需要转换回浮点数
  bids = [random.randint(int_lower, int_upper) / factor for i in range(num_bids)]

  # 排序报价
  bids.sort()

  # 计算基准价为次低评标价
  benchmark = bids[1]

  # 记录基准价
  results.append(benchmark)


# 计算平均值、中位数和标准差
mean_val = np.mean(results)
median_val = np.median(results)
std_val = np.std(results)
low = mean_val - std_val
high = mean_val + std_val

# 创建一个包含统计数据的DataFrame
stats_data = pd.DataFrame({
    'Stats': ['平均值', '中位数', '标准差', '平均值- 1*标准差', '平均值+ 1*标准差'],
    'Value': [f"{mean_val:.6f}", f"{median_val:.6f}", f"{std_val:.6f}", f"{low:.6f}", f"{high:.6f}"]
})

# 在Streamlit中以表格形式展示统计数据
st.table(stats_data.set_index('Stats'))


# 绘制基准价分布
plt.figure()  # Add this line to create a new figure for method 2
plt.hist(results, bins=500) # 按0.01的间隔划分bin
mean = statistics.mean(results)
plt.axvline(mean, color='r', linestyle='--', label='Mean')

plt.legend()

sigma = statistics.stdev(results)
low = mean - sigma
high = mean + sigma

plt.axvline(low, color='b', linestyle=':', label='Mean - Std')
plt.axvline(high, color='b', linestyle=':', label='Mean + Std')

plt.legend()

plt.xlabel('Benchmark')
plt.ylabel('Frenquency')
plt.xlim(lower, upper)

# 在Streamlit中显示图表
st.pyplot(plt)


########################################
# 方法四

st.title("方法四")

# 输入自定义数值 K2
K = st.number_input('输入自定义数值 K', value = 0.91, format="%.2f")

# 记录所有基准价
results = []

# 循环指定次数实现
for i in range(num_iterations):

  # 随机选择报价数量
  num_bids = random.randint(num_bids_lower, num_bids_upper)

  # 生成报价，这里需要转换回浮点数
  bids = [random.randint(int_lower, int_upper) / factor for i in range(num_bids)]

  # 排序报价
  bids.sort()

  # 根据报价数量计算去除的报价数
  remove_num = int(round(num_bids * 0.2))

  if num_bids >= 7:
    bids = bids[remove_num:-remove_num]

  elif 4 <= num_bids < 7:
    bids = bids[:-1]

  else:
    bids = bids[1]

  # 计算平均价A
  A = statistics.mean(bids)

  # 计算基准价
  benchmark = A * K

  # 记录基准价
  results.append(benchmark)


# 计算平均值、中位数和标准差
mean_val = np.mean(results)
median_val = np.median(results)
std_val = np.std(results)
low = mean_val - std_val
high = mean_val + std_val

# 创建一个包含统计数据的DataFrame
stats_data = pd.DataFrame({
    'Stats': ['平均值', '中位数', '标准差', '平均值- 1*标准差', '平均值+ 1*标准差'],
    'Value': [f"{mean_val:.6f}", f"{median_val:.6f}", f"{std_val:.6f}", f"{low:.6f}", f"{high:.6f}"]
})

# 在Streamlit中以表格形式展示统计数据
st.table(stats_data.set_index('Stats'))


# 绘制基准价分布
plt.figure()  # Add this line to create a new figure for method 2
plt.hist(results, bins=500) # 按0.01的间隔划分bin
mean = statistics.mean(results)
plt.axvline(mean, color='r', linestyle='--', label='Mean')

plt.legend()

sigma = statistics.stdev(results)
low = mean - sigma
high = mean + sigma

plt.axvline(low, color='b', linestyle=':', label='Mean - Std')
plt.axvline(high, color='b', linestyle=':', label='Mean + Std')

plt.legend()

plt.xlabel('Benchmark')
plt.ylabel('Frenquency')
plt.xlim(lower, upper)


# 在Streamlit中显示图表
st.pyplot(plt)


########################################
# 方法五

st.title("方法五")

# 设置默认的 deltas 和 Ks
default_deltas = [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
default_Ks = [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98]

# 用户自定义 deltas 和 Ks 输入
input_deltas = st.text_input("录入下浮率Δ，用逗号分隔开：", value=','.join(map(str, default_deltas)))
input_Ks = st.text_input("录入下浮系数K，用逗号分隔开：", value=','.join(map(str, default_Ks)))

# 转换用户输入为浮点数列表，如果转换失败则使用默认值
try:
    deltas = [float(delta.strip()) for delta in input_deltas.split(",")]
    Ks = [float(K.strip()) for K in input_Ks.split(",")]
except ValueError:
    st.error("下浮率Δ和下浮系数K必须是由逗号分隔的数字。")
    deltas = default_deltas
    Ks = default_Ks


def get_value(bids, A, C):

  # 过滤bids
  btw95_92 = [b for b in bids if A * 0.92 <= b < A * 0.95]
  btw92_89 = [b for b in bids if A * 0.89 <= b < A * 0.92]
  btw89_C = [b for b in bids if C < b < A * 0.89]

  if btw95_92:
    val1 = random.choice(btw95_92)
  else:
    val1 = None

  if btw92_89:
    val2 = random.choice(btw92_89)
  else:
    val2 = None

  # new_bids的范围
  values = [val1, val2, *btw89_C]
  new_bids = [v for v in values if v]

  # bids除去C的范围
  ex_C = [v for v in bids if v != C]

  # 随机抽取一个数就是B值
  if any(values):
      return random.choice(new_bids)
  else:
      return random.choice(ex_C)

# 定义一个列表来存放所有基准价
results = []

# 循环指定次数实现
for i in range(num_iterations):

  # 随机选择报价数量
  num_bids = random.randint(num_bids_lower, num_bids_upper)

  # 生成报价，这里需要转换回浮点数
  bids = [random.randint(int_lower, int_upper) / factor for i in range(num_bids)]

  # A=招标控制价×（100%－下浮率Δ）
  delta = random.choice(deltas)
  A = control_price * (1 - delta)

  # 计算C值
  # 计算评标价平均值 和范围下限
  bid_mean = statistics.mean(bids)
  lower_limit = (bid_mean * 0.7 + control_price * 0.3) * 0.75

  # 过滤bids,得到在范围内的bids
  in_range_bids = [b for b in bids if b >= lower_limit]

  # C=在规定范围内的最低评标价
  in_range_bids.sort()
  C = in_range_bids[0]

  # 计算B值
  B = get_value(bids, A, C)

  # 步骤8: 随机选择K值
  K = random.choice(Ks)

  # 步骤9: 计算加权和和基准价
  benchmark = (A*0.5 + B*0.3 + C*0.2) * K

  # 步骤10: 记录基准价
  results.append(benchmark)


# 计算平均值、中位数和标准差
mean_val = np.mean(results)
median_val = np.median(results)
std_val = np.std(results)
low = mean_val - std_val
high = mean_val + std_val

# 创建一个包含统计数据的DataFrame
stats_data = pd.DataFrame({
    'Stats': ['平均值', '中位数', '标准差', '平均值- 1*标准差', '平均值+ 1*标准差'],
    'Value': [f"{mean_val:.6f}", f"{median_val:.6f}", f"{std_val:.6f}", f"{low:.6f}", f"{high:.6f}"]
})

# 在Streamlit中以表格形式展示统计数据
st.table(stats_data.set_index('Stats'))


# 绘制基准价分布
plt.figure()  # Add this line to create a new figure for method 2
plt.hist(results, bins=500) # 按0.01的间隔划分bin
mean = statistics.mean(results)
plt.axvline(mean, color='r', linestyle='--', label='Mean')

sigma = statistics.stdev(results)
low = mean - sigma
high = mean + sigma

plt.axvline(low, color='b', linestyle=':', label='Mean - Std')
plt.axvline(high, color='b', linestyle=':', label='Mean + Std')

plt.legend()

plt.xlabel('Benchmark')
plt.ylabel('Frenquency')
plt.xlim(lower, upper)

# 在Streamlit中显示图表
st.pyplot(plt)

