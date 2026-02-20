# 测试说明

## 测试文件

- `test_feature_engineering.py`: 测试特征工程模块
- `test_labels.py`: 测试标签生成模块
- `run_tests.py`: 运行所有测试的脚本

## 运行测试

### 1. 安装依赖

```bash
# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装必需的包（用于测试）
pip install pandas numpy
```

### 2. 运行所有测试

```bash
# 方法 1: 使用 unittest
python3 -m unittest discover tests -v

# 方法 2: 使用测试脚本
python3 tests/run_tests.py

# 方法 3: 运行单个测试文件
python3 tests/test_feature_engineering.py
python3 tests/test_labels.py
```

### 3. 使用 pytest（可选，更详细的输出）

```bash
# 安装 pytest
pip install pytest

# 运行测试
pytest tests/ -v
pytest tests/test_feature_engineering.py -v
pytest tests/test_labels.py -v
```

## 测试覆盖

### test_feature_engineering.py

- **TestPriceStructureFeatures**: 价格结构特征
  - 特征计算
  - 价差计算验证

- **TestSupplyDemandFeatures**: 供需平衡特征
  - 净负荷计算
  - 风电异常
  - 煤电压力标志

- **TestWeatherFeatures**: 天气驱动特征
  - 温度异常
  - 风寒指数
  - 冷锋标志

- **TestTemporalFeatures**: 时间特征
  - 小时范围
  - 晚高峰标志

- **TestFeatureEngineer**: 特征工程主类
  - 计算所有特征
  - 获取特征名称

### test_labels.py

- **TestSpikeLabels**: Spike 标签生成
  - SpikeEvent 标签
  - LeadSpike 标签
  - Regime 标签
  - 分位数阈值

- **TestLabelGenerator**: 标签生成主类
  - 生成所有标签
  - 识别独立事件
  - 标签一致性
  - 多区域标签

- **TestEdgeCases**: 边界情况
  - 无 spike 数据
  - 全 spike 数据
  - 短暂 spike

## 测试状态

当前测试需要以下依赖包才能运行：
- pandas
- numpy

这些包已包含在 `requirements.txt` 中。

## CI/CD 集成（未来）

可以将测试集成到 CI/CD pipeline：

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python3 -m unittest discover tests -v
```
