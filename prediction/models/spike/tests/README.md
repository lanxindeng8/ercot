# Test Documentation

## Test Files

- `test_feature_engineering.py`: Tests for the feature engineering module
- `test_labels.py`: Tests for the label generation module
- `run_tests.py`: Script to run all tests

## Running Tests

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install required packages (for testing)
pip install pandas numpy
```

### 2. Run All Tests

```bash
# Method 1: Using unittest
python3 -m unittest discover tests -v

# Method 2: Using the test script
python3 tests/run_tests.py

# Method 3: Run individual test files
python3 tests/test_feature_engineering.py
python3 tests/test_labels.py
```

### 3. Using pytest (Optional, More Detailed Output)

```bash
# Install pytest
pip install pytest

# Run tests
pytest tests/ -v
pytest tests/test_feature_engineering.py -v
pytest tests/test_labels.py -v
```

## Test Coverage

### test_feature_engineering.py

- **TestPriceStructureFeatures**: Price structure features
  - Feature computation
  - Spread calculation validation

- **TestSupplyDemandFeatures**: Supply-demand balance features
  - Net load calculation
  - Wind anomaly
  - Coal stress flag

- **TestWeatherFeatures**: Weather-driven features
  - Temperature anomaly
  - Wind chill index
  - Cold front flag

- **TestTemporalFeatures**: Temporal features
  - Hour range
  - Evening peak flag

- **TestFeatureEngineer**: Feature engineering main class
  - Compute all features
  - Get feature names

### test_labels.py

- **TestSpikeLabels**: Spike label generation
  - SpikeEvent label
  - LeadSpike label
  - Regime label
  - Percentile threshold

- **TestLabelGenerator**: Label generator main class
  - Generate all labels
  - Identify independent events
  - Label consistency
  - Multi-zone labels

- **TestEdgeCases**: Edge cases
  - No spike data
  - All spike data
  - Short-lived spike

## Test Status

The current tests require the following packages to run:
- pandas
- numpy

These packages are included in `requirements.txt`.

## CI/CD Integration (Future)

Tests can be integrated into a CI/CD pipeline:

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
