"""
运行所有测试
"""

import sys
import unittest

# 发现并运行所有测试
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回退出码
    sys.exit(0 if result.wasSuccessful() else 1)
