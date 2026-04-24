"""
Run all tests
"""

import sys
import unittest

# Discover and run all tests
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    sys.exit(0 if result.wasSuccessful() else 1)
