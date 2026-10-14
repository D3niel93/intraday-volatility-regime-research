import unittest
import pandas as pd
from evaluate import evaluate
class EvaluationTests(unittest.TestCase):
    def test_coverage_not_all_bars(self):
        f=pd.DataFrame(dict(actual_expansive=[1,0,1,0],probability_expansive=[.9,.1,.51,.49]))
        m=evaluate(f)
        self.assertEqual(m["coverage_pct"],50)
        self.assertEqual(m["selective_accuracy_pct"],100)
    def test_bad_probability(self):
        with self.assertRaises(ValueError):
            evaluate(pd.DataFrame(dict(actual_expansive=[1,0],probability_expansive=[1.2,.5])))
