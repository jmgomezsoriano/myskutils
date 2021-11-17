import unittest

from results import conf_95, conf_99, mean, sd, se

from mysutils.file import load_json

from myskutils.metrics import sk_measure, SIMPLE_ACCURACY, format_value, select_metrics, MACRO_JACCARD, MACRO_F1, \
    MACRO_RECALL
from myskutils.stats import confidence_score, measures_mean, standard_deviation, standard_error

Y_TRUES_FILE = 'test/trues.json'
Y_PRED_FILE = 'test/pred.json'
MEASURES_FILE = 'test/data.json'
CONF_95_FILE = 'test/conf_95.json'
CONF_99_FILE = 'test/conf_99.json'
result = {
    'simple_accuracy': 0.7314814814814815, 'balanced_accuracy': 0.7336080586080586,
    'micro_f1': 0.7314814814814816, 'macro_f1': 0.6676911015978718, 'weighted_f1': 0.7064382543140714,
    'micro_precision': 0.7314814814814815, 'macro_precision': 0.6733602875112309,
    'weighted_precision': 0.7421737213403881, 'micro_recall': 0.7314814814814815,
    'macro_recall': 0.7197663971248877, 'weighted_recall': 0.7314814814814815,
    'micro_jaccard': 0.5766423357664233, 'macro_jaccard': 0.6027144762993819,
    'weighted_jaccard': 0.6215661910106355
}


class MyTestCase(unittest.TestCase):
    def test_measurement(self):
        y_trues, y_pred = load_json(Y_TRUES_FILE), load_json(Y_PRED_FILE)
        measure = sk_measure(y_trues, y_pred)
        self.assertDictEqual(result, measure)  # add assertion here
        self.assertEqual(format_value(measure[SIMPLE_ACCURACY]), '73.15')
        self.assertEqual(format_value(measure[SIMPLE_ACCURACY], 3), '73.148')
        self.assertEqual(format_value(measure[SIMPLE_ACCURACY], 0), '73')
        measures = load_json(MEASURES_FILE)
        self.assertDictEqual(confidence_score(measures), conf_95)
        self.assertDictEqual(confidence_score(measures, 0.99), conf_99)
        self.assertDictEqual(measures_mean(measures), mean)
        self.assertDictEqual(standard_deviation(measures), sd)
        self.assertDictEqual(standard_error(measures), se)
        self.assertDictEqual(select_metrics(measure), {})
        self.assertDictEqual(select_metrics(measure, SIMPLE_ACCURACY, MACRO_RECALL, MACRO_F1, MACRO_JACCARD),
                             {'macro_f1': 0.6676911015978718, 'macro_jaccard': 0.6027144762993819,
                              'macro_recall': 0.7197663971248877, 'simple_accuracy': 0.7314814814814815})


if __name__ == '__main__':
    unittest.main()
