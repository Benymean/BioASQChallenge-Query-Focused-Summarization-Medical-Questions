
import unittest
import a3_1

class TestBasic(unittest.TestCase):
    def test_task1(self):
        a,b = a3_1.stats_pos('data/dev_test.csv')
        self.assertEqual((a,b),
                             ([('.', 0.1201), ('ADJ', 0.0892), ('ADP', 0.1119), ('ADV', 0.011), ('CONJ', 0.0085), 
                               ('DET', 0.085), ('NOUN', 0.3536), ('NUM', 0.0056), ('PRON', 0.0377), ('PRT', 0.0104), 
                               ('VERB', 0.1659), ('X', 0.0011)], 
                              [('.', 0.1236), ('ADJ', 0.1204), ('ADP', 0.117), ('ADV', 0.0245), ('CONJ', 0.0349), 
                               ('DET', 0.0769), ('NOUN', 0.3462), ('NUM', 0.0186), ('PRON', 0.01), ('PRT', 0.0159), 
                               ('VERB', 0.1112), ('X', 0.0009)])
                        )

    def test_task2(self):
        self.assertEqual(a3_1.stats_top_stem_ngrams('data/dev_test.csv', 2, 5),
                         ([(('what', 'is'), 0.0294), (('is', 'the'), 0.0265), (('of', 'the'), 0.0104), (('in', 'the'), 0.006), 
                           (('are', 'the'), 0.0055)], 
                          [(('of', 'the'), 0.0064), (('in', 'the'), 0.0054), ((',', 'and'), 0.0054), ((')', ','), 0.0041), 
                           (('is', 'a'), 0.0029)])
                         )

    def test_task3(self):
        a,b = a3_1.stats_ne('data/dev_test.csv')
        self.assertEqual((a,b),
                         ([('CARDINAL', 0.0966), ('DATE', 0.0207), ('EVENT', 0.0046), ('FAC', 0.0046), 
                           ('GPE', 0.1172), ('LAW', 0.0092), ('LOC', 0.0138), ('NORP', 0.0529), ('ORDINAL', 0.0115), 
                           ('ORG', 0.3977), ('PERCENT', 0.0023), ('PERSON', 0.2115), ('PRODUCT', 0.0483), 
                           ('QUANTITY', 0.0023), ('WORK_OF_ART', 0.0069)], 
                          [('CARDINAL', 0.1887), ('DATE', 0.013), ('FAC', 0.0043), ('GPE', 0.0564), ('LAW', 0.0174), 
                           ('LOC', 0.0043), ('MONEY', 0.0022), ('NORP', 0.0412), ('ORDINAL', 0.026), ('ORG', 0.4512), 
                           ('PERCENT', 0.013), ('PERSON', 0.1367), ('PRODUCT', 0.0347), ('QUANTITY', 0.0043), 
                           ('TIME', 0.0043), ('WORK_OF_ART', 0.0022)])
                         )

    def test_task4(self):
        self.assertEqual(a3_1.stats_tfidf('data/dev_test.csv'),0.4876)


if __name__ == "__main__":
    unittest.main()
