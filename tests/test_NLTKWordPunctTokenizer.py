from unittest import TestCase
from transformers import NLTKWordPunctTokenizer as tok
from pyspark.sql import SparkSession

class TestNLTKWordPunctTokenizer(TestCase):

    def setUp(self):
        self.tokenizer = tok.NLTKWordPunctTokenizer(inputCol="sentence", outputCol="words",stopwords=['are','I'])
        self.spark = SparkSession.builder.getOrCreate()
        return

    def test__transform(self):
        sentenceDataFrame = self.spark.createDataFrame([
            (0, "Hi I heard about Spark"),
            (0, "I wish Java could use case classes"),
            (1, "Logistic regression models are neat")
        ], ["label", "sentence"])

        df_transformed = self.tokenizer.transform(sentenceDataFrame)
        df_transformed.show()
        self.assertTrue(True)
        return
