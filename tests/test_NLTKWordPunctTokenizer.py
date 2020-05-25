from unittest import TestCase
from community-transformers import NLTKWordPunctTokenizer
from pyspark.sql import SparkSession

class TestNLTKWordPunctTokenizer(TestCase):

    def setUp(self):
        self.tokenizer = NLTKWordPunctTokenizer(inputCol="sentence", outputCol="words",stopwords=['are','I'])
        self.spark = SparkSession.builder.getOrCreate()

        print("Testing simple count")

        # The Spark code will execute on the Azure Databricks cluster.
        print(self.spark.range(100).count())
        return

    def test__transform(self):
        sentenceDataFrame = self.spark.createDataFrame([
            (0, "Hi I heard about Spark"),
            (0, "I wish Java could use case classes"),
            (1, "Logistic regression models are neat")
        ], ["label", "sentence"])

        df_transformed = self.tokenizer.transform(sentenceDataFrame)
        df_transformed.show()
        self.fail()
        return
