import pytest
from transformers import t as tok
from pyspark.sql import SparkSession

def test__first_test():
    tokenizer = tok.NLTKWordPunctTokenizer(inputCol="sentence", outputCol="words", stopwords=['are', 'I'])
    spark = SparkSession.builder.getOrCreate()
    sentenceDataFrame = spark.createDataFrame([
        (0, "Hi I heard about Spark"),
        (0, "I wish Java could use case classes"),
        (1, "Logistic regression models are neat")
    ], ["label", "sentence"])

    df_transformed = tokenizer.transform(sentenceDataFrame)
    df_transformed.show()
    assert True