import pytest
from transformers import t as tok
from pyspark.sql import SparkSession

# This runs before the tests and creates objects that can be used by the tests
@pytest.fixture
def simple_test_dataframe():
    """This is a simple dataframe for test use"""
    # get a reference to spark
    spark = SparkSession.builder.getOrCreate()

    # create a test data frame
    simple_df = spark.createDataFrame([
        (0, "Hi I heard about Spark"),
        (0, "I wish Java could use case classes"),
        (1, "Logistic regression models are neat")
    ], ["label", "sentence"])

    return simple_df

def test__NLTKWordPunctTokenizer(simple_test_dataframe):

    # Create the transformer
    transformer = tok.NLTKWordPunctTokenizer(inputCol="sentence", outputCol="words", stopwords=['are', 'I'])
    df_transformed = transformer.transform(simple_test_dataframe)
    df_transformed.show()

    # If we make it this far without crashing we pass (plus I'm visually reviewing results)
    assert True