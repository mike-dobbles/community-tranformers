import pytest
import transformers.t as ct
from pyspark.sql import SparkSession
from mlflow.spark import save_model, load_model
from pyspark.ml import Pipeline

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
    transformer = ct.NLTKWordPunctTokenizer(inputCol="sentence", outputCol="words", stopwords=['are', 'I'])
    df_transformed = transformer.transform(simple_test_dataframe)

    # Create a pipeline from the transformer
    pipeline = Pipeline(stages=[transformer])

    # fit the test data (which also builds the pipeline)
    model = pipeline.fit(simple_test_dataframe)

    # Test the pipeline
    df_original_transformed = model.transform(simple_test_dataframe)

    # Log the model and performance
    save_model(model, "unit_test_model")
    retrieved_model = load_model("unit_test_model")
    df_retreived_transformed = retrieved_model.transform(simple_test_dataframe)

    # Assert the retrieved model give the same results as the saved model
    rows_in_common = df_original_transformed.intersect(df_retreived_transformed).count()
    assert (df_original_transformed.count() == rows_in_common)

    # Print results for visual inspection
    print("\n")
    print("test__NLTKWordPunctTokenizer: The following should show sentences broken into words")
    df_retreived_transformed.select('sentence','words').show()

    # If we make it this far without crashing we pass (plus I'm visually reviewing results)
    assert True