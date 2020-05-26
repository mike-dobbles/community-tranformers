import pytest
import transformers.t as ct
from pyspark.sql import SparkSession
from mlflow.spark import save_model, load_model
from pyspark.ml import Pipeline
import os
import pandas as pd


# This runs before the tests and creates objects that can be used by the tests
@pytest.fixture
def simple_test_dataframe():
    """This is a simple dataframe for test use"""

    # get a reference to spark
    spark = SparkSession.builder.getOrCreate()

    # create a test data frame
    pdf = pd.DataFrame(columns=['text'],
                 data=["This sentence ends with br and will prevent nltk sentence tokenization<br>This sentence ends normally. As does this one",
                       "Some sentences run together.The previous was an example",
                       "This is a normal first sentence.  This is a normal second sentence."
                       ])

    return spark.createDataFrame(pdf)

@pytest.fixture
def numbers_dataframe():
    """This is a dataframe filled with text of numbers for test use"""

    # get a reference to spark
    spark = SparkSession.builder.getOrCreate()

    # create a test data frame
    pdf = pd.DataFrame(columns=['text'],
                 data=["onethousand two three four five",
                       "six seven eight nine eight-hundred-ninetyfive"
                       ])
    return spark.createDataFrame(pdf)

def test__NLTKWordPunctTokenizer(simple_test_dataframe):

    # Create the transformer
    transformer = ct.NLTKWordPunctTokenizer(inputCol="text", outputCol="words", stopwords=['are', 'I'])

    # Create a pipeline from the transformer
    pipeline = Pipeline(stages=[transformer])

    # fit the test data (which also builds the pipeline)
    model = pipeline.fit(simple_test_dataframe)

    # Test the pipeline
    df_original_transformed = model.transform(simple_test_dataframe)

    # Delete any previously save model (if it exists)
    # (There may be a more elegant way to do this)
    if os.path.exists("unit_test_model"):
        os.system("rm -rf unit_test_model")

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
    df_retreived_transformed.show(truncate=False)

    # If we make it this far without crashing we pass (plus I'm visually reviewing results)
    assert True

def test__RegexSubstituter(simple_test_dataframe):

    # Create the transformer
    regexMatchers= ['(?<=[a-zA-Z])\.(?=[A-Z])',
                    '<BR>',
                    '<br>']
    substitutions= ['. ',
                    '. ',
                    '. ']
    transformer = ct.RegexSubstituter(inputCol="text", outputCol="regexcorrected",
                                      regexMatchers=regexMatchers, substitutions=substitutions)

    # Create a pipeline from the transformer
    pipeline = Pipeline(stages=[transformer])

    # fit the test data (which also builds the pipeline)
    model = pipeline.fit(simple_test_dataframe)

    # Test the pipeline
    df_original_transformed = model.transform(simple_test_dataframe)

    # Delete any previously save model (if it exists)
    # (There may be a more elegant way to do this)
    if os.path.exists("unit_test_model"):
        os.system("rm -rf unit_test_model")

    # Log the model and performance
    save_model(model, "unit_test_model")
    retrieved_model = load_model("unit_test_model")
    df_retreived_transformed = retrieved_model.transform(simple_test_dataframe)

    # Assert the retrieved model give the same results as the saved model
    rows_in_common = df_original_transformed.intersect(df_retreived_transformed).count()
    assert (df_original_transformed.count() == rows_in_common)

    # Print results for visual inspection
    print("\n")
    print("test__RegexSubstituter: The following should show sentences broken into words")
    df_retreived_transformed.show(truncate=False)

    # If we make it this far without crashing we pass (plus I'm visually reviewing results)
    assert True

def test__TokenSubstituter(numbers_dataframe):

    # Create the transformer
    tokenizer = ct.NLTKWordPunctTokenizer(inputCol="text", outputCol="tokens")

    # Create the transformer
    tokenMatchers= ['two',
                    'four',
                    'nine']
    substitutions= ['two-sub',
                    'four-sub',
                    'nine-sub']
    toksub = ct.TokenSubstituter(inputCol="tokens", outputCol="swapped_tokens", tokenMatchers=tokenMatchers, substitutions=substitutions)

    # Create a pipeline from the transformer
    pipeline = Pipeline(stages=[tokenizer, toksub])


    # fit the test data (which also builds the pipeline)
    model = pipeline.fit(numbers_dataframe)

    # Test the pipeline
    df_original_transformed = model.transform(numbers_dataframe)

    # Delete any previously save model (if it exists)
    # (There may be a more elegant way to do this)
    if os.path.exists("unit_test_model"):
        os.system("rm -rf unit_test_model")

    # Log the model and performance
    save_model(model, "unit_test_model")
    retrieved_model = load_model("unit_test_model")
    df_retreived_transformed = retrieved_model.transform(numbers_dataframe)

    # Assert the retrieved model give the same results as the saved model
    rows_in_common = df_original_transformed.intersect(df_retreived_transformed).count()
    assert (df_original_transformed.count() == rows_in_common)

    # Print results for visual inspection
    print("\n")
    print("test__TokenSubstituter: two, four, and nine should be substituted")
    df_retreived_transformed.show(truncate=False)

    # If we make it this far without crashing we pass (plus I'm visually reviewing results)
    assert True

def test__SentenceSplitter(simple_test_dataframe):

    # Create the transformer
    transformer = ct.SentenceSplitter(inputCol="text", outputCol="sentences")

    # Create a pipeline from the transformer
    pipeline = Pipeline(stages=[transformer])

    # fit the test data (which also builds the pipeline)
    model = pipeline.fit(simple_test_dataframe)

    # Test the pipeline
    df_original_transformed = model.transform(simple_test_dataframe)

    # Delete any previously save model (if it exists)
    # (There may be a more elegant way to do this)
    if os.path.exists("unit_test_model"):
        os.system("rm -rf unit_test_model")

    # Log the model and performance
    save_model(model, "unit_test_model")
    retrieved_model = load_model("unit_test_model")
    df_retreived_transformed = retrieved_model.transform(simple_test_dataframe)

    # Assert the retrieved model give the same results as the saved model
    rows_in_common = df_original_transformed.intersect(df_retreived_transformed).count()
    assert (df_original_transformed.count() == rows_in_common)

    # Print results for visual inspection
    print("\n")
    print("test__SentenceSplitter: The following should show text broken into sentences")
    df_retreived_transformed.show(truncate=False)

    # If we make it this far without crashing we pass (plus I'm visually reviewing results)
    assert True

def test__LevenshteinSubstituter(numbers_dataframe):

    # Create the transformer
    tokenizer = ct.NLTKWordPunctTokenizer(inputCol="text", outputCol="tokens")

    # Create the transformer
    tokenMatchers= ['two1',
                    'four2',
                    'nineee']
    toksub = ct.LevenshteinSubstituter(inputCol="tokens", outputCol="swapped_tokens", tokenMatchers=tokenMatchers, levenshteinThresh=1)

    # Create a pipeline from the transformer
    pipeline = Pipeline(stages=[tokenizer, toksub])


    # fit the test data (which also builds the pipeline)
    model = pipeline.fit(numbers_dataframe)

    # Test the pipeline
    df_original_transformed = model.transform(numbers_dataframe)

    # Delete any previously save model (if it exists)
    # (There may be a more elegant way to do this)
    if os.path.exists("unit_test_model"):
        os.system("rm -rf unit_test_model")

    # Log the model and performance
    save_model(model, "unit_test_model")
    retrieved_model = load_model("unit_test_model")
    df_retreived_transformed = retrieved_model.transform(numbers_dataframe)

    # Assert the retrieved model give the same results as the saved model
    rows_in_common = df_original_transformed.intersect(df_retreived_transformed).count()
    assert (df_original_transformed.count() == rows_in_common)

    # Print results for visual inspection
    print("\n")
    print("test__LevenshteinSubstituter: two and four shold be substituted and nine should not")
    df_retreived_transformed.show(truncate=False)

    # If we make it this far without crashing we pass (plus I'm visually reviewing results)
    assert True

def test__GoWordFilter(numbers_dataframe):

    # Create the transformer
    tokenizer = ct.NLTKWordPunctTokenizer(inputCol="text", outputCol="tokens")

    # Create the transformer
    goWords= ['two','four','eight','nine']
    toksub = ct.GoWordFilter(inputCol="tokens", outputCol="go_word_filtered_tokens", goWords=goWords)

    # Create a pipeline from the transformer
    pipeline = Pipeline(stages=[tokenizer, toksub])


    # fit the test data (which also builds the pipeline)
    model = pipeline.fit(numbers_dataframe)

    # Test the pipeline
    df_original_transformed = model.transform(numbers_dataframe)

    # Delete any previously save model (if it exists)
    # (There may be a more elegant way to do this)
    if os.path.exists("unit_test_model"):
        os.system("rm -rf unit_test_model")

    # Log the model and performance
    save_model(model, "unit_test_model")
    retrieved_model = load_model("unit_test_model")
    df_retreived_transformed = retrieved_model.transform(numbers_dataframe)

    # Assert the retrieved model give the same results as the saved model
    rows_in_common = df_original_transformed.intersect(df_retreived_transformed).count()
    assert (df_original_transformed.count() == rows_in_common)

    # Print results for visual inspection
    print("\n")
    print("test__GoWordFilter: two, four, eight, nine should be the only tokesn left")
    df_retreived_transformed.show(truncate=False)

    # If we make it this far without crashing we pass (plus I'm visually reviewing results)
    assert True