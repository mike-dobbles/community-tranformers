import nltk
from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
# Available in PySpark >= 2.3.0
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable, JavaMLReader, MLReadable, MLWritable
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import re

# Credits https://stackoverflow.com/a/52467470
# by https://stackoverflow.com/users/234944/benjamin-manns

class NLTKWordPunctTokenizer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable, MLReadable, MLWritable):
    """This class uses nltk.tokenize.wordpunct_tokenize to generate an output column tranformation of a spark dataframe."""

    stopwords = Param(Params._dummy(), "stopwords", "stopwords",
                      typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, stopwords=None):
        super(NLTKWordPunctTokenizer, self).__init__()
        module = __import__("__main__")
        setattr(module, 'NLTKWordPunctTokenizer', NLTKWordPunctTokenizer)
        self.stopwords = Param(self, "stopwords", "")
        self._setDefault(stopwords=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, stopwords=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setStopwords(self, value):
        return self._set(stopwords=list(value))

    def getStopwords(self):
        return self.getOrDefault(self.stopwords)

    # Required in Spark >= 3.0
    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    # Required in Spark >= 3.0
    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def _transform(self, dataset):
        stopwords = set(self.getStopwords())

        def f(s):
            tokens = nltk.tokenize.wordpunct_tokenize(s)
            return [t for t in tokens if t.lower() not in stopwords]

        t = ArrayType(StringType())
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))


class RegexSubstituter(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable, MLReadable,
                       MLWritable):
    """This class generates an output column tranformation of the input spark dataframe where the input column string
    is searched with regex and a substitution is made where matches are found."""

    regexMatchers = Param(Params._dummy(), "regexMatchers", "regexMatchers",
                      typeConverter=TypeConverters.toListString)
    substitutions = Param(Params._dummy(), "substitutions", "substitutions",
                      typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, regexMatchers=None, substitutions=None):
        module = __import__("__main__")
        setattr(module, 'RegexSubstituter', RegexSubstituter)
        super(RegexSubstituter, self).__init__()
        self.regexMatchers = Param(self, "regexMatchers", "")
        self.substitutions = Param(self, "substitutions", "")
        self._setDefault(regexMatchers=[],substitutions=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, regexMatchers=None, substitutions=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setRegexMatchers(self, value):
        self._paramMap[self.regexMatchers] = value
        return self

    def getRegexMatchers(self):
        return self.getOrDefault(self.regexMatchers)

    def setSubstitutions(self, value):
        self._paramMap[self.substitutions] = value
        return self

    def getSubstitutions(self):
        return self.getOrDefault(self.substitutions)

    def _transform(self, dataset):
        regexMatchers = self.getRegexMatchers()
        substitutions = self.getSubstitutions()

        # throw error if the regex patterns and substitution arrays don't match
        if len(substitutions) != len(regexMatchers):
            raise ValueError("regexMatchers and substitutions must be the same length")

        # user defined function to loop through each of the substitutions and apply
        # them to the passed text
        t = StringType()

        def f(text):
            for idx, reg in enumerate(regexMatchers):
                text = re.sub(reg, substitutions[idx], text)
            return text

        # Select the input column
        in_col = dataset[self.getInputCol()]

        # Get the name of the output column
        out_col = self.getOutputCol()

        return dataset.withColumn(out_col, udf(f, t)(in_col))

class TokenSubstituter(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable,
                       MLReadable, MLWritable):
    """This class expects the input column to have an array of string tokens.  It searches those tokens one by one
    and replaces them with the substitutions if a match is found."""

    tokenMatchers = Param(Params._dummy(), "tokenMatchers", "tokenMatchers",
                          typeConverter=TypeConverters.toListString)
    substitutions = Param(Params._dummy(), "substitutions", "substitutions",
                          typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, tokenMatchers=None, substitutions=None):
        module = __import__("__main__")
        setattr(module, 'TokenSubstituter', RegexSubstituter)
        super(TokenSubstituter, self).__init__()
        self.tokenMatchers = Param(self, "tokenMatchers", "")
        self.substitutions = Param(self, "substitutions", "")
        self._setDefault(tokenMatchers=[], substitutions=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, tokenMatchers=None, substitutions=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setTokenMatchers(self, value):
        self._paramMap[self.tokenMatchers] = value
        return self

    def getTokenMatchers(self):
        return self.getOrDefault(self.tokenMatchers)

    def setSubstitutions(self, value):
        self._paramMap[self.substitutions] = value
        return self

    def getSubstitutions(self):
        return self.getOrDefault(self.substitutions)

    def _transform(self, dataset):
        tokenMatchers = self.getTokenMatchers()
        substitutions = self.getSubstitutions()

        # throw error if the regex patterns and substitution arrays don't match
        if len(substitutions) != len(tokenMatchers):
            raise ValueError("tokenMatchers and substitutions must be the same length")

        # user defined function to loop through each of the substitutions and apply
        # them to the passed text
        t = ArrayType(StringType())

        def f(token_array):
            # Cycle through the tokens in the passed column cell one by one
            # If it matches a token in the tokenMatchers array, then swap on that token,
            # otherwise, leave it alone
            returned_array = []
            for tok in token_array:

                #See if we can find the token and if we do swap it for the token in the same position in subsitutions
                try:
                    idx = tokenMatchers.index(tok)
                    returned_array.append(substitutions[idx])
                except:
                    returned_array.append(tok)

            return returned_array

        # Select the input column
        in_col = dataset[self.getInputCol()]

        # Get the name of the output column
        out_col = self.getOutputCol()

        return dataset.withColumn(out_col, udf(f, t)(in_col))