# import nltk
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
# Available in PySpark >= 2.3.0
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable, JavaMLReader, MLReadable, MLWritable
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import re
from pyspark.sql import functions as F


# Credits https://stackoverflow.com/a/52467470
# by https://stackoverflow.com/users/234944/benjamin-manns

class NLTKWordPunctTokenizer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable,
                             MLReadable, MLWritable):
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
            # tokens = nltk.tokenize.wordpunct_tokenize(s)
            tokens = wordpunct_tokenize(s)
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
        self._setDefault(regexMatchers=[], substitutions=[])
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
        setattr(module, 'TokenSubstituter', TokenSubstituter)
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

                # See if we can find the token and if we do swap it for the token in the same position in subsitutions
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


class LevenshteinSubstituter(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable,
                             MLReadable, MLWritable):
    """This class expects the input column to have an array of string tokens.  It searches those tokens one by one
    and replaces them with the substitutions if a match is found."""

    tokenMatchers = Param(Params._dummy(), "tokenMatchers", "tokenMatchers",
                          typeConverter=TypeConverters.toListString)
    levenshteinThresh = Param(Params._dummy(), "levenshteinThresh", "levenshteinThresh", typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, tokenMatchers=None, levenshteinThresh=None):
        module = __import__("__main__")
        setattr(module, 'LevenshteinSubstituter', LevenshteinSubstituter)
        super(LevenshteinSubstituter, self).__init__()
        self.tokenMatchers = Param(self, "tokenMatchers", "")
        self.levenshteinThresh = Param(self, "levenshteinThresh", "")
        self._setDefault(tokenMatchers=[], levenshteinThresh=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, tokenMatchers=None, levenshteinThresh=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setTokenMatchers(self, value):
        self._paramMap[self.tokenMatchers] = value
        return self

    def getTokenMatchers(self):
        return self.getOrDefault(self.tokenMatchers)

    def setLevenshteinThresh(self, value):
        self._paramMap[self.levenshteinThresh] = value
        return self

    def getLevenshteinThresh(self):
        return self.getOrDefault(self.levenshteinThresh)

    def _transform(self, dataset):
        tokenMatchers = self.getTokenMatchers()
        levenshteinThresh = self.getLevenshteinThresh()

        def _levenshteinDist(s, t):
            """
            Determines how different two strings are based on the levenshtein algorithm
            Algorithm retrieved from
             https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
            which claims to be based on a version of the algorithm described in the Levenshtein wikipedia article

            :param s: string 1
            :param t: string 2
            :return:  levenshtein edit distance
            """

            if s == t:
                return 0
            elif len(s) == 0:
                return len(t)
            elif len(t) == 0:
                return len(s)
            v0 = [None] * (len(t) + 1)
            v1 = [None] * (len(t) + 1)
            for i in range(len(v0)):
                v0[i] = i
            for i in range(len(s)):
                v1[0] = i + 1
                for j in range(len(t)):
                    cost = 0 if s[i] == t[j] else 1
                    v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
                for j in range(len(v0)):
                    v0[j] = v1[j]

            return v1[len(t)]

        def _first_token_within_edit_distance(original_token, matcher_tokens, edit_dist_thresh):
            """Loops through the token in the matcher token list and if it finds one within the edit distance, it
            returns it"""
            token_within_edit_dist = None
            for matcher_token in matcher_tokens:
                if _levenshteinDist(original_token, matcher_token) <= edit_dist_thresh:
                    return matcher_token

            return token_within_edit_dist

        # user defined function to loop through each of the substitutions and apply
        # them to the passed text
        t = ArrayType(StringType())
        def f(original_token_array):
            # Cycle through the tokens in the passed column cell one by one
            # If the edit distance between the token and tokenMatcher token is below the edit distance
            # threshold then swap the token with the tokenMatcher
            # threshold then swap it out
            updated_token_list = []
            for orginal_tok in original_token_array:
                token_within_edit_dist = _first_token_within_edit_distance(orginal_tok, tokenMatchers, levenshteinThresh)
                if token_within_edit_dist:
                    updated_token_list.append(token_within_edit_dist)
                else:
                    updated_token_list.append(orginal_tok)

            return updated_token_list

        # Select the input column
        in_col = dataset[self.getInputCol()]

        # Get the name of the output column
        out_col = self.getOutputCol()

        return dataset.withColumn(out_col, udf(f, t)(in_col))


class SentenceSplitter(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable, MLReadable,
                       MLWritable):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        module = __import__("__main__")
        setattr(module, 'SentenceSplitter', SentenceSplitter)
        super(SentenceSplitter, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        # User defined function to actually split the text
        text2sents = udf(lambda text: sent_tokenize(text), ArrayType(StringType()))

        # Select the input column
        in_col = dataset[self.getInputCol()]

        # Get the name of the output column
        out_col = self.getOutputCol()

        return dataset.withColumn(out_col, F.explode(text2sents(in_col)))

class GoWordFilter(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable,
                       MLReadable, MLWritable):
    """This class expects the input column to have an array of string tokens.  It searches those tokens one by one
    and replaces them with the substitutions if a match is found."""

    tokenMatchers = Param(Params._dummy(), "goWords", "goWords",
                          typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, goWords=None):
        module = __import__("__main__")
        setattr(module, 'TokenSubstituter', TokenSubstituter)
        super(GoWordFilter, self).__init__()
        self.goWords = Param(self, "goWords", "")
        self._setDefault(goWords=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, goWords=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setGoWords(self, value):
        self._paramMap[self.goWords] = value
        return self

    def getGoWords(self):
        return self.getOrDefault(self.goWords)


    def _transform(self, dataset):
        goWords = self.getGoWords()


        # user defined function to loop through each of the substitutions and apply
        # them to the passed text
        t = ArrayType(StringType())
        def f(original_token_array):
            # Cycle through the tokens in the passed column cell one by one
            # If it matches a token in the tokenMatchers array, then swap on that token,
            # otherwise, leave it alone
            returned_token_array = []
            for tok in original_token_array:
                # Only keep tokens that are in the GoWords list
                if tok in goWords:
                    returned_token_array.append(tok)

            return returned_token_array

        # Select the input column
        in_col = dataset[self.getInputCol()]

        # Get the name of the output column
        out_col = self.getOutputCol()

        return dataset.withColumn(out_col, udf(f, t)(in_col))
