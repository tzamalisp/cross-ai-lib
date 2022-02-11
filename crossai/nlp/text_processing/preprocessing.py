import logging
import re
import urllib
import string
import unicodedata
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.corpus import stopwords
from spacy.lang.el.stop_words import STOP_WORDS
from pymongo import MongoClient


def tokenizing(text, mode="tweet"):
    """
    Creates tokens based on the mode you choose. Note that the "tweet_extend" mode does not work properly in Greek
    where the TweetTokenizer() continues cutting the words with repetitive chars greater than 3 in some words.
    Args:
        text: (str) The raw text.
        mode: (str) The mode of tokenizer object. Default: "tweet". Other accepted: "tweet_extend", "word".

    Returns:
        List of tokens.
    """
    if mode == "tweet":
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(text)
    elif mode == "tweet_extend":
        tokenizer = TweetTokenizer(strip_handles=False, reduce_len=False)
        tokens = tokenizer.tokenize(text)
    elif mode == "word":
        tokens = word_tokenize(text)
    else:
        logging.error("Please define a valid tokenizer.")
        raise

    return tokens


def clean_up_punctuation_main(text, mode="tweet"):
    """
    source: stackoverflow
    I'm assuming you REQUIRE this function as per your assignment
    otherwise, just substitute str.strip(string.punctuation) anywhere
    you'd otherwise call clean_up(str)

    Args:
        text: (str) The raw text.
        mode: The tokenizer mode. Default: "tweet". Other tokenizers: "tweet_extend", "word"

    Returns:
        The text without punctuations.
    """
    tokens = tokenizing(text, mode)
    punctuation_list = list(string.punctuation)
    special_words = ["rt", "via", "…", "...", ".."]
    punctuations = punctuation_list + special_words
    return [tok for tok in tokens if tok not in punctuations]


def clean_up_digits_main(text, mode="tweet"):
    """
    source: stackoverflow
    I'm assuming you REQUIRE this function as per your assignment
    otherwise, just substitute str.strip(string.punctuation) anywhere
    you'd otherwise call clean_up(str)

    Args:
        text: (str) The raw text.
        mode: (str) The mode of tokenizer object. Default: "tweet". Other accepted: "tweet_extend", "word".

    Returns:
        The text without digits.
    """
    tokens = tokenizing(text, mode)
    capture_float_comma = re.compile(r'\d+(?:\,\d*)')
    capture_float_period = re.compile(r'\d+(?:\.\d*)')
    tokens_pure = [ele for ele in tokens if not capture_float_comma.match(ele) and not capture_float_period.match(ele)]
    return [tok for tok in tokens_pure if not tok.isdigit()]


def clean_up_punctuation_digits(text):
    """
    Clean up the text from punctuation and digits contained in it.
    Args:
        text: (str) The raw text.

    Returns:
        The text without the punctuations and the digits.
    """
    punctuation = string.punctuation + string.digits
    return text.strip(punctuation)


def clean_up_punctuation_simple(text, punctuation="!\"',;:.-?)([]<>*#\n\\"):
    """

    Args:
        text: (str) The raw text.
        punctuation: The values of punctuations.

    Returns:
        The text without punctuations.
    """
    return text.strip(punctuation)


def extract_hashtags(text):
    """
    Extracts a list with the hashtags that are included inside a text.
    Args:
        text: (str) The raw text.

    Returns:
        List with the found hashtags.
    """
    return list(set(part for part in text.split() if part.startswith('#')))


def extract_mentions(text):
    """
    Extracts a list with the mentions that are included inside a text.
    Args:
        text: (str) The raw text.

    Returns:
        List with the found mentions.
    """
    return list(set(part for part in text.split() if part.startswith('@')))


def strip_entities(text):
    """
    Strip text from entities like "@" and "#".

    Args:
        text: (str) The raw text.

    Returns:
        The clean text without the entities.
    """
    entity_prefixes = ["@", "#", "\n"]
#     for separator in  string.punctuation:
#         if separator not in entity_prefixes :
#             text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


def strip_urls_entities_emojis(text):
    """
    Strip text from all the URLs and the entities (hashtags, mentions, \n) that are contained inside the text.

    Args:
        text: (str) The raw text.

    Returns:
        The clean text without the entities (i.e. hashtags, mentions, \n).
    """
    # clean urls
    text = re.sub(r"http\S+", "", text)
    # clean emojis
    text = strip_emojis(text)
    # clean from entities: @, #, and \n
    clean_text = strip_entities(text)

    return clean_text


# extract emojis from text
def extract_emojis(text):
    """

    Args:
        text: The raw text.

    Returns:
        List with the found emojis.
    """
    emojis = re.findall(u'[\U0001f600-\U0001f650]', text)
    return emojis


def strip_emojis(text):
    """
    Remove emojis from text. Complete emojis removal list: https://stackoverflow.com/a/49146722/330558

    Args:
        text: (str) The raw text.

    Returns:
        The clean text without the emojis.
    """
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# extract external urls from tweet
def extract_external_urls(text):
    """
     Extracts a list with the URLs that are included inside a text.
    Args:
        text: (str) The raw text.

    Returns:
        List with the found URLs.
    """
    urls = re.findall("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", text)
    valid_urls_list = []
    invalid_urls_list = []
    for url in urls:
        try:
            opener = urllib.request.build_opener()
            request = urllib.request.Request(url)
            response = opener.open(request)
            actual_url = response.geturl()
            valid_urls_list.append(actual_url)
        except Exception as e:
            invalid_urls_list.append(url)
            logging.debug("URL is invalid. Error: {}".format(e))

    return valid_urls_list


def extract_words(text, mode="tweet", lang="greek"):
    """
    Process the text of a tweet:
        1. Lowercase
        2. Tokenize
        3. Floats removal (kannot be captured by isdigit())
        4. Stopword removal
        5. Digits removal
        6. remove tokens <= 3 characters
        7. Return list
    Args:
        text: (str) The raw text.
        mode: (str) The mode of tokenizer object. Default: "tweet". Other accepted: "tweet_extend", "word".
        lang: (str) The language of the stopwords from nltk, and spacy libraries.

    Returns:
        List with all the pure tokens.
    """
    # remove emojis
    text_without_emojis = strip_emojis(text)
    # remove all digits in a tweet (and these that are next to characters too)
    # match all digits in the string and replace them by empty string
    pattern = r'[0-9]'
    text_clean = re.sub(pattern, "", text_without_emojis)
    # gather stopwords
    punct = list(string.punctuation)
    stop_words_spacy = list(STOP_WORDS)
    stop_words_nltk = stopwords.words(lang)
    stop_words = stop_words_nltk + stop_words_spacy
    special_words = ["rt", "via", "…", "...", "..", "«", "»", "στις", "τους", "ή"]
    stop_word_list = stop_words + punct + special_words
    # lowercase the characters and tokenize based on the Tokenizer object
    tokens = tokenizing(text_clean.lower(), mode)
    # remove floats that have the comma or period to distinguish the digits
    capture_float_comma = re.compile(r'\d+(?:\,\d*)')
    capture_float_period = re.compile(r'\d+(?:\.\d*)')
    tokens_pure = [ele for ele in tokens if not capture_float_comma.match(ele) and not capture_float_period.match(ele)]

    return [tok for tok in tokens_pure if tok not in stop_word_list and not tok.isdigit() and
            not tok.startswith(('#', '@', 'http', 'https', "\n")) and len(tok) > 3]


def clean_urls_entities(text, mode="tweet"):
    """
    Cleans the text from all the URLs and the entities (hashtags, mentions, \n) that are contained inside the text.
    However, because of the TweetTokenizer it does not the exact cleaning of the text, but it also cuts some characters
    from the long repeated characters in the word (or token)

    Args:
        text: (str) The raw text.
        mode: (str) The mode of tokenizer object. Default: "tweet". Other accepted: "tweet_extend", "word".

    Returns:
        The clean text without the entities (i.e. hashtags, mentions, \n).
    """
    tokens = tokenizing(text, mode)
    return " ".join([tok for tok in tokens if not tok.startswith(('#', '@', 'http', 'https', "\n"))])


def extract_stopwords(text, mode="tweet"):
    """

    Args:
        text: (str) The raw text.
        mode: (str) The mode of tokenizer object. Default: "tweet". Other accepted: "tweet_extend", "word".

    Returns:

    """
    stop_words_spacy = list(STOP_WORDS)
    stop_words_nltk = stopwords.words("greek")
    stop_words = stop_words_nltk + stop_words_spacy
    special_words = ["rt", "via", "…", "...", "«", "»", "στις", "τους", "ή"]
    stop_word_list = stop_words + special_words
    tokens = tokenizing(text.lower(), mode)

    return [tok for tok in tokens if tok in stop_word_list]


def strip_accents(clean_text):
    """

    Args:
        clean_text: (str) The cleaned text.

    Returns:
        Returns a string without the accents.
    """
    # clean_text = " ".join(clean_up_punctuation_main(clean_text))
    return "".join(c for c in unicodedata.normalize('NFD', clean_text) if unicodedata.category(c) != 'Mn')
