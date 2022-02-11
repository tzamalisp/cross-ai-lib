import re
from emosent import get_emoji_sentiment_rank
import nltk.data
import math
from itertools import groupby
from collections import Counter
from crossai.nlp.text_processing.preprocessing import strip_urls_entities_emojis, strip_emojis
from crossai.nlp.text_processing.preprocessing import clean_up_punctuation_main, strip_accents
from crossai.nlp.text_processing.preprocessing import clean_up_digits_main, tokenizing
# TTR
from lexicalrichness import LexicalRichness
# tokenizer
from nltk.tokenize import TweetTokenizer, word_tokenize
# sentiment analysis
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load('el_core_news_sm')  # load the greek corpus
spacy_text_blob = SpacyTextBlob()
nlp.add_pipe(spacy_text_blob)


def text_length(text):
    """
    # get text length
    :param text:
    :return:
    """
    text_list = text.split()
    text = "".join(text_list)
    return len(text)


def emojis_sentiments_sum(emojis_list):
    neg_sentiment_list = []
    neu_sentiment_list = []
    pos_sentiment_list = []
    if len(emojis_list) > 0:
        for emoji in emojis_list:
            try:
                neg = get_emoji_sentiment_rank(emoji)["negative"] / get_emoji_sentiment_rank(emoji)["occurrences"]
                neu = get_emoji_sentiment_rank(emoji)["neutral"] / get_emoji_sentiment_rank(emoji)["occurrences"]
                pos = get_emoji_sentiment_rank(emoji)["positive"] / get_emoji_sentiment_rank(emoji)["occurrences"]
            except:
                neg = 0
                neu = 0
                pos = 0
            neg_sentiment_list.append(neg)
            neu_sentiment_list.append(neu)
            pos_sentiment_list.append(pos)

    sentiment_sums = {
        "neg": sum(neg_sentiment_list),
        "neu": sum(neu_sentiment_list),
        "pos": sum(pos_sentiment_list)
    }

    return sentiment_sums


def entropy(string):
    """
    Calculates the Shannon entropy of a string
    :param string:
    :return:
    """
    # get probability of chars in string
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    # calculate the entropy
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])

    return entropy


def entropy_ideal(length):
    """
    Calculates the ideal Shannon entropy of a string with given length
    :param length:
    :return:
    """
    prob = 1.0 / length

    return -1.0 * length * prob * math.log(prob) / math.log(2.0)


def average_word_length_sentence(text):
    total_length = sum(len(strip_urls_entities_emojis(word)) for sentence in text for word in sentence.split())
    num_words = sum(len(sentence.split()) for sentence in text)
    if num_words == 0:
        average = 0
    else:
        average = total_length/num_words

    return average


def average_word_length(text):
    cleaned_words = [strip_urls_entities_emojis(w) for w in (w for l in text for w in l.split())]
    if len(cleaned_words) == 0:
        average = 0
    else:
        average = sum(map(len, cleaned_words)) / len(cleaned_words)

    return average


def avg_chars_per_sentence(clean_text, nltk_language="greek"):
    """
    Calculates the average number of characters per sentence in a text.

    Args:
        clean_text: (str) The cleaned text from urls and entities
        nltk_language: (str) The language related to NLTK package.

    Returns:
        The average number of characters per sentence in text.
    """
    tokenizer = nltk.data.load('tokenizers/punkt/{}.pickle'.format(nltk_language))
    sentences = tokenizer.tokenize(clean_text)
    sentences_num_chars_list = []
    for sentence in sentences:
        # remove punctuations from sentence and rebuild again the sentence
        sentence_clean = " ".join(clean_up_punctuation_main(sentence, mode="word"))
        sentences_num_chars_list.append(len(sentence_clean))

    if len(sentences_num_chars_list) == 0:
        average = 0
    else:
        average = sum(sentences_num_chars_list) / len(sentences_num_chars_list)

    return average


def count_vowels_consonants(clean_text, vowels="αεηιυοω", consonants="βγδζθκλμνξπρστφχψ"):
    """
    Count the number of vowels in a text.
    Args:
        clean_text: (str) The cleaned text from urls and entities
        vowels: (str) The vowels of a language. Default: Greek
        consonants: (str) The consonants of a language. Default: Greek
    Returns:

    """
    clean_text = strip_accents(clean_text.lower())
    count_vowels = 0
    count_consonants = 0
    for char in clean_text:
        if char in vowels:
            count_vowels += 1
        if char in consonants:
            count_consonants += 1

    vowels_consonants_dict = {
        "vowels": count_vowels,
        "consonants": count_consonants
    }

    return vowels_consonants_dict


def count_consecutive_chars(clean_text, vowels="αεηιυοωaeiou", mode="tweet"):
    """
    Counts consecutive characters found in a string. These are, simple characters, consecutive vowels, and consecutive
    found consonants.
    Args:
        clean_text: (str) The clean text without the entities.
        vowels: (str) The vowels to be excluded or included to the counting.
        mode: The tokenizer mode. Default: "tweet". Other tokenizers: "tweet_extend", "word"

    Returns:

    """
    # remove accents and lowercase the text
    clean_text = strip_accents(clean_text.lower())
    # clean up punctuations and return a string
    clean_text = " ".join(clean_up_punctuation_main(clean_text, mode))
    # clean up all the digits and return a list
    clean_text_list = clean_up_digits_main(clean_text)
    consonants_list = []
    vowels_list = []
    for item in clean_text_list:
        consonants_list.append("".join(re.findall(r"[^{}]+".format(vowels), item.lower())))
        vowels_list.append("".join(re.findall(r"[{}]+".format(vowels), item.lower())))

    consonants_lengths = [len(word) for word in consonants_list]
    vowels_list = [len(word) for word in vowels_list]

    # return the max number of identical repetitions found in the cleaned text greater than a given value.
    n_identical_repetitions = repetition_identical_cons_chars(clean_text)

    return {
        "max_seq_vowels": max(vowels_list),
        "max_seq_consonants": max(consonants_lengths),
        "n_identical_repetitions": n_identical_repetitions
        }


def chars_various_feats(clean_text):
    """

    Args:
        clean_text: (str) The cleaned text from urls and entities.

    Returns:

    """
    n_uppercase = sum(1 for c in clean_text if c.isupper())
    n_lowercase = sum(1 for c in clean_text if c.islower())
    n_digits = sum(c.isdigit() for c in clean_text)
    n_letters = sum(c.isalpha() for c in clean_text)
    if n_digits == 0:
        ratio_letters_digits = 1
    else:
        ratio_letters_digits = n_letters / n_digits
    chars_feats_dict = {
        "n_uppercase": n_uppercase,
        "n_lowercase": n_lowercase,
        "n_digits": n_digits,
        "n_letters": n_letters,
        "ratio_letters_digits": ratio_letters_digits
    }

    return chars_feats_dict


def nlp_post_processing(clean_text):
    """

    Args:
        clean_text: (str) The cleaned text from urls and entities.

    Returns:

    """
    # lexical richness of the text
    lex = LexicalRichness(clean_text)

    # export spaCy features from text
    doc = nlp(clean_text)
    # Part of Speech Tagging
    token_list = []
    pos_list = []
    lemma_list = []
    for token in doc:
        # uncomment the next two lines to see the tokens' features
        # print("Token: {}, Tag: {}, POS: {}, Explain: {}"
        #       .format(token, token.tag_, token.pos_, spacy.explain(token.tag_)))
        token_list.append(token.text)
        pos_list.append(token.pos_)
        lemma_list.append(token.lemma_)

    # PoS list to counted objects dictionary
    nlp_post_process_dict = dict(Counter(pos_list))
    # Type of Token Ratio (TTR) of the text
    nlp_post_process_dict["tokens"] = token_list
    nlp_post_process_dict["lemmas"] = lemma_list
    nlp_post_process_dict["TTR"] = lex.ttr
    # sentiment retrieval
    nlp_post_process_dict["sentiment_polarity"] = doc._.sentiment.polarity
    nlp_post_process_dict["sentiment_subjectivity"] = doc._.sentiment.subjectivity

    return nlp_post_process_dict


def count_entities(text, tokenizer=TweetTokenizer(strip_handles=False, reduce_len=False)):
    """
    Counts the entities of the text (i.e. hashtags, user mentions, links)

    Args:
        text: (str) The raw text.
        tokenizer: The tokenizer object. Default: TweetTokenizer()

    Returns:
        The clean text without the entities (i.e. hashtags, mentions, \n).
    """
    tokens = tokenizer.tokenize(text)
    entities = [tok for tok in tokens if tok.startswith(('#', '@', 'http', 'https'))]
    return len(entities)


def repetition_identical_cons_chars(clean_text, repetition=3):
    """
    Counts the number of repetitions of an identical char that it is found in a string greater than a value.

    Args:
        clean_text: (str) The cleaned text from urls, entities, punctuations, and digits.
        repetition: (int) The number of times an identical char is repetitive.

    Returns:
        The number of repetitions that are found and are greater than a value.
    """
    chars_repetitions_list = [len(list(j)) for _, j in groupby(clean_text)]
    return sum(i > repetition for i in chars_repetitions_list)
