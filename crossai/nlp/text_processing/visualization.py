import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud


def word_cloud_creation(text_list, max_words=150):
    """

    Args:
        text_list:
        max_words:

    Returns:

    """
    count_all_terms = Counter()
    for tokens in text_list:
        count_all_terms.update(tokens)
    print('Most 10 frequently used terms:')
    print(count_all_terms.most_common(10))
    word_cloud = WordCloud(width=800, height=500, max_font_size=110,
                           random_state=100, max_words=max_words).generate_from_frequencies(count_all_terms)
    plt.figure(figsize=(10, 7))
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
