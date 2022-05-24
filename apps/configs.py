wordcloud_backgrounds = [
    'ivory',
    'whitesmoke',
    'snow',
    'floralwhite',
    'honeydew',
    'azure',
    'beige',
    'mintcream',
    'oldlace',
    'ghostwhite'
]

model_about_fields = [
    'Source',
    'URL',
    'Preprocessing',
    'Training filename',
    'Training column',
    'Number of rows',
    'Using ngrams',
    'Number of topics discovered',
    'Topic reduction'
]

model_about_values = {
    'models/tomo-60k.pkl': [
        'News 2.7M',
        'https://components.one/datasets/all-the-news-2-news-articles-dataset/',
        'gensim preprocessor',
        '0_combined_set_60k_date.csv',
        'title_clean',
        '60,000',
        'No',
        '358',
        '40'
    ],
    'models/tomo-healthtech-titles-single-17.pkl': [
        'News 2.7M',
        'https://components.one/datasets/all-the-news-2-news-articles-dataset/',
        'Bryan Kim\'s preprocessor',
        'health_tech.csv',
        'title_clean',
        '129,682',
        'No',
        '582',
        '40'
    ],
    'models/tomo-healthtech-articles-single-17.pkl': [
        'News 2.7M',
        'https://components.one/datasets/all-the-news-2-news-articles-dataset/',
        'Bryan Kim\'s preprocessor',
        'health_tech.csv',
        'article_clean',
        '129,682',
        'No',
        '696',
        '40'
    ],
    'models/tomo-all-87k-articles-single-21.pkl': [
        'News 2.7M',
        'https://components.one/datasets/all-the-news-2-news-articles-dataset/',
        'gensim.simple_preprocessing',
        'news2.7m-gensim-articles.csv',
        'article_clean',
        '87,693',
        'No',
        '529',
        '40'
    ]
}

topic_labels = {
    0: 'o',
    1: 'h',
    2: 'o',
    3: 'o',
    4: 't',
    5: 't',
    6: 'o',
    7: 'o',
    8: 'h',
    9: 't',
    10: 't',
    11: 'o',
    12: 'h',
    13: 'o',
    14: 't',
    15: 'o',
    16: 't',
    17: 'o',
    18: 'o',
    19: 'o',
    20: 'o',
    21: 't',
    22: 't',
    23: 't',
    24: 't',
    25: 't',
    26: 'o',
    27: 'o',
    28: 't',
    29: 'h',
    30: 'o',
    31: 'o',
    32: 'o',
    33: 'h',
    34: 'o',
    35: 'o',
    36: 'h',
    37: 'o',
    38: 't',
    39: 't'
}