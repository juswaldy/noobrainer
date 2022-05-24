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
