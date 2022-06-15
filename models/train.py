"""
@author: noobrainers
@contact: Bryan T. Kim bryantaekim at gmail dot com
          Daniel Lee dslee47 at gmail dot com
          Juswaldy Jusman juswaldy at gmail dot com
@description: The main script for training our models
@content:
    def ner()
    def clustr()
    def tomo()
    def tomo_wholeshebang()
"""

import argparse
import numpy as np 
import pandas as pd 
from pandas import DataFrame
from typing import Tuple
import json
import os
from timeit import default_timer as timer
import tensorflow as tf


""" Configs """

class Configs:
    def __init__(self, **kwargs):
        # NER.
        self.ner = 'super awesome'

        # Hierarchical Clustering.
        self.clustr = 'super cool'

        # Topic Modeling.
        self.num_topics = 40                # Final number of topics after reduction
        self.article_words_threshold = 200  # Minimum number of article words to be considered for training
        self.num_latest_articles = 1000     # The number of the latest articles (by date) to use for testing/demo
        self.training_time = 'deep-learn'   # How long to train the model: 'fast-learn', 'learn', or 'deep-learn'
        self.random_state = 52              # Random state for reproducibility


""" Argument parsing and checking """
def parse_args() -> argparse.Namespace:
    desc = 'Train a model for our GLG capstone project'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--fn', type=str, default='tomo', help='ner, clustr, or tomo', required=True)
    parser.add_argument('--modelname', type=str, default='universal-sentence-encoder', help='Which model are we working on?', required=False)
    parser.add_argument('--action', type=str, default='train', help='train, test, inference', required=True)
    parser.add_argument('--trainfile', type=str, default='./data/health_tech_time.csv', help='Path to the training file', required=False)
    parser.add_argument('--testfile', type=str, default='./data/health_tech_time_test.csv', help='Path to the test file', required=False)
    parser.add_argument('--usebigrams', action='store_true', help='Use bigrams?', required=False)
    parser.add_argument('--wholeshebang', action='store_true', help='Do the whole-shebang for the specified model?', required=False)
    parser.add_argument('--outputfolder', type=str, default='./models', help='Output folder', required=False)
    parser.add_argument('--outputfile', type=str, default='z.pkl', help='Output filename', required=False)
    return check_args(parser.parse_args())

def check_args(args: argparse.Namespace) -> argparse.Namespace:
	return args


""" Helper functions """


""" Named Entity Recognition """
def ner():
    return 'Almost there!', 0.0


""" Hierarchical Clustering """
def clustr():
    return 'Wait for it!', 0.0


""" Topic Modeling """
def tomo(action: str,
         df: DataFrame,
         col: str,
         model_name: str = 'universal-sentence-encoder',
         use_phrases: bool = False,
         speed: str = 'learn',
         workers: int = 8) -> Tuple[object, float]:
    """Train topic modeling using Top2Vec.

    Parameters
    ----------
    action : str
        The action to perform. Either 'train', 'test', or 'inference'

    df : DataFrame
        The dataframe for training/testing/inference

    col : str
        'title_clean' or 'article_clean'

    model_name : str
        'universal-sentence-encoder' or 'universal-sentence-encoder-large'

    use_phrases : bool
        Enable ngram_vocab if specified

    speed : str
        'fast-learn', 'learn', or 'deep-learn'. default='learn'

    workers : int
        Number of worker threads

    Returns
    -------
    model, float
        The trained model, elapsed time in float seconds
    """
    from top2vec import Top2Vec

    start = timer()

    # If articles, apply threshold.
    if col == 'article_clean':
        df = df.loc[df.num_words_article > config.article_words_threshold]

    # Pick only data rows that are strings.
    df = df[['id', col]]
    df = df[df[col].apply(type)==str]
    
    ## Segmentation fault if > 90k. (??)
    #if col == 'title_clean':
    #    df = df.sample(n=90000, random_state=config.random_state)

    print('-'*80, flush=True)
    print(f'{col}', len(df), flush=True)

    #try:
    # Train model.
    model = Top2Vec(
        documents=df[col].tolist(),
        document_ids=df['id'].tolist(),
        keep_documents=False,
        ngram_vocab=use_phrases,
        embedding_model=model_name,
        speed=speed,
        workers=workers
    )
    #except:
    #    # Set to None if failed.
    #    model = None

    finish = timer()

    return model, (finish-start)

""" Topic Modeling Do Everything in One Go """
def tomo_wholeshebang(args: argparse.Namespace) -> Tuple[DataFrame, float]:
    """Train for years, quarters, seasons, months, yearseasons, yearmonths, and weekday/weekend.
    First, train and save the main "everything" model, to be versioned and deployed.
    Next, train each period, and collect the returned topics into a dataframe, and save it.
    Warning: This might take more than 3 days :p (Order of magnitude may vary)

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the script
    
    Returns
    -------
    DataFrame, float
        The dataframe of topics, and the elapsed time in float seconds
    """
    from top2vec import Top2Vec

    # Read in the data.
    source_df = pd.read_csv(args.trainfile)

    # Remove latest articles from training.
    training_df = source_df.sort_values('date').iloc[:-config.num_latest_articles]
    
    # Setup the training grid.
    models = { 'small': 'universal-sentence-encoder', 'large': 'universal-sentence-encoder-large' }
    cols = { 'articles': 'article_clean' } #, 'titles': 'title_clean' } # Why are titles causing segmentation fault?
    phrasings = { 'single': False, 'ngram': True }

    # Go through grid.
    for model_size, model_name in models.items():
        for col_name, col in cols.items():
            for phrasing, use_phrases in phrasings.items():

                # Setup result rows for saving to csv later.
                result_rows = []

                # Name the model file and print it.
                model_file = f'{args.outputfolder}/{args.fn}-{col_name}-{phrasing}-{model_name}.t2v'
                print('-'*80)
                print(model_file)

                # Only go through with this if the model file doesn't exist.
                if not os.path.exists(model_file):
                    # Train for the currennt grid cell.
                    model, _ = tomo(
                        action='train',
                        df=training_df,
                        col=col,
                        model_name=model_name,
                        use_phrases=use_phrases,
                        speed=config.training_time,
                        workers=8)

                    # If no error, perform topic reduction, and save it.
                    # Also, collect the topics discovered and append it to the result rows.
                    if model:
                        model_num_topics = model.get_num_topics()
                        model.hierarchical_topic_reduction(num_topics=config.num_topics if model_num_topics > config.num_topics else model_num_topics)
                        model.save(model_file)
                        topic_wordss, word_scoress, topic_ids = model.get_topics(num_topics=config.num_topics, reduced=True)
                        for topic_id, topic_words, word_scores in zip(topic_ids, topic_wordss, word_scoress):
                            result_rows.append([model_size, col_name, phrasing, 'all', topic_id, topic_words, word_scores])

                    # Also train for each time period.
                    for period in ['year', 'quarter', 'season', 'month', 'yearseason', 'yearquarter', 'yearmonth', 'isweekend']:
                        for x in training_df[period].value_counts().index: 
                            # Filter the dataframe for the chosen period and print it.
                            df = training_df.loc[training_df[period] == x]
                            print('-'*40)
                            print(f'{model_size}-{col_name}-{phrasing}-{period}-{x}')

                            # Train for the current grid cell, chosen period.
                            model, _ = tomo(
                                action='train',
                                df=df,
                                col=col,
                                model_name=model_name,
                                use_phrases=use_phrases,
                                speed=config.training_time,
                                workers=8)

                            # If no error, collect the topics discovered and append it to the result rows.
                            if model:
                                model_num_topics = model.get_num_topics()
                                topic_wordss, word_scoress, topic_ids = model.get_topics(num_topics=config.num_topics if model_num_topics > config.num_topics else model_num_topics)
                                for topic_id, topic_words, word_scores in zip(topic_ids, topic_wordss, word_scoress):
                                    result_rows.append([model_size, col_name, phrasing, f'{period}-{x}', topic_id, topic_words, word_scores])

                    # Save the result rows to csv file.
                    result_file = f'data/{args.fn}-{col_name}-{phrasing}-{model_name}.csv'
                    result_df = DataFrame(result_rows, columns=['model_size', 'col_name', 'phrasing', 'period', 'topic_id', 'topic_words', 'word_scores'])
                    result_df.to_csv(result_file, index=False)

def get_topn(model, infile, col, n):
    dfin = pd.read_csv(infile)
    results = []
    chunk = 0
    for i, x in dfin.iterrows():
        # Get the top n topics for the current document.
        topics_words, words_scores, topic_scores, topic_nums = model.query_topics(
            query=str(x[col]),
            num_topics=n
        )
        results.append({'id': x.id, f'top{n}': topic_nums, f'top{n}_scores': topic_scores})

        # Display every 1k row.
        if i % 1000 == 0:
            print(f'Processing #{i}', flush=True)

        # Save every 200k rows.
        if i > 0 and i % 200000 == 0:
            chunk += 1
            dfout = pd.DataFrame(results, columns=['id', f'top{n}', f'top{n}_scores'])
            basename = os.path.splitext(infile)[0]
            outfile = '{}-{:02}.csv'.format(basename, chunk)
            print(f'############## Saving {outfile}', flush=True)
            dfout.to_csv(outfile, index=None)
            results = []

"""main"""
def main():

    args = parse_args()
    if args is None:
        print('Problem!')
        exit()

    if args.fn == 'ner':
        ner()
    elif args.fn == 'clustr':
        clustr()
    elif args.fn == 'tomo':
        if args.wholeshebang:
            # Repurpose for temporal analysis.
            # tomo_wholeshebang(args=args)

            from top2vec import Top2Vec

            # Load article model.
            model = Top2Vec.load('models/tomo-60k.pkl')

            # Run for every title.
            get_topn(model, 'data/news2.7m-gensim-titles.csv', 'title_clean', 3)

            # Run for every article.
            get_topn(model, 'data/news2.7m-gensim-articles.csv', 'article_clean', 3)

        else:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logical_gpus = tf.config.list_logical_devices('GPU')
                        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    print(e)

            df = pd.read_csv(args.trainfile, low_memory=False)
            model, _ = tomo(
                action=args.action,
                df=df,
                col='title_clean',
                model_name=args.modelname,
                use_phrases=args.usebigrams,
                speed='deep-learn',
                workers=8
            )
            if model:
                model_num_topics = model.get_num_topics()
                model.hierarchical_topic_reduction(num_topics=config.num_topics if model_num_topics > config.num_topics else model_num_topics)
                topic_wordss, word_scoress, topic_ids = model.get_topics(num_topics=config.num_topics, reduced=True)
                for topic_words, word_scores, topic_id in zip(topic_ids, topic_wordss, word_scoress):
                    print(f'{topic_id} {topic_words} {word_scores}')
                model.save('{}/{}'.format(args.outputfolder, args.outputfile))


if __name__ == '__main__':
    config = Configs()
    main()
