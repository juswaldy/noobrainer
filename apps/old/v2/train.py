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


""" Argument parsing and checking """
def parse_args() -> argparse.Namespace:
	desc = 'Train a model for our GLG capstone project'
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--fn', type=str, default='tomo', help='ner, clustr, or tomo', required=True)
	parser.add_argument('--modelname', type=str, default='universal-sentence-encoder', help='Which model are we working on?', required=False)
	parser.add_argument('--action', type=str, default='train', help='train, test, inference', required=True)
	parser.add_argument('--trainfile', type=str, default='health_tech_time.csv', help='Path to the training file', required=False)
	parser.add_argument('--testfile', type=str, default='health_tech_time_test.csv', help='Path to the test file', required=False)
	parser.add_argument('--wholeshebang', action='store_true', help='Do the whole-shebang for the specified model?', required=False)
	parser.add_argument('--outputfolder', type=str, default='.', help='Output folder', required=False)
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
        df = df.loc[df.num_words_per_article > config.article_words_threshold]

    # Pick only data rows that are strings.
    docs = df[col].tolist()
    docs = [ x for x in docs if type(x) == str ]
    print(f'{col}', len(docs))

    try:
        # Train model.
        model = Top2Vec(
            documents=docs,
            ngram_vocab=use_phrases,
            embedding_model=model_name,
            speed=speed,
            workers=workers
        )
    except:
        # Set to None if failed.
        model = None

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
                model_file = f'{args.outputfolder}/{model_size}-{col_name}-{phrasing}.t2v'
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
                    result_file = f'{args.outputfolder}/{model_size}-{col_name}-{phrasing}.csv'
                    result_df = DataFrame(result_rows, columns=['model_size', 'col_name', 'phrasing', 'period', 'topic_id', 'topic_words', 'word_scores'])
                    result_df.to_csv(result_file, index=False)


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
            tomo_wholeshebang(args=args)
        else:
            df = pd.read_csv(args.trainfile)
            model, _ = tomo(
                action=args.action,
                df=df,
                col='article_clean',
                model_name=args.modelname,
                use_phrases=False,
                speed='deep-learn',
                workers=8
            )
            if model:
                model.save('z.t2v')
                model_num_topics = model.get_num_topics()
                model.hierarchical_topic_reduction(num_topics=config.num_topics if model_num_topics > config.num_topics else model_num_topics)
                topic_wordss, word_scoress, topic_ids = model.get_topics(num_topics=config.num_topics, reduced=True)
                for topic_words, word_scores, topic_id in zip(topic_ids, topic_wordss, word_scoress):
                    print(f'{topic_id} {topic_words} {word_scores}')


if __name__ == '__main__':
    config = Configs()
    main()
