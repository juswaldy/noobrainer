"""
@author: noobrainers
@contact: Bryan T. Kim bryantaekim at gmail dot com
          Daniel Lee dslee47 at gmail dot com
          Juswaldy Jusman juswaldy at gmail dot com
@description: Utility functions for Topic Modeling.
@content:
    def load()
    def augment_with_time_columns()
"""

import numpy as np
from pandas import DataFrame
from datetime import datetime
import re
import math
import sys
sys.path.append('../')
from models.schemas import *


################################################################################
# API utilities.
def get_topic_results(
    topics_words: List[List[str]],
    words_scores: List[List[float]],
    topic_scores: List[float],
    topic_nums: List[int]
) -> List[TopicResult]:
    topic_results = []
    for words, word_scores, topic_score, topic_num in zip(topics_words, words_scores, topic_scores, topic_nums):
        topic_results.append(TopicResult(
            topic_num=topic_num,
            topic_words=list(words),
            word_scores=list(word_scores),
            topic_score=topic_score))
    return topic_results


################################################################################
# Model utilities.
def load(model_path: str) -> object:
    """Load the model.

    Parameters
    ----------
    model_path : str
        Path to the model file.
    
    Returns
    -------
    object
        The loaded model.
    """
    from top2vec import Top2Vec
    model = Top2Vec.load(model_path)
    doc_id_type = str if model.doc_id_type is np.str_ else int
    has_documents = False if model.documents is None else True
    return model, doc_id_type, has_documents


################################################################################
# Miscellaneous utilities.
def augment_with_time_columns(df: DataFrame) -> DataFrame:
    """Augment the dataframe with time columns:
    year, month, day, dayofweek, isweekend, weeknum, season, yearseason, quarter, yearquarter.

    Parameters
    ----------
    df : DataFrame
        The dataframe to augment. Must have a `date` column.

    Returns
    -------
    DataFrame
        The augmented dataframe.
    """

    def date2season(seasons: dict, monthday: str) -> str:
        """Convert a month and day to a season.

        Parameters
        ----------
        seasons : dict
            A dictionary of regexes -> seasons.

        monthday : str
            The month and day to convert.

        Returns
        -------
        str
            The season.
        """
        result = None
        for regex, season in seasons.items():
            if re.search(regex, monthday):
                result = season
                break
        return result

    def make_yearseason(yearstart: dict, x: object):
        """Make a yearseason string for the given row, so that we can order it sequentially.

        Parameters
        ----------
        yearstart : dict
            A dictionary of season -> seasonindex.

        x : object
            The row to make the yearseason string for.

        Returns
        -------
        str
            The yearseason string: year-seasonindex-season.
        """
        year = int(x.year)-1 if (x.season=='winter' and x.month in ['01', '02', '03']) else x.year
        return f'{year}-{yearstart[x.season]}-{x.season}'

    # Add year, month, day.
    df[['year', 'month', 'day']] = df.date.str.split(' ', expand=True).rename(columns={0: 'date', 1: 'time'}).date.str.split('-', expand=True)

    # Add dayofweek. Starts with Sunday=0, ends with Saturday=6.
    df['dayofweek'] = df.apply(lambda x: (datetime(int(x.year), int(x.month), int(x.day)).weekday()+1)%7, axis=1)

    # Add isweekend.
    weekend = [0, 6] # Sunday and Saturday
    df['isweekend'] = df.apply(lambda x: x.dayofweek in weekend, axis=1)

    # Add weeknum.
    df['weeknum'] = df.apply(lambda x: int(datetime(int(x.year), int(x.month), int(x.day)).strftime('%U')), axis=1)

    # Make quick regexes for converting dates to seasons.
    # CAUTION: will not check for invalid dates.
    seasons_astronomical = {
        '^(03(2[0-9]|3.*)|04.*|05.*|06([01].*|20))$': 'spring',
        '^(06(2[1-9]|3.*)|07.*|08.*|09([01].*|20|21))$': 'summer',
        '^(09(2[2-9]|3.*)|10.*|11.*|12([01].*|20))$': 'fall',
        '^(12(2[1-9]|3.*)|01.*|02.*|03([01].*))$': 'winter'
    }
    seasons_meteorological = {
        '^0[3-5].*$': 'spring',
        '^0[6-8].*$': 'summer',
        '^(09|10|11).*$': 'fall',
        '^(12|01|02).*$': 'winter'
    }
    seasons = seasons_astronomical

    # Add seasons.
    df['season'] = df.apply(lambda x: date2season(seasons, f'{x.month}{x.day}'), axis=1)

    # Make a yearstart dictionary for ordering the year/seasons sequentially.
    yearstart_spring = {
        'spring': 0,
        'summer': 1,
        'fall': 2,
        'winter': 3
    }
    yearstart = yearstart_spring

    # Add yearseason.
    df['yearseason'] = df.apply(lambda x: make_yearseason(yearstart, x), axis=1)

    # Add quarter.
    df['quarter'] = df.apply(lambda x: math.ceil(int(x.month)/3), axis=1)

    # Add yearquarter.
    df['yearquarter'] = df.apply(lambda x: f'{x.year}-{x.quarter}', axis=1)

    # Add yearmonth.
    df['yearmonth'] = df.apply(lambda x: f'{x.year}-{x.month}', axis=1)

    return df
