from pydantic import BaseModel
from typing import List

################################################################################
# Common schemas.

class PredictionRequest(BaseModel):
    query_string: str
    num_topics: int
    topics_reduced: bool

class ModelRefresh(BaseModel):
    client_id: str
    model_path: str

################################################################################
# NER schemas.

class Classification(BaseModel):
    query_string: str
    class_num: int
    class_str: str

################################################################################
# Clustering schemas.

################################################################################
# Topic Modeling schemas.

class Document(BaseModel):
    score: float
    doc_id: int

class DocumentSearch(BaseModel):
    doc_ids: List[int]
    doc_ids_neg: List[int]
    num_docs: int

class NumTopics(BaseModel):
    num_topics: int

class TopicSizes(BaseModel):
    topic_nums: List[int]
    topic_sizes: List[int]

class Topic(BaseModel):
    topic_num: int
    topic_words: List[str]
    word_scores: List[float]

class TopicResult(Topic):
    topic_num: int
    topic_score: float
    topic_words: List[str]
    word_scores: List[float]

class KeywordSearch(BaseModel):
    keywords: List[str]
    keywords_neg: List[str]

class KeywordSearchDocument(KeywordSearch):
    num_docs: int

class KeywordSearchTopic(KeywordSearch):
    num_topics: int

class KeywordSearchWord(KeywordSearch):
    num_words: int

class WordResult(BaseModel):
    word: str
    score: float
