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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, BaseSettings
from typing import List
#from top2vec import Top2Vec
import numpy as np
from utils import tomo

################################################################################
# Prepare configs/settings and load models.

class Settings(BaseSettings):
    api_version: str = '2.0'
    api_title: str = 'noobrainer API'
    api_description: str = 'API for GLG Capstone by {bryantaekim, dslee47, juswaldy} @ gmail.com'
    ner_model_path: str = 'models/ner_model.pkl'
    clustr_model_path: str = 'models/clustr_model.pkl'
    tomo_model_path: str = "models/small-articles-single.t2v"

settings = Settings()

################################################################################
# Load models and get them ready.

top2vec = tomo.load(settings.tomo_model_path)


# Determine top2vec index type and if it has documents.
doc_id_type = str if top2vec.doc_id_type is np.str_ else int
has_documents = False if top2vec.documents is None else True


# Prepare schemas.
class Document(BaseModel):
    if has_documents:
        content: str
    score: float
    doc_id: doc_id_type


class DocumentSearch(BaseModel):
    doc_ids: List[doc_id_type]
    doc_ids_neg: List[doc_id_type]
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
    topic_score: float


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


################################################################################


app = FastAPI(title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
)


################################################################################
# Handle errors.

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=404,
        content={"message": str(exc)},
    )


################################################################################
# Prepare schemas.

class PredictionRequest(BaseModel):
    query_string: str


################################################################################
# Generic endpoints.

@app.get("/health")
def health():
    return "Service is online."


################################################################################
# Named Entity Recognition endpoints.

@app.get("/ner",
    description="Named Entity Recognition rootpath",
    tags=["Named Entity Recognition"])
def ner():
    return "Almost there!"

################################################################################
# Hierarchical Clustering endpoints.

@app.get("/clustr",
    description="Hierarchical Clustering rootpath",
    tags=["Hierarchical Clustering"])
def clustr():
    return "Wait for it!"

################################################################################
# Topic Modeling endpoints.

@app.get("/tomo",
    description="Topic Modeling rootpath",
    tags=["Topic Modeling"])
def tomo():
    return "Friendly tomo!"


@app.get("/tomo/topics/number",
    response_model=NumTopics,
    description="Returns number of topics in the model.",
    tags=["Topic Modeling"])
async def get_number_of_topics():
    return NumTopics(num_topics=top2vec.get_num_topics())


@app.get("/tomo/topics/sizes",
    response_model=TopicSizes,
    description="Returns the number of documents in each topic.",
    tags=["Topic Modeling"])
async def get_topic_sizes():
    topic_sizes, topic_nums = top2vec.get_topic_sizes()
    return TopicSizes(topic_nums=list(topic_nums), topic_sizes=list(topic_sizes))


@app.get("/tomo/topics/get-topics",
    response_model=List[Topic],
    description="Get number of topics.",
    tags=["Topic Modeling"])
async def get_topics(num_topics: int):
    topic_words, word_scores, topic_nums = top2vec.get_topics(num_topics)

    topics = []
    for words, scores, num in zip(topic_words, word_scores, topic_nums):
        topics.append(Topic(topic_num=num, topic_words=list(words), word_scores=list(scores)))

    return topics


@app.post("/tomo/topics/search",
    response_model=List[TopicResult],
    description="Semantic search of topics using keywords.",
    tags=["Topic Modeling"])
async def search_topics_by_keywords(keyword_search: KeywordSearchTopic):
    topic_words, word_scores, topic_scores, topic_nums = top2vec.search_topics(keyword_search.keywords,
                                                                               keyword_search.num_topics,
                                                                               keyword_search.keywords_neg)

    topic_results = []
    for words, word_scores, topic_score, topic_num in zip(topic_words, word_scores, topic_scores, topic_nums):
        topic_results.append(TopicResult(topic_num=topic_num, topic_words=list(words),
                                         word_scores=list(word_scores), topic_score=topic_score))

    return topic_results


@app.get("/tomo/documents/search-by-topic",
    response_model=List[Document],
    description="Semantic search of documents using topic number.",
    tags=["Topic Modeling"])
async def search_documents_by_topic(topic_num: int, num_docs: int):
    documents = []

    if has_documents:
        docs, doc_scores, doc_ids = top2vec.search_documents_by_topic(topic_num, num_docs)
        for doc, score, num in zip(docs, doc_scores, doc_ids):
            documents.append(Document(content=doc, score=score, doc_id=num))

    else:
        doc_scores, doc_ids = top2vec.search_documents_by_topic(topic_num, num_docs)
        for score, num in zip(doc_scores, doc_ids):
            documents.append(Document(score=score, doc_id=num))

    return documents


@app.post("/tomo/documents/search-by-keywords", response_model=List[Document], description="Search documents by keywords.",
          tags=["Topic Modeling"])
async def search_documents_by_keywords(keyword_search: KeywordSearchDocument):
    documents = []

    if has_documents:
        docs, doc_scores, doc_ids = top2vec.search_documents_by_keywords(keyword_search.keywords,
                                                                         keyword_search.num_docs,
                                                                         keyword_search.keywords_neg)
        for doc, score, num in zip(docs, doc_scores, doc_ids):
            documents.append(Document(content=doc, score=score, doc_id=num))
    else:
        doc_scores, doc_ids = top2vec.search_documents_by_keywords(keyword_search.keywords,
                                                                   keyword_search.num_docs,
                                                                   keyword_search.keywords_neg)
        for score, num in zip(doc_scores, doc_ids):
            documents.append(Document(score=score, doc_id=num))

    return documents


@app.post("/tomo/documents/search-by-documents", response_model=List[Document], description="Find similar documents.",
          tags=["Topic Modeling"])
async def search_documents_by_documents(document_search: DocumentSearch):
    documents = []

    if has_documents:
        docs, doc_scores, doc_ids = top2vec.search_documents_by_documents(document_search.doc_ids,
                                                                          document_search.num_docs,
                                                                          document_search.doc_ids_neg)
        for doc, score, num in zip(docs, doc_scores, doc_ids):
            documents.append(Document(content=doc, score=score, doc_id=num))
    else:
        doc_scores, doc_ids = top2vec.search_documents_by_documents(document_search.doc_ids,
                                                                    document_search.num_docs,
                                                                    document_search.doc_ids_neg)
        for score, num in zip(doc_scores, doc_ids):
            documents.append(Document(score=score, doc_id=num))

    return documents


@app.post("/tomo/words/find-similar", response_model=List[WordResult], description="Search documents by keywords.",
          tags=["Topic Modeling"])
async def find_similar_words(keyword_search: KeywordSearchWord):
    words, word_scores = top2vec.similar_words(keyword_search.keywords, keyword_search.num_words,
                                               keyword_search.keywords_neg)

    word_results = []
    for word, score in zip(words, word_scores):
        word_results.append(WordResult(word=word, score=score))

    return word_results
