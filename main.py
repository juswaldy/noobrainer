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
from utils import ner, clustr, tomo
from models.schemas import PredictionRequest, Document, DocumentSearch, NumTopics, TopicSizes, Topic, TopicResult, KeywordSearch, KeywordSearchDocument, KeywordSearchTopic, KeywordSearchWord, WordResult

################################################################################
# Prepare configs/settings and load models.

class Settings(BaseSettings):
    api_version: str = '3.0'
    api_title: str = 'noobrainer API'
    api_description: str = 'API for GLG Capstone by {bryantaekim, dslee47, juswaldy} @ gmail.com'
    
    # NER.
    ner_model_path: str = 'models/ner-person-org-location.pkl'

    # Clustering.
    clustr_model_path: str = 'models/clustr-health_tech-2020-01.pkl'
    
    # Topic Modeling.
    tomo_model_path: str = 'models/tomo-titles-single-17.pkl'
    num_topics: int = 10
    topics_reduced: bool = False
    top2vec: object = None

settings = Settings()

################################################################################
# Load models and get them ready.

settings.top2vec = tomo.load(settings.tomo_model_path)

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
# Generic endpoints.

@app.get("/health")
def health():
    return "Service is online."

################################################################################
# Named Entity Recognition endpoints.

@app.get("/ner",
    description="Named Entity Recognition rootpath",
    tags=["Named Entity Recognition"])
def _ner():
    return "Almost there!"

################################################################################
# Hierarchical Clustering endpoints.

@app.get("/clustr",
    description="Hierarchical Clustering rootpath",
    tags=["Hierarchical Clustering"])
def _clustr():
    return "Wait for it!"

################################################################################
# Topic Modeling endpoints.

@app.get("/tomo",
    description="Topic Modeling rootpath",
    tags=["Topic Modeling"])
def _tomo():
    return "Friendly tomo!"

@app.get("/tomo/topics/number",
    response_model=NumTopics,
    description="Returns number of topics in the model.",
    tags=["Topic Modeling"])
async def get_number_of_topics():
    return NumTopics(num_topics=settings.top2vec.get_num_topics())

@app.get("/tomo/topics/sizes",
    response_model=TopicSizes,
    description="Returns the number of documents in each topic.",
    tags=["Topic Modeling"])
async def get_topic_sizes():
    topic_sizes, topic_nums = settings.top2vec.get_topic_sizes()
    return TopicSizes(topic_nums=list(topic_nums), topic_sizes=list(topic_sizes))

@app.get("/tomo/topics/get-topics",
    response_model=List[Topic],
    description="Get number of topics.",
    tags=["Topic Modeling"])
async def get_topics(num_topics: int):
    topic_words, word_scores, topic_nums = settings.top2vec.get_topics(num_topics)
    topics = []
    for words, scores, num in zip(topic_words, word_scores, topic_nums):
        topics.append(Topic(topic_num=num, topic_words=list(words), word_scores=list(scores)))
    return topics

@app.post("/tomo/topics/query",
    response_model=List[TopicResult],
    description="Query the model for topics by freeform text.",
    tags=["Topic Modeling"])
async def query_topics(request: PredictionRequest):
    topics_words, words_scores, topic_scores, topic_nums = settings.top2vec.query_topics(
        query=request.query_string,
        num_topics=request.num_topics,
        reduced=request.topics_reduced)
    return tomo.get_topic_results(topics_words, words_scores, topic_scores, topic_nums)

@app.post("/tomo/topics/search",
    response_model=List[TopicResult],
    description="Semantic search of topics using keywords.",
    tags=["Topic Modeling"])
async def search_topics_by_keywords(keyword_search: KeywordSearchTopic):
    topics_words, words_scores, topic_scores, topic_nums = settings.top2vec.search_topics(
        keyword_search.keywords,
        keyword_search.num_topics,
        keyword_search.keywords_neg)
    return tomo.get_topic_results(topics_words, words_scores, topic_scores, topic_nums)

@app.get("/tomo/documents/search-by-topic",
    response_model=List[Document],
    description="Semantic search of documents using topic number.",
    tags=["Topic Modeling"])
async def search_documents_by_topic(topic_num: int, num_docs: int):
    documents = []
    docs, doc_scores, doc_ids = settings.top2vec.search_documents_by_topic(topic_num, num_docs)
    for doc, score, num in zip(docs, doc_scores, doc_ids):
        documents.append(Document(content=doc, score=score, doc_id=num))
    return documents

@app.post("/tomo/documents/search-by-keywords", response_model=List[Document], description="Search documents by keywords.",
          tags=["Topic Modeling"])
async def search_documents_by_keywords(keyword_search: KeywordSearchDocument):
    documents = []
    docs, doc_scores, doc_ids = settings.top2vec.search_documents_by_keywords(
        keyword_search.keywords,
        keyword_search.num_docs,
        keyword_search.keywords_neg)
    for doc, score, num in zip(docs, doc_scores, doc_ids):
        documents.append(Document(content=doc, score=score, doc_id=num))
    return documents

@app.post("/tomo/documents/search-by-documents", response_model=List[Document], description="Find similar documents.",
          tags=["Topic Modeling"])
async def search_documents_by_documents(document_search: DocumentSearch):
    documents = []
    docs, doc_scores, doc_ids = settings.top2vec.search_documents_by_documents(
        document_search.doc_ids,
        document_search.num_docs,
        document_search.doc_ids_neg)
    for doc, score, num in zip(docs, doc_scores, doc_ids):
        documents.append(Document(content=doc, score=score, doc_id=num))
    return documents

@app.post("/tomo/words/find-similar", response_model=List[WordResult], description="Search documents by keywords.",
          tags=["Topic Modeling"])
async def find_similar_words(keyword_search: KeywordSearchWord):
    words, word_scores = settings.top2vec.similar_words(
        keyword_search.keywords,
        keyword_search.num_words,
        keyword_search.keywords_neg)
    word_results = []
    for word, score in zip(words, word_scores):
        word_results.append(WordResult(word=word, score=score))
    return word_results
