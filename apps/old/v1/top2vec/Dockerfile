FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
EXPOSE 8501
COPY ./app /app
ENV model_name="Top2Vec API"
ENV model_path=/app/top2vec_model
COPY requirements.txt /app
RUN pip install -r requirements.txt
