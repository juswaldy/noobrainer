FROM python:3.8
EXPOSE 8501
WORKDIR .
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "streamlit", "run", "app.py" ]
