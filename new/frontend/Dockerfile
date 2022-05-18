FROM python:3.7
EXPOSE 8501
WORKDIR .
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "streamlit", "run", "app.py" ]