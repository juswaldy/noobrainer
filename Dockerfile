FROM python:3.7
EXPOSE 8000
WORKDIR .
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "uvicorn", "--host", "0.0.0.0", "--port", "8000", "--reload", "main:app" ]