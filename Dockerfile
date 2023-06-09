FROM python:3.9.16-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk;  nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('wordnet')"

COPY . .

EXPOSE 80

ENTRYPOINT [ "python" ]
CMD ["server.py"]