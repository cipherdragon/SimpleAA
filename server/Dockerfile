FROM python:3.10-slim
RUN apt update \
    && apt install -y redis-server \
    && apt clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV NLTK_DATA="./nltk_data"
EXPOSE 5000
CMD [ "./start.sh" ]