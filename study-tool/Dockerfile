FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /data

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ressources
RUN mkdir -p images
COPY --chown=root:root images images

RUN mkdir -p thumbnails
COPY --chown=root:root thumbnails thumbnails

RUN mkdir -p video_candidates
COPY --chown=root:root video_candidates video_candidates

# Script
COPY main.py .


EXPOSE 8001
CMD ["python","-m","uvicorn","main:app","--host","0.0.0.0","--port","8001"]
