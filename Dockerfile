FROM python:3.10-slim-buster

# Install libraries
RUN apt-get -qq update && apt-get install -y --no-install-recommends python3-opencv
COPY ./requirements.txt ./
RUN pip install -r requirements.txt

# Setup container directories
RUN mkdir /app

# Copy local code to the container
COPY ./app /app

# launch server with gunicorn
WORKDIR /app
EXPOSE 8080
CMD ["gunicorn", "main:app", "--timeout=0", "--reload", "--preload", "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]
