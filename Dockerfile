# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY src /app/src
COPY vocab /app/vocab
COPY checkpoints /app/checkpoints
COPY inference.py /app
COPY config.yaml /app
COPY requirements.txt /app
COPY translation_api.py /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["uvicorn", "translation_api:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t translation_api .
# docker run -p 8000:8000 translation_api