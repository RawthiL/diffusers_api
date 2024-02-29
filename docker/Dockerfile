FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /code

# Setup server requriements
COPY docker/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy over app
WORKDIR /code/app
COPY api /code/app

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
