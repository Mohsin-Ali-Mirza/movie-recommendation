# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all necessary files into the container
COPY Backend/content_bases_model.py .
COPY Backend/fast_api.py .
COPY Backend/svd_bases_recomendor.py .
COPY Backend/movies.py .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]