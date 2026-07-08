# Use official Python image as base
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy all application files
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Set environment variables to avoid Streamlit telemetry
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]
