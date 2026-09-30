# Use a minimal base image with Python installed
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container to /app
WORKDIR /app

# Install Python dependencies listed in requirements.txt (copied first for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app
COPY . .

EXPOSE 8501

# Run the Streamlit app on port 8501 and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--browser.gatherUsageStats=false"]

# NOTE: XSRF protection and CORS checks are disabled so the app works through the Codespaces forwarded-port proxy
