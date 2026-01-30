# Use official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports
EXPOSE 8000 8501

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$SERVICE" = "api" ]; then\n\
    uvicorn api.main:app --host 0.0.0.0 --port 8000\n\
elif [ "$SERVICE" = "streamlit" ]; then\n\
    streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0\n\
else\n\
    echo "Please set SERVICE environment variable to api or streamlit"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
