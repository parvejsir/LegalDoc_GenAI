# Stage 1: The Build Environment
# Use a full Python image to ensure all build dependencies are available
FROM python:3.12.7 as builder

# Set the working directory
WORKDIR /app

# Install system dependencies needed for libraries like PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libharfbuzz-dev \
    libjpeg-dev \
    libtiff-dev \
    libopenjp2-7 \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libzstd-dev \
    libopenexr-dev \
    libgomp1 \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --default-timeout=3600 -r requirements.txt

# Copy all application code
COPY . .

# Stage 2: The Final, Leaner Runtime Environment
# Use a minimal Python image with glibc for better compatibility
FROM python:3.12.7-slim

# Set the working directory
WORKDIR /app

# Copy only the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
# Copy the application source code from the builder stage
COPY --from=builder /app /app

# Expose the port
EXPOSE 8000

# Set environment variables from the .env.example
ENV GOOGLE_API_KEY="your-google-api-key"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
ENV LLM_MODEL="gemini-2.5-flash"
ENV EMBEDDING_DIM=384
ENV PINECONE_API_KEY="your-pinecone-api-key"
ENV PINECONE_INDEX_NAME="legaldocstore"
ENV PINECONE_METRIC="cosine"

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]