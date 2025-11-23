FROM python:3.10-slim

WORKDIR /app

# Install Java (required for PySpark)
RUN apt-get update && \
    apt-get install -y openjdk-21-jre-headless procps && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    pyspark==3.5.0 \
    streamlit==1.28.0 \
    pandas==2.1.0 \
    numpy==1.24.3 \
    plotly==5.17.0 \
    pymongo \
    && rm -rf /root/.cache/pip

# Copy application files
COPY app.py .
COPY test_model.py .

# Create models directory
RUN mkdir -p /app/models

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD pgrep -f streamlit || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
