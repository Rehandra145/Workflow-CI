# ============================================
# OPTIMIZED DOCKERFILE - MLflow Serve
# ============================================

FROM python:3.10-slim

WORKDIR /app

# Install dependencies dengan optimasi
COPY requirements.txt .
RUN pip install --no-cache-dir --no-compile -r requirements.txt \
  && find /usr/local -name "*.pyc" -delete \
  && find /usr/local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true \
  && find /usr/local -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true \
  && rm -rf /root/.cache/pip

# Copy model
COPY model/ ./model/

# Expose port
EXPOSE 5000

# Jalankan MLflow serve
CMD ["mlflow", "models", "serve", "-m", "model", "-h", "0.0.0.0", "-p", "5000", "--no-conda"]
