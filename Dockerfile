FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=7860

# Create non-root user with UID 1000 (Hugging Face Spaces standard)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy remaining project files
COPY --chown=user:user . .

# Hugging Face Spaces expects traffic on port 7860
EXPOSE 7860

CMD ["python", "run_dashboard.py"]
