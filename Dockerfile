FROM python:3.11-slim

LABEL maintainer="your-name"
LABEL description="Model Regression Detection System — LLM eval pipeline"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY prompts/ prompts/
COPY golden_dataset/ golden_dataset/
COPY run_eval.py .

# Create directories for outputs
RUN mkdir -p reports runs

# Default environment (override at runtime)
ENV SLACK_WEBHOOK_URL=""
ENV OPENAI_API_KEY=""
ENV EVAL_WARN_THRESHOLD="0.03"
ENV EVAL_CRITICAL_THRESHOLD="0.08"

ENTRYPOINT ["python", "run_eval.py"]
CMD ["--prompt", "v1", "--ci"]
