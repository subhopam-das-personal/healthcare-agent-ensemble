FROM python:3.12-slim

WORKDIR /app

# Install deps in a separate layer — only reruns when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source after deps so code changes don't bust the pip cache layer
COPY . .

CMD ["bash", "start.sh"]
