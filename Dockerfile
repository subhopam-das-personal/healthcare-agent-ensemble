FROM python:3.12-slim

WORKDIR /app

# Install deps in a separate layer — only reruns when requirements.txt changes
COPY requirements.txt .
# Install all deps except zai-sdk first so mcp pulls in pyjwt>=2.10.1
RUN grep -v '^zai-sdk' requirements.txt > /tmp/reqs-no-zai.txt && \
    pip install --no-cache-dir -r /tmp/reqs-no-zai.txt
# Install zai-sdk without its declared deps to avoid the pyjwt<2.9.0 constraint.
# pyjwt>=2.10.1 is already installed above; zai-sdk's JWT code is compatible.
RUN pip install --no-cache-dir --no-deps 'zai-sdk>=0.2.2'

# Copy source after deps so code changes don't bust the pip cache layer
COPY . .

CMD ["bash", "start.sh"]
