FROM python:3.11-slim

# Install system dependencies including Node.js (for Reflex frontend) 
# and WeasyPrint native libraries (Pango, Cairo, GObject)
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz-subset0 \
    libgobject-2.0-0 \
    libcairo2 \
    libffi-dev \
    shared-mime-info && \
    curl -fsSL https://deb.nodejsource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*    

WORKDIR /app
COPY . .

# Explicitly install the production servers to avoid "binary not found"
RUN pip install --no-cache-dir -r requirements.txt

# Initialize and Export
RUN reflex init

# 1. Point to your professional domain
ENV API_URL=https://clinicalprognosis.app

# 2. Tell the backend to TRUST the new domain (Crucial for WebSockets)
ENV CORS_ALLOWED_ORIGINS='["https://clinicalprognosis.app"]'

ENV REFLEX_PROTOCOL_HTTP=true

RUN reflex export --frontend-only --no-zip

# START COMMAND
EXPOSE 8080

# We use 'reflex run' but point it to the internal module.
# This ensures Reflex sets up the static file mounting for the frontend.
CMD ["reflex", "run", "--env", "prod", "--backend-host", "0.0.0.0", "--backend-port", "8080", "--loglevel", "debug"]