FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl unzip && \
    curl -fsSL https://deb.nodejsource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Explicitly install the production servers to avoid "binary not found"
# RUN pip install --no-cache-dir -r requirements.txt uvicorn gunicorn (uvicorn & gunicorn are in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Initialize and Export
RUN reflex init

# -- This url is created in the GCR Aritfact Registry: its: Sevice Name.Project Number.Region
# ENV API_URL=https://survival-prognosis-service-100651041936.us-central1.run.app   

# -- This is the Unique Hash URL. It is assigned to the Service when it was deployed for the first time. This is the official one and should be used
# ENV API_URL=https://survival-prognosis-service-43ggbiilka-uc.a.run.app

# Needs to be changed to the Firebase Assigned User friendl URL:
ENV API_URL=https://healthcare-analytics-v1.web.app

RUN reflex export --frontend-only --no-zip

# START COMMAND
EXPOSE 8080

# We use 'reflex run' but point it to the internal module.
# This ensures Reflex sets up the static file mounting for the frontend.
CMD ["reflex", "run", "--env", "prod", "--backend-host", "0.0.0.0", "--backend-port", "8080", "--loglevel", "debug"]