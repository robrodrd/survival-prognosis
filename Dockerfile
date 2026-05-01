# --- 1. Use a lightweight Linux OS (Python 3.12 on Linux)
FROM python:3.11-slim

# Install system dependencies for Reflex (Node.js and unzip)
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# --- 2. CREATE WORKING DIRECTORY inside the CONTAINER ---
WORKDIR /app

# --- 3. Copy requirements (dependencies) and install them inside the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 4. COPY APPLICATION CODE ---
# Copy everything from your local folder into the /app folder
COPY . .

# Initialize Reflex (compiles the frontend)
RUN reflex init

# Expose ports for the Frontend (3000) and Backend (8000)
EXPOSE 3000 8000

# --- 6. ENVIRONMENT VARIABLES ---
# Ensure Python output is sent straight to terminal (good for debugging)
ENV PYTHONUNBUFFERED=1

# --- 7. START COMMAND ---
# This is the "Entry Point" that runs when you start the container
# Run the app in production mode
CMD ["reflex", "run", "--env", "prod"]