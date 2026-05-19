# 🚀 How to Deploy — DeepSearch Engine

This guide covers every deployment method for the DeepSearch Graph-Based Research Engine, from local development to cloud production.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Deploy on Render (Recommended)](#deploy-on-render)
4. [Deploy on Railway](#deploy-on-railway)
5. [Deploy with Docker](#deploy-with-docker)
6. [Deploy on AWS EC2](#deploy-on-aws-ec2)
7. [Deploy on Google Cloud Run](#deploy-on-google-cloud-run)
8. [Environment Variables Reference](#environment-variables-reference)
9. [Post-Deployment Verification](#post-deployment-verification)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before deploying, ensure you have:

- **Python 3.10+** installed
- **Git** installed and configured
- API keys (see [Environment Variables](#environment-variables-reference))
- A GitHub account with this repository pushed

### Required API Keys

| Key | Required | How to Get |
|---|---|---|
| `GROQ_API_KEY` | Recommended | Sign up at [console.groq.com](https://console.groq.com) → API Keys |
| `BRAVE_API_KEY` | Optional | Sign up at [brave.com/search/api](https://brave.com/search/api/) |

> **Note:** The engine works without any API keys — it falls back to DuckDuckGo for search and TextRank for answer generation. However, Groq API provides significantly better answer quality.

---

## Local Development

### Step 1: Clone & Setup

```bash
# Clone the repository
git clone https://github.com/RKG765/deep-based-search-engine.git
cd deep-based-search-engine/deep_search_engine   # All code lives here

# Create a virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate          # Windows (PowerShell)
venv\Scripts\activate.bat      # Windows (CMD)
source venv/bin/activate       # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 2: Configure Environment

Create a `.env` file in the `deep_search_engine/` directory:

```env
# Required for best answer quality
GROQ_API_KEY=gsk_your_groq_api_key_here

# Optional — enables Brave Search (DuckDuckGo is the free fallback)
BRAVE_API_KEY=your_brave_api_key_here

# Optional — if running Ollama locally as LLM fallback
OLLAMA_BASE_URL=http://localhost:11434
```

### Step 3: Run

**API Server:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Streamlit Demo (separate terminal):**
```bash
streamlit run live_demo.py --server.port 8501
```

**Verify:**
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","app_name":"Deep Recursive Research Search Engine"}
```

---

## Deploy on Render

> **Recommended for quick, free deployment.** Render auto-deploys from GitHub.

### Step 1: Create `render.yaml` (Blueprint)

Create this file in the **repository root** (one level above this folder):

```yaml
# render.yaml
services:
  - type: web
    name: deepsearch-api
    runtime: python
    region: oregon
    plan: free
    buildCommand: |
      cd deep_search_engine
      pip install -r requirements.txt
      python -m spacy download en_core_web_sm
    startCommand: |
      cd deep_search_engine
      uvicorn app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: BRAVE_API_KEY
        sync: false
      - key: PYTHON_VERSION
        value: "3.11"
```

### Step 2: Deploy

1. Push `render.yaml` to GitHub
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**
3. Connect your GitHub repo `RKG765/deep-based-search-engine`
4. Render will detect `render.yaml` and configure the service
5. Add your environment variables in the Render dashboard
6. Click **Deploy**

### Step 3: Verify

```bash
curl https://deepsearch-api.onrender.com/health
```

> ⚠️ **Free tier note:** Render free services spin down after 15 minutes of inactivity. The first request after sleep takes ~30 seconds.

---

## Deploy on Railway

### Step 1: Create `Procfile`

Create in the **repository root** (one level above this folder):

```
web: cd deep_search_engine && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Step 2: Create `railway.json`

```json
{
  "$schema": "https://railway.com/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "cd deep_search_engine && uvicorn app.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

### Step 3: Create `nixpacks.toml`

```toml
[phases.setup]
nixPkgs = ["python311"]

[phases.install]
cmds = [
    "cd deep_search_engine && pip install -r requirements.txt",
    "python -m spacy download en_core_web_sm"
]
```

### Step 4: Deploy

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
2. Select `RKG765/deep-based-search-engine`
3. Add environment variables: `GROQ_API_KEY`, `BRAVE_API_KEY`
4. Railway auto-deploys on push

---

## Deploy with Docker

### Step 1: Create `Dockerfile`

Create in the **repository root** (one level above this folder):

```dockerfile
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY deep_search_engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy application code
COPY deep_search_engine/ .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2: Create `.dockerignore`

```
.git
.gitignore
__pycache__
*.pyc
.env
venv/
.venv/
*.egg-info
.pytest_cache
*.pptx
*.docx
diagrams/
images/
```

### Step 3: Build & Run

```bash
# Build the image
docker build -t deepsearch-engine .

# Run with environment variables
docker run -d \
    --name deepsearch \
    -p 8000:8000 \
    -e GROQ_API_KEY=your_key_here \
    -e BRAVE_API_KEY=your_key_here \
    deepsearch-engine

# Check logs
docker logs -f deepsearch
```

### Step 4: Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - BRAVE_API_KEY=${BRAVE_API_KEY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# Run with compose
docker compose up -d
```

---

## Deploy on AWS EC2

### Step 1: Launch an EC2 Instance

- **AMI:** Ubuntu 22.04 LTS
- **Instance type:** `t3.medium` (2 vCPU, 4 GB RAM — minimum recommended)
- **Storage:** 20 GB gp3
- **Security Group:** Allow inbound on port `8000` (or `80`/`443` behind Nginx)

### Step 2: SSH & Setup

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Clone repository
git clone https://github.com/RKG765/deep-based-search-engine.git
cd deep-based-search-engine/deep_search_engine

# Create venv
python3.11 -m venv venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set environment variables
export GROQ_API_KEY="your_key_here"
export BRAVE_API_KEY="your_key_here"
```

### Step 3: Run with systemd (Production)

Create `/etc/systemd/system/deepsearch.service`:

```ini
[Unit]
Description=DeepSearch API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/deep-based-search-engine/deep_search_engine
ExecStart=/home/ubuntu/deep-based-search-engine/deep_search_engine/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
Environment="GROQ_API_KEY=your_key_here"
Environment="BRAVE_API_KEY=your_key_here"

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable deepsearch
sudo systemctl start deepsearch
sudo systemctl status deepsearch
```

### Step 4: Nginx Reverse Proxy (Optional)

```bash
sudo apt install -y nginx

sudo tee /etc/nginx/sites-available/deepsearch << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/deepsearch /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
```

---

## Deploy on Google Cloud Run

### Step 1: Build & Push to Container Registry

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build with Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/deepsearch-engine .
```

### Step 2: Deploy

```bash
gcloud run deploy deepsearch-api \
    --image gcr.io/YOUR_PROJECT_ID/deepsearch-engine \
    --platform managed \
    --region us-central1 \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 120 \
    --set-env-vars "GROQ_API_KEY=your_key,BRAVE_API_KEY=your_key" \
    --allow-unauthenticated
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Recommended | — | Groq cloud LLM API key |
| `BRAVE_API_KEY` | Optional | — | Brave Search API key |
| `OLLAMA_BASE_URL` | Optional | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | Optional | `llama3` | Ollama model name |
| `DEBUG` | Optional | `true` | Enable debug logging |
| `PORT` | Auto-set | `8000` | Port (set by cloud platforms) |

---

## Post-Deployment Verification

After deploying, run these checks:

### 1. Health Check

```bash
curl https://YOUR_DEPLOYED_URL/health
# Expected: {"status":"ok","app_name":"Deep Recursive Research Search Engine"}
```

### 2. API Test

```bash
curl -X POST https://YOUR_DEPLOYED_URL/api/v1/deep-search \
    -H "Content-Type: application/json" \
    -d '{"query": "what is pagerank algorithm", "depth": 1, "pruning": true}'
```

### 3. OpenAPI Docs

Visit `https://YOUR_DEPLOYED_URL/docs` for the interactive Swagger UI.

---

## Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: spacy` | spaCy model not downloaded | Run `python -m spacy download en_core_web_sm` |
| `Connection refused on port 8000` | App not binding to `0.0.0.0` | Use `--host 0.0.0.0` in uvicorn command |
| Timeout on first request | Embedding model loading (~20s) | Normal — first request loads `all-MiniLM-L6-v2` into memory |
| `GROQ_API_KEY` errors | Invalid or missing key | Verify key at [console.groq.com](https://console.groq.com). Engine falls back to TextRank. |
| `Out of memory` on free tier | Embedding model needs ~1.5 GB RAM | Use `t3.medium` or Railway/Render paid tier |
| No search results returned | Rate limiting from DuckDuckGo | Add `BRAVE_API_KEY` for more reliable search |

### Minimum Resource Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| **RAM** | 1.5 GB | 4 GB |
| **CPU** | 1 vCPU | 2 vCPU |
| **Disk** | 5 GB | 10 GB |
| **Python** | 3.10 | 3.11 |

---

<div align="center">

**Questions?** Open an issue on [GitHub](https://github.com/RKG765/deep-based-search-engine/issues).

</div>
