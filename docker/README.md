# AWS Learning Environment (JupyterLab + AWS CLI)

This repository contains a containerized development environment designed for hands-on cloud engineering, machine learning, and Generative AI practice. It provisions an isolated Ubuntu-based environment pre-configured with a virtual environment, core ML frameworks, and the AWS CLI v2.

---

## Features

* **Multi-Architecture Support:** Dynamically detects and installs the correct AWS CLI v2 binary depending on whether you are running on Apple Silicon (`arm64`) or Intel/AMD (`x86_64`).
* **Rootless Security:** The container runs under a dedicated non-root user (`docker-usr`) for runtime security compliance.
* **Pre-installed Stack:**
  * **Languages & Package Managers:** Python 3, Built-in `venv`, Pip (upgraded).
  * **Core Tools:** AWS CLI v2, curl, unzip.
  * **AI/ML & Gen-AI:** `jupyterlab`, `streamlit`, `langchain-chroma`.

---

## Prerequisites

1. **Docker Desktop** installed and running on your system.
2. A file named `packages` in the root of your repository containing any extra `apt` packages you wish to install (one package per line). If you don't have any, just create an empty file:
   ```bash
   touch packages

## Build the image

```bash
docker build -t <image-name>:latest .
```

## Run the container

```bash
docker run -d \
  -p 8888:8888 \
  -v <path/to/local/filesystem>:/mnt \
  -e JUPYTER_TOKEN="token" \
  --name <container-name> \
  <image-name>:latest
```