FROM ubuntu:latest

ENV DOCKER_USR=docker-usr

RUN useradd -m -s /bin/bash ${DOCKER_USR}

COPY packages /tmp/apt-packages.txt

RUN apt-get update && \
    apt-get install -y $(cat /tmp/apt-packages.txt) python3 python3-venv curl unzip && \
    rm -rf /tmp/apt-packages.txt /var/lib/apt/lists/* && \
    apt-get clean

# Install AWS CLI v2 (auto-detects ARM64 vs x86_64)
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then \
        CLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip"; \
    else \
        CLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"; \
    fi && \
    curl "$CLI_URL" -o "/tmp/awscliv2.zip" && \
    cd /tmp && unzip awscliv2.zip && ./aws/install

RUN python3 -m venv /home/${DOCKER_USR}/.venv && \
    chown -R ${DOCKER_USR}:${DOCKER_USR} /home/${DOCKER_USR}/.venv

ENV PATH="/home/${DOCKER_USR}/.venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install jupyterlab streamlit langchain-chroma

# Switch to non-root user
USER ${DOCKER_USR}
WORKDIR /home/${DOCKER_USR}

EXPOSE 8888

CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/home/${DOCKER_USR}