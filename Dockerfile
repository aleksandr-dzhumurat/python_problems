# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim-buster

ARG PROJECT_NAME=python_problems

ARG REMOTE_USER=vscode
ARG GROUP_ID=5000
ARG USER_ID=5000

ENV VIRTUAL_ENV=/srv/${PROJECT_NAME}/.venv \
    PATH="$VIRTUAL_ENV/bin:$PATH" \
    \
    # эта переменная среды обеспечивает правильность работы импортов
    PYTHONPATH=/srv/${PROJECT_NAME} \
    # Keeps Python from generating .pyc files in the container
    PYTHONDONTWRITEBYTECODE=1 \
    # Turns off buffering for easier container logging
    PYTHONUNBUFFERED=1

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN groupadd --gid ${GROUP_ID} ${REMOTE_USER} && \
    useradd --home-dir /home/${REMOTE_USER} --create-home --uid ${USER_ID} \
        --gid ${GROUP_ID} --shell /bin/sh --skel /dev/null ${REMOTE_USER} && \
    chmod +x /usr/local/bin/docker-entrypoint.sh && \
    mkdir /srv/${PROJECT_NAME}

WORKDIR /srv/${PROJECT_NAME}

# Install pip requirements
COPY requirements.txt /srv/${PROJECT_NAME}

RUN \
    apt-get update && apt install -y git && \
    python3 -m venv --system-site-packages $VIRTUAL_ENV && \
    python3 -m pip install --no-cache -r requirements.txt && \
    chown -R ${REMOTE_USER}:${REMOTE_USER} /srv/${PROJECT_NAME}

COPY ${PROJECT_NAME}/ /srv/${PROJECT_NAME}/${PROJECT_NAME}/

USER ${REMOTE_USER}

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
ENTRYPOINT ["docker-entrypoint.sh"]
