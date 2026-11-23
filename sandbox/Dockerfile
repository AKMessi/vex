FROM node:24-bookworm-slim AS node
FROM python:3.12-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && apt-get update \
    && apt-get install -y --no-install-recommends chromium chromium-headless-shell ffmpeg libcairo2-dev libpango1.0-dev pkg-config build-essential fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir manim==0.20.1
WORKDIR /opt/vex
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force
COPY sandbox/render.py sandbox/remotion.mjs ./sandbox/
ENV HOME=/tmp HYPERFRAMES_BROWSER_PATH=/usr/bin/chromium-headless-shell HYPERFRAMES_NO_TELEMETRY=1
USER 65534:65534
