# ──────────────────────────────────────────────────────────────────────────────
# PostgreSQL 16 + Apache AGE + pgvector — single DB container
#
# Base:    apache/age:PG16  (PostgreSQL 16 with AGE pre-installed)
# Adds:    pgvector (built from source against PG16 headers)
# ──────────────────────────────────────────────────────────────────────────────
FROM apache/age:PG16_latest

ARG PGVECTOR_VERSION=0.7.4

# Build dependencies for pgvector
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        postgresql-server-dev-16 \
    && rm -rf /var/lib/apt/lists/*

# Compile and install pgvector
RUN cd /tmp \
    && git clone --depth 1 --branch "v${PGVECTOR_VERSION}" \
        https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# Remove build deps to keep the image lean
RUN apt-get purge -y --auto-remove build-essential git postgresql-server-dev-16

# Initialization scripts run once on first container start (alphabetical order)
COPY init/ /docker-entrypoint-initdb.d/
