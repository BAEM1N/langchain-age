# ──────────────────────────────────────────────────────────────────────────────
# PostgreSQL 16 + Apache AGE + pgvector — single DB container
#
# Base:    apache/age:PG16  (PostgreSQL 16 with AGE pre-installed)
# Adds:    pgvector (built from source against PG16 headers)
# ──────────────────────────────────────────────────────────────────────────────
FROM apache/age:latest

# pgvector: use HEAD (main) until a stable release supports PostgreSQL 18
# (v0.8.0 uses the old vacuum_delay_point() signature that changed in PG18)
ARG PGVECTOR_VERSION=master

# Install build deps + ca-certificates (needed for git clone over HTTPS)
RUN PG_MAJOR=$(pg_config --version | grep -oP '\d+' | head -1) \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        git \
        postgresql-server-dev-${PG_MAJOR} \
    && rm -rf /var/lib/apt/lists/*

# Compile and install pgvector from source
# OPTFLAGS="" disables SIMD/AVX extensions that cause SIGILL in emulated environments
RUN cd /tmp \
    && git clone --depth 1 --branch "${PGVECTOR_VERSION}" \
        https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make OPTFLAGS="" \
    && make install \
    && rm -rf /tmp/pgvector

# Strip build-only packages to keep the final image lean
RUN PG_MAJOR=$(pg_config --version | grep -oP '\d+' | head -1) \
    && apt-get purge -y --auto-remove \
        build-essential git postgresql-server-dev-${PG_MAJOR}

# Initialization scripts run once on first container start (alphabetical order)
COPY init/ /docker-entrypoint-initdb.d/
