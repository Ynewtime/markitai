# MarkIt - Document to Markdown Converter
# Multi-stage build for optimal image size

# Stage 1: Build oxipng from source
# Note: oxipng is not available in apt. Options:
#   - cargo install (used here, smallest image size)
#   - brew install (requires full Homebrew, larger image)
#   - download prebuilt binary from GitHub releases
FROM rust:slim AS rust-builder

RUN cargo install oxipng

# Stage 2: Build Python dependencies
FROM python:3.13-slim AS python-builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies (including OCR support)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[ocr]"

# Stage 3: Runtime stage
FROM python:3.13-slim AS runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Pandoc for document conversion
    pandoc \
    # LibreOffice for legacy format support (headless mode)
    libreoffice-writer-nogui \
    libreoffice-calc-nogui \
    libreoffice-impress-nogui \
    # Image compression tools (libjpeg-turbo)
    libjpeg-turbo-progs \
    # EMF/WMF support
    libwmf-bin \
    # python-magic dependency
    libmagic1 \
    # OpenCV dependencies (for OCR support)
    libgl1 \
    libglib2.0-0 \
    # Fonts for document rendering
    fonts-liberation \
    fonts-dejavu-core \
    # CJK fonts for Chinese/Japanese/Korean document support
    fonts-noto-cjk \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy oxipng binary from rust builder
COPY --from=rust-builder /usr/local/cargo/bin/oxipng /usr/local/bin/oxipng

# Copy Python packages from builder
COPY --from=python-builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY src/markit/ ./src/markit/
COPY pyproject.toml README.md ./

# Install the application
RUN pip install -e . --no-deps

# Create non-root user for security
RUN useradd -m -u 1000 markit
USER markit

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default output directory
VOLUME ["/data/input", "/data/output"]

# Entry point
ENTRYPOINT ["python", "-m", "markit"]

# Default command (show help)
CMD ["--help"]

# Labels
LABEL org.opencontainers.image.title="MarkIt"
LABEL org.opencontainers.image.description="Intelligent Document to Markdown Converter"
LABEL org.opencontainers.image.version="0.1.5"
LABEL org.opencontainers.image.source="https://github.com/Ynewtime/markit"
