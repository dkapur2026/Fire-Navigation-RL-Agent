FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY blueprints ./blueprints
COPY tests ./tests
COPY scripts ./scripts
COPY docs ./docs

RUN pip install --no-cache-dir -e ".[dev]"

# Default: show that the world runs end to end (no GPU, no extras needed).
CMD ["emberpath", "run", "--profile", "single", "--novelty", "door_jam", "--agent", "oracle", "--seed", "1"]
