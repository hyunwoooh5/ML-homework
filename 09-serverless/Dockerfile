FROM agrigorev/model-2025-hairstyle:v1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --locked

COPY app.py ./

CMD ["uv", "run", "python", "app.py"]

# To avoid lambda entrypoint
ENTRYPOINT ["/bin/sh", "-c"]

CMD ["uv run python app.py"]
