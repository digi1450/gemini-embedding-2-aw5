# Gemini Embedding 2 (AW5) Dify plugin

Private model-provider plugin for the AW5 multimodal RAG pilot. It exposes only
Google's stable `gemini-embedding-2` model and can coexist with Dify's official
Gemini plugin.

## Why this fork exists

The official Dify Gemini plugin currently lists the preview model but not the
stable model. Stable Gemini Embedding 2 also rejects the legacy `task_type`
request field. This fork therefore:

- registers `gemini-embedding-2` as a vision-capable text-embedding model;
- formats RAG queries as `task: question answering | query: ...`;
- formats text documents as `title: none | text: ...`;
- omits `task_type` for the stable model;
- requests 1536-dimensional embeddings for both text and images;
- splits multimodal requests so each Gemini call contains at most six images;
- limits sustained model operations and retries temporary quota/server errors;
- estimates chunk tokens locally instead of consuming Gemini model-operation quota.

## Current version

Version `0.1.4` is the AW5 deployment candidate. Its unit suite currently passes
61 tests, with three credential-dependent live smoke tests skipped unless a local
Remote Debug environment is configured.

## Safe development flow

1. Copy `.env.example` to `.env` and paste the Remote Debug values from Dify
   Cloud. Never commit `.env`.
2. Run `uv sync --dev`.
3. Run `uv run pytest -m "not live"`.
4. Run `uv run python -m main` for Dify Cloud remote debugging.
5. After the text and image smoke tests pass, package with the Dify plugin CLI
   and install the generated `.difypkg` via **Plugins > Install Plugin > Via
   Local File**.

Do not index the full 474-page SDS until a small multimodal pilot passes.

## Upstream provenance

Derived from `langgenius/dify-official-plugins`, Gemini plugin commit
`c41f1679f41ccffdc35b363123ac89f583a9d88c` (retrieved 2026-08-19).
