"""Billable smoke tests for the real Gemini Embedding 2 API."""

import os
from io import BytesIO

import pytest
from google import genai
from google.genai import types
from google.genai.types import EmbedContentConfig
from PIL import Image


MODEL = "gemini-embedding-2"
DIMENSION = 1536
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="Set GEMINI_API_KEY to run billable live tests",
    ),
]


@pytest.fixture(scope="module")
def client():
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _content(part: str | types.Part) -> types.UserContent:
    return types.UserContent(parts=[part])


def _tiny_png() -> bytes:
    image = Image.new("RGB", (8, 8), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_stable_text_query_and_document(client):
    contents = [
        _content("task: question answering | query: สถานะล็อกบัญชีอยู่คอลัมน์ใด"),
        _content("title: none | text: Table UM_USER { IS_LOCK varchar(1) }"),
    ]
    response = client.models.embed_content(
        model=MODEL,
        contents=contents,
        config=EmbedContentConfig(output_dimensionality=DIMENSION),
    )

    assert response.embeddings is not None
    assert len(response.embeddings) == 2
    assert all(len(embedding.values or []) == DIMENSION for embedding in response.embeddings)


def test_stable_png_embedding(client):
    image_part = types.Part.from_bytes(data=_tiny_png(), mime_type="image/png")
    response = client.models.embed_content(
        model=MODEL,
        contents=[_content(image_part)],
        config=EmbedContentConfig(output_dimensionality=DIMENSION),
    )

    assert response.embeddings is not None
    assert len(response.embeddings) == 1
    assert len(response.embeddings[0].values or []) == DIMENSION


def test_stable_mixed_batch_returns_one_vector_per_content(client):
    image_part = types.Part.from_bytes(data=_tiny_png(), mime_type="image/png")
    response = client.models.embed_content(
        model=MODEL,
        contents=[
            _content("task: question answering | query: รูปสี่เหลี่ยมสีแดง"),
            _content(image_part),
        ],
        config=EmbedContentConfig(output_dimensionality=DIMENSION),
    )

    assert response.embeddings is not None
    assert len(response.embeddings) == 2
    assert all(len(embedding.values or []) == DIMENSION for embedding in response.embeddings)

