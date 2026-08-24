import base64
import binascii
import logging
import random
import re
import threading
import time
from collections.abc import Mapping
from typing import Optional, Union

import numpy as np
from google import genai
from google.genai import errors as genai_errors, types
from google.genai.types import EmbedContentConfig

from dify_plugin import TextEmbeddingModel
from dify_plugin.entities.model import EmbeddingInputType, PriceType
from dify_plugin.entities.model.text_embedding import (
    EmbeddingUsage,
    MultiModalContent,
    MultiModalContentType,
    MultiModalEmbeddingResult,
    TextEmbeddingResult,
)
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError

from ..common_gemini import _CommonGemini

logger = logging.getLogger(__name__)

# Embedding and number of tokens used
EmbeddingTokenPair = tuple[list[float], Optional[int]]

TASK_TYPE_BY_INPUT_TYPE = {
    EmbeddingInputType.DOCUMENT: "RETRIEVAL_DOCUMENT",
    EmbeddingInputType.QUERY: "RETRIEVAL_QUERY",
}

STABLE_EMBEDDING_2_MODEL = "gemini-embedding-2"
STABLE_EMBEDDING_2_OUTPUT_DIMENSION = 1536

# Keep sustained traffic comfortably below the observed project quota of
# 200 model operations per minute. The limiter is shared by every model
# instance in this plugin process, including text, image, and credential calls.
REQUESTS_PER_MINUTE = 120
MIN_REQUEST_INTERVAL_SECONDS = 60.0 / REQUESTS_PER_MINUTE
MAX_RETRIES = 8
MAX_RETRY_DELAY_SECONDS = 60.0
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

_RATE_LIMIT_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0


class GeminiTextEmbeddingModel(_CommonGemini, TextEmbeddingModel):
    """
    Model class for Gemini text embedding model.
    """

    # ---- Gemini Embedding 2 modality-specific limits ----
    # https://ai.google.dev/gemini-api/docs/embeddings#modality-limits
    # Image: max 6 per request, PNG/JPEG only
    MAX_IMAGES_PER_REQUEST = 6
    SUPPORTED_IMAGE_FORMATS = {"image/jpeg", "image/png"}
    # Audio: max 80 seconds, MP3/WAV (not yet supported)
    # Video: max 128 seconds, MP4/MOV, codecs: H264/H265/AV1/VP9 (not yet supported)
    # Document (PDF): max 6 pages (not yet supported)

    # Fallback token estimate for image content when API does not return statistics.
    # Google's documentation indicates images are processed at ~258 tokens on average.
    IMAGE_TOKEN_ESTIMATE = 258

    @staticmethod
    def _wait_for_rate_slot() -> None:
        """Reserve one process-wide API request slot and wait until it is due."""
        global _NEXT_REQUEST_AT

        with _RATE_LIMIT_LOCK:
            now = time.monotonic()
            request_at = max(now, _NEXT_REQUEST_AT)
            _NEXT_REQUEST_AT = request_at + MIN_REQUEST_INTERVAL_SECONDS

        delay = request_at - now
        if delay > 0:
            time.sleep(delay)

    def _embed_content_with_retry(
        self,
        client: genai.Client,
        *,
        model: str,
        contents,
        config: Optional[EmbedContentConfig] = None,
    ):
        """Call Gemini embedding with throttling and bounded exponential backoff."""
        for attempt in range(MAX_RETRIES + 1):
            self._wait_for_rate_slot()
            try:
                return client.models.embed_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except genai_errors.APIError as ex:
                if ex.code not in RETRYABLE_STATUS_CODES or attempt >= MAX_RETRIES:
                    raise

                delay = min(MAX_RETRY_DELAY_SECONDS, 2**attempt)
                delay += random.uniform(0.0, 1.0)
                logger.warning(
                    "Gemini embedding request failed with HTTP %s; retrying in %.1fs "
                    "(attempt %s/%s)",
                    ex.code,
                    delay,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(delay)

        raise InvokeError(f"Unable to get embeddings from '{model}' model")

    @staticmethod
    def _as_user_content(part: Union[str, types.Part]) -> types.UserContent:
        return types.UserContent(parts=[part])

    @staticmethod
    def _prepare_stable_text(text: str, input_type: EmbeddingInputType) -> str:
        """Apply Google's asymmetric retrieval format for Gemini Embedding 2."""
        if input_type == EmbeddingInputType.QUERY:
            return f"task: question answering | query: {text}"
        return f"title: none | text: {text}"

    def _prepare_text(
        self, model: str, text: str, input_type: EmbeddingInputType
    ) -> str:
        if model == STABLE_EMBEDDING_2_MODEL:
            return self._prepare_stable_text(text, input_type)
        return text

    @staticmethod
    def _embedding_config(
        model: str,
        input_type: EmbeddingInputType,
        output_dimension: Optional[int] = None,
    ) -> Optional[EmbedContentConfig]:
        """Build an API config without sending unsupported stable-model fields."""
        if model == STABLE_EMBEDDING_2_MODEL:
            return EmbedContentConfig(
                output_dimensionality=(
                    output_dimension or STABLE_EMBEDDING_2_OUTPUT_DIMENSION
                )
            )

        task_type = TASK_TYPE_BY_INPUT_TYPE.get(input_type)
        config_kwargs = {}
        if task_type:
            config_kwargs["task_type"] = task_type
        if output_dimension:
            config_kwargs["output_dimensionality"] = output_dimension
        return EmbedContentConfig(**config_kwargs) if config_kwargs else None

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        """
        Invoke text embedding model

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :param user: unique user id
        :return: embeddings result
        """
        client = genai.Client(api_key=credentials["google_api_key"])

        # get model properties
        context_size = self._get_context_size(model, credentials)
        max_chunks = self._get_max_chunks(model, credentials)

        # splitted texts in case the chunks are bigger than the context size
        splitted_texts = [
            self._split_texts_to_fit_model_specs(client, model, [text], context_size)
            for text in texts
        ]

        # list batched texts of size <= max_chunks containing (text index, text)
        batched_texts: list[list[tuple[int, str]]] = [[]]
        for i, splitted_text in enumerate(splitted_texts):
            for text, _ in splitted_text:
                if len(batched_texts[-1]) >= max_chunks:
                    batched_texts.append([])
                batched_texts[-1].append((i, text))

        # list of embeddings following the same arrangement as splitted_texts
        splitted_embeddings: list[list[EmbeddingTokenPair]] = []
        for batch in batched_texts:
            embeddings_batch = self._embedding_invoke(
                model=model,
                client=client,
                texts=[text for _, text in batch],
                input_type=input_type,
            )
            for i, (j, _) in enumerate(batch):
                if j >= len(splitted_embeddings):
                    splitted_embeddings.append([])
                splitted_embeddings[j].append(embeddings_batch[i])

        # merge embeddings by averaging them
        merged_embeddings: list[list[float]] = []
        used_tokens = 0
        for i, embeddings in enumerate(splitted_embeddings):
            embeddings, num_tokens = zip(*embeddings)
            if len(embeddings) == 1:
                embedding = embeddings[0]
            else:
                # `num_tokens` may contain ``None`` when the Gemini API omits token
                # usage for a chunk. ``np.average`` cannot use ``None`` weights, so
                # fall back to an unweighted average in that case.
                weights = num_tokens if all(t is not None for t in num_tokens) else None
                average = np.average(embeddings, axis=0, weights=weights)
                embedding = (average / np.linalg.norm(average)).tolist()
                if np.isnan(embedding).any():
                    raise ValueError("Normalized embedding is nan please try again")
            merged_embeddings.append(embedding)
            # sum up the number of tokens used if available or the count estimation from the text chunking
            used_tokens += sum(
                [
                    used_token or chunk_size
                    for used_token, [_, chunk_size] in zip(
                        num_tokens, splitted_texts[i]
                    )
                ]
            )

        # calc usage
        usage = self._calc_response_usage(
            model=model, credentials=credentials, tokens=used_tokens
        )

        return TextEmbeddingResult(
            embeddings=merged_embeddings, usage=usage, model=model
        )

    def _split_texts_to_fit_model_specs(
        self, client: genai.Client, model: str, texts: list[str], context_size: int
    ) -> list[tuple[str, int]]:
        """
        Split text to fit model specs based on the model context size

        :param client: model client
        :param model: model name
        :param text: text to truncate
        :return: list of tuples (text, estimated chunk size)
        """
        splitted_text = []
        for text in texts:
            num_tokens = self._count_tokens(client, model, text)
            if num_tokens >= context_size and len(text) > 1:
                # `context_size` is a token budget, not a character index. Estimate a
                # character cutoff from the token-to-character ratio so the head is
                # likely to fit, then clamp it to ``[1, len(text) - 1]`` so both the
                # head and the tail are strictly shorter than ``text``. This guarantees
                # progress and terminates the recursion even for token-dense content
                # (e.g. CJK, code, base64) where ``len(text)`` may be <= ``context_size``.
                cutoff = max(1, len(text) * context_size // max(1, num_tokens))
                cutoff = min(cutoff, len(text) - 1)
                # prefer to split on the closest punctuation mark, then comma, then
                # whitespace, searching forward from the estimated cutoff. Never let the
                # boundary reach the end of the text, which would empty the tail.
                for pattern in [r"[.!?]", r",", r"\s"]:
                    match = re.search(pattern, text[cutoff:])
                    if match:
                        boundary = cutoff + match.start() + 1
                        if boundary < len(text):
                            cutoff = boundary
                        break
                splitted_text.extend(
                    self._split_texts_to_fit_model_specs(
                        client, model, [text[:cutoff]], context_size
                    )
                )
                splitted_text.extend(
                    self._split_texts_to_fit_model_specs(
                        client, model, [text[cutoff:]], context_size
                    )
                )
            else:
                splitted_text.append((text, num_tokens))
        return splitted_text

    def get_num_tokens(
        self, model: str, credentials: dict, texts: list[str]
    ) -> list[int]:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :return: list of estimated token counts
        """
        # Use _get_num_tokens_by_gpt2 as it provides a faster estimation of token counts
        # compared to using the count_tokens action for each text.
        return [self._get_num_tokens_by_gpt2(text) for text in texts]

    def _count_tokens(self, client: genai.Client, model: str, text: str) -> int:
        """
        Estimate tokens locally without consuming Gemini model-operation quota.

        :param client: model client
        :param model: model name
        :param text: text to embed
        :return: estimated token count
        """
        del client, model
        return max(1, self._get_num_tokens_by_gpt2(text))

    def validate_credentials(self, model: str, credentials: Mapping) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            client = genai.Client(api_key=credentials["google_api_key"])
            self._embed_content_with_retry(
                client,
                model=model,
                contents=["ping"],
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _embedding_invoke(
        self,
        model: str,
        client: genai.Client,
        texts: Union[list[str], str],
        input_type: EmbeddingInputType,
    ) -> list[EmbeddingTokenPair]:
        """
        Invoke embedding model

        :param model: model name
        :param client: model client
        :param texts: texts to embed
        :param extra_model_kwargs: extra model kwargs
        :return: embeddings and used tokens
        """

        # call embedding model
        logical_texts = texts if isinstance(texts, list) else [texts]
        prepared_texts = [
            self._prepare_text(model, text, input_type) for text in logical_texts
        ]
        contents = [self._as_user_content(text) for text in prepared_texts]
        config = self._embedding_config(model, input_type)
        response = self._embed_content_with_retry(
            client,
            model=model,
            contents=contents,
            config=config,
        )

        if response.embeddings is None:
            raise InvokeError(f"Unable to get embeddings from '{model}' model")

        if len(response.embeddings) != len(logical_texts):
            raise InvokeError(
                f"Expected {len(logical_texts)} embeddings from '{model}' model, "
                f"got {len(response.embeddings)}"
            )

        result: list[tuple[list[float], Optional[int]]] = []
        for embedding in response.embeddings:
            embeddings = embedding.values or []
            used_tokens = (
                embedding.statistics.token_count if embedding.statistics else None
            )
            result.append((embeddings, int(used_tokens) if used_tokens else None))

        return result

    def _calc_response_usage(
        self, model: str, credentials: dict, tokens: int
    ) -> EmbeddingUsage:
        """
        Calculate response usage

        :param model: model name
        :param credentials: model credentials
        :param tokens: input tokens
        :return: usage
        """
        # get input price info
        input_price_info = self.get_price(
            model=model,
            credentials=credentials,
            price_type=PriceType.INPUT,
            tokens=tokens,
        )

        # transform usage
        usage = EmbeddingUsage(
            tokens=tokens,
            total_tokens=tokens,
            unit_price=input_price_info.unit_price,
            price_unit=input_price_info.unit,
            total_price=input_price_info.total_amount,
            currency=input_price_info.currency,
            latency=time.perf_counter() - self.started_at,
        )

        return usage

    def _detect_image_mime_type(
        self, base64_str: str, validate_format: bool = False
    ) -> str:
        """
        Detect image MIME type from base64 string

        :param base64_str: base64 string
        :param validate_format: if True, raise error for unsupported formats
        :return: MIME type (e.g., 'image/jpeg', 'image/png')
        """
        try:
            # Remove data URI prefix if present
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]

            data = base64.b64decode(base64_str, validate=True)

            # Check file signatures
            if data.startswith(b"\xff\xd8\xff"):
                return "image/jpeg"
            elif data.startswith(b"\x89PNG\r\n\x1a\n"):
                return "image/png"
            elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
                mime = "image/gif"
            elif data.startswith(b"WEBP", 8):
                mime = "image/webp"
            else:
                mime = "image/jpeg"

            # Validate format for Gemini Embedding 2 (only JPEG and PNG are supported)
            if validate_format and mime not in self.SUPPORTED_IMAGE_FORMATS:
                raise ValueError(
                    f"Unsupported image format: {mime}. "
                    f"Gemini Embedding 2 only supports: {', '.join(sorted(self.SUPPORTED_IMAGE_FORMATS))}"
                )
            return mime
        except ValueError:
            raise
        except binascii.Error:
            logger.warning(
                "Failed to decode base64 image data, defaulting to image/jpeg",
                exc_info=True,
            )
            return "image/jpeg"

    def _get_output_dimension(self, model: str, credentials: dict) -> Optional[int]:
        """
        Get output dimension from model properties (for MRL support)

        :param model: model name
        :param credentials: model credentials
        :return: output dimension if configured, None otherwise
        """
        try:
            model_schema = self.get_model_schema(model, credentials)
            if model_schema and model_schema.model_properties:
                return model_schema.model_properties.get("output_dimension")
        except Exception:
            logger.warning(
                "Failed to get output_dimension from model schema", exc_info=True
            )
        return None

    def _invoke_multimodal(
        self,
        model: str,
        credentials: dict,
        documents: list[MultiModalContent],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> MultiModalEmbeddingResult:
        """
        Invoke multimodal embedding model

        :param model: model name
        :param credentials: model credentials
        :param documents: multimodal documents to embed
        :param user: unique user id
        :param input_type: input type
        :return: embeddings result
        """
        self.started_at = time.perf_counter()
        client = genai.Client(api_key=credentials["google_api_key"])

        # Convert MultiModalContent to Google Genai format, tracking content types
        contents = []  # converted content for API call
        content_is_image = []  # parallel list: True if image, False if text
        original_texts = []  # parallel list: original text string (or None for images)
        for document in documents:
            if document.content_type == MultiModalContentType.TEXT:
                prepared_text = self._prepare_text(
                    model, document.content, input_type
                )
                contents.append(self._as_user_content(prepared_text))
                content_is_image.append(False)
                original_texts.append(prepared_text)
            elif document.content_type == MultiModalContentType.IMAGE:
                # Validate image format (Gemini Embedding 2 only supports JPEG and PNG)
                mime_type = self._detect_image_mime_type(
                    document.content, validate_format=True
                )
                # Decode base64 and create Part object
                base64_str = document.content
                if "," in base64_str:
                    base64_str = base64_str.split(",", 1)[1]
                image_data = base64.b64decode(base64_str)
                part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
                contents.append(self._as_user_content(part))
                content_is_image.append(True)
                original_texts.append(None)
            else:
                raise ValueError(
                    f"Unsupported content type: {document.content_type}. "
                    f"Gemini Embedding 2 currently supports TEXT and IMAGE."
                )

        max_chunks = self._get_max_chunks(model, credentials)

        # Batch processing if needed
        embeddings = []
        used_tokens = 0

        # Build batches without changing document order. Gemini Embedding 2
        # accepts at most six images in one request, while Dify may send many
        # more images in a single multimodal invocation.
        batches = []
        batch_contents = []
        batch_is_image = []
        batch_original_texts = []
        batch_image_count = 0

        for content, is_image, original_text in zip(
            contents, content_is_image, original_texts, strict=True
        ):
            exceeds_chunk_limit = len(batch_contents) >= max_chunks
            exceeds_image_limit = (
                is_image and batch_image_count >= self.MAX_IMAGES_PER_REQUEST
            )
            if batch_contents and (exceeds_chunk_limit or exceeds_image_limit):
                batches.append(
                    (batch_contents, batch_is_image, batch_original_texts)
                )
                batch_contents = []
                batch_is_image = []
                batch_original_texts = []
                batch_image_count = 0

            batch_contents.append(content)
            batch_is_image.append(is_image)
            batch_original_texts.append(original_text)
            if is_image:
                batch_image_count += 1

        if batch_contents:
            batches.append((batch_contents, batch_is_image, batch_original_texts))

        # Prepare config with optional output_dimension (MRL support)
        output_dimension = self._get_output_dimension(model, credentials)
        config = self._embedding_config(
            model, input_type, output_dimension=output_dimension
        )

        # Process each safe-sized batch and concatenate the results in order.
        for batch_contents, batch_is_image, batch_original_texts in batches:

            # Call embedding API
            response = self._embed_content_with_retry(
                client,
                model=model,
                contents=batch_contents,
                config=config,
            )

            if response.embeddings is None:
                raise InvokeError(f"Unable to get embeddings from '{model}' model")

            if len(response.embeddings) != len(batch_contents):
                raise InvokeError(
                    f"Expected {len(batch_contents)} embeddings from '{model}' model, "
                    f"got {len(response.embeddings)}"
                )

            # Process embeddings
            for j, embedding in enumerate(response.embeddings):
                embedding_values = embedding.values or []
                embeddings.append(embedding_values)

                # Count tokens: prefer API statistics, then estimate by content type
                if embedding.statistics and embedding.statistics.token_count:
                    used_tokens += embedding.statistics.token_count
                elif batch_is_image[j]:
                    # Image: use fixed estimate
                    used_tokens += self.IMAGE_TOKEN_ESTIMATE
                elif batch_original_texts[j] is not None:
                    # Text: estimate using GPT-2 tokenizer
                    used_tokens += self._get_num_tokens_by_gpt2(batch_original_texts[j])
                else:
                    # Final fallback
                    used_tokens += self.IMAGE_TOKEN_ESTIMATE

        # Calculate usage
        usage = self._calc_response_usage(
            model=model, credentials=credentials, tokens=used_tokens
        )

        return MultiModalEmbeddingResult(
            model=model,
            embeddings=embeddings,
            usage=usage,
        )
