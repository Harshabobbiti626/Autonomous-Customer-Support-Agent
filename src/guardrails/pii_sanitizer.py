import re


class PIISanitizer:
    """Masks sensitive personally identifiable information (PII) before LLM ingestion and logging."""

    # Credit card regex (Visa, MasterCard, Amex, Discover, standard 13-19 digits with separators)
    CC_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b")

    # Social Security Number (US SSN: XXX-XX-XXXX)
    SSN_PATTERN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")

    # US / International Phone pattern
    PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

    # Email pattern (optional masking, often preserved for auth or redacted in public logs)
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

    @classmethod
    def sanitize(cls, text: str, mask_emails: bool = False) -> str:
        """Sanitize text by replacing sensitive patterns with redaction placeholders."""
        if not text:
            return ""

        sanitized = cls.CC_PATTERN.sub("[REDACTED_CARD]", text)
        sanitized = cls.SSN_PATTERN.sub("[REDACTED_SSN]", sanitized)
        sanitized = cls.PHONE_PATTERN.sub("[REDACTED_PHONE]", sanitized)

        if mask_emails:
            sanitized = cls.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)

        return sanitized
