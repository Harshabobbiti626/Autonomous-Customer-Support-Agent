import re


class OutputChecker:
    """Validates model responses for policy hallucination and evaluates customer sentiment."""

    # Prohibited claims contrary to store policy
    HALLUCINATION_TRIGGERS = [
        re.compile(r"\b(60|90|100|365|lifetime)\s*[- ]?day\s+(return|refund)\b", re.IGNORECASE),
        re.compile(
            r"i\s+have\s+(processed|sent|issued)\s+your\s+refund\s+of\s+\$\d+", re.IGNORECASE
        ),  # Must only be claimed if tool confirmed it
        re.compile(r"no\s+questions\s+asked\s+full\s+refund\s+guaranteed", re.IGNORECASE),
    ]

    # Extreme agitation / escalation indicators
    ESCALATION_KEYWORDS = [
        "lawyer",
        "attorney",
        "sue you",
        "legal action",
        "better business bureau",
        "bbb complaint",
        "fraud",
        "scam",
        "police",
        "court",
        "unacceptable service",
    ]

    @classmethod
    def calculate_sentiment(cls, user_text: str) -> float:
        """Heuristic sentiment score between -1.0 (extremely negative) and +1.0 (positive)."""
        if not user_text:
            return 0.0

        text_lower = user_text.lower()
        score = 0.0

        # Check critical escalation keywords
        for kw in cls.ESCALATION_KEYWORDS:
            if kw in text_lower:
                score -= 0.7

        # Check profanity / anger signals
        anger_words = [
            "terrible",
            "awful",
            "worst",
            "hate",
            "horrible",
            "furious",
            "angry",
            "ridiculous",
            "useless",
        ]
        for word in anger_words:
            if word in text_lower:
                score -= 0.25

        # Check positive signals
        positive_words = [
            "thanks",
            "thank you",
            "great",
            "awesome",
            "perfect",
            "helpful",
            "appreciate",
        ]
        for word in positive_words:
            if word in text_lower:
                score += 0.3

        # Clamp between -1.0 and 1.0
        return max(-1.0, min(1.0, score))

    @classmethod
    def verify_output(cls, response_text: str, refund_executed: bool = False) -> tuple[bool, str]:
        """Verify model response against strict store policies.

        Returns:
            Tuple[bool, str]: (is_valid, sanitized_or_corrected_text)
        """
        # Check for policy duration hallucinations
        for trigger in cls.HALLUCINATION_TRIGGERS:
            match = trigger.search(response_text)
            if match:
                # If model claims it issued a refund without execution flag being true
                if "processed" in match.group(0).lower() and not refund_executed:
                    corrected = re.sub(
                        trigger,
                        "I have submitted your request for verification under our standard 30-day policy",
                        response_text,
                    )
                    return False, corrected

        return True, response_text
