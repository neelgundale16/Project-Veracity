import re
import difflib
import logging
from models import VaultResult, NLIResult

logger = logging.getLogger(__name__)

# REVERSE algorithm constants
TAU_GENERATIVE      = 0.003
TAU_DISCRIMINATIVE  = 0.5
MAX_RESAMPLE        = 3
TEMPERATURE_STEP    = 0.1
MAX_TEMP_BOOST      = 0.5

# Patterns to extract entities from text
NUMBER_PATTERN = re.compile(
    r'\$[\d,.]+\s*(?:billion|million|trillion|thousand)?'
    r'|\d+(?:\.\d+)?%'
    r'|\d+(?:\.\d+)?\s*(?:billion|million|trillion)'
    r'|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b',
    re.IGNORECASE
)
DATE_PATTERN = re.compile(
    r'\bQ[1-4]\s*\d{4}\b'
    r'|\b(?:January|February|March|April|May|June|July|'
    r'August|September|October|November|December)\s+\d{4}\b'
    r'|\bFY\s*\d{4}\b'
    r'|\bfiscal\s+year\s+\d{4}\b',
    re.IGNORECASE
)


class ReverseRewriter:

    def _extract_entities(self, text: str) -> dict:
        """Extract numbers, dates from text."""
        return {
            "numbers": NUMBER_PATTERN.findall(text),
            "dates":   DATE_PATTERN.findall(text),
        }

    def _replace_entity(self, original: str,
                        wrong_val: str, correct_val: str) -> str:
        """Replace wrong value with correct value in original sentence."""
        # Try exact match first
        if wrong_val in original:
            return original.replace(wrong_val, correct_val, 1)

        # Try case-insensitive
        pattern = re.compile(re.escape(wrong_val), re.IGNORECASE)
        result  = pattern.sub(correct_val, original, count=1)
        if result != original:
            return result

        # Fuzzy fallback using difflib
        words       = original.split()
        wrong_words = wrong_val.split()
        matches     = difflib.get_close_matches(
            wrong_words[0], words, n=1, cutoff=0.6
        )
        if matches:
            idx = words.index(matches[0])
            words[idx] = correct_val
            return " ".join(words)

        return original

    def rewrite(self, original_sentence: str,
                vault_result: VaultResult,
                nli_result: NLIResult) -> str:
        """
        REVERSE algorithm — rewrite hallucinated sentence with correct facts.

        Strategy 1: Find mismatched numeric values and swap them.
        Strategy 2: Build corrected sentence from vault matched text.
        Strategy 3: Fallback — return vault matched text directly.
        """
        original_entities = self._extract_entities(original_sentence)
        correct_entities  = self._extract_entities(vault_result.matched_text)

        # Strategy 1 — Direct numeric entity replacement
        corrected = original_sentence
        attempt   = 0

        orig_numbers    = original_entities["numbers"]
        correct_numbers = correct_entities["numbers"]

        if orig_numbers and correct_numbers:
            for i, wrong_num in enumerate(orig_numbers):
                if i < len(correct_numbers):
                    correct_num = correct_numbers[i]
                    if wrong_num.strip() != correct_num.strip():
                        new_sentence = self._replace_entity(
                            corrected, wrong_num, correct_num
                        )
                        if new_sentence != corrected:
                            corrected = new_sentence
                            attempt  += 1
                            logger.debug(
                                f"Strategy 1 replaced: "
                                f"'{wrong_num}' → '{correct_num}'"
                            )

        if attempt > 0:
            return corrected

        # Strategy 2 — Use vault sentence structure
        # Keep subject from original, swap predicate from vault
        orig_lower  = original_sentence.lower()
        vault_lower = vault_result.matched_text.lower()

        matcher = difflib.SequenceMatcher(None, orig_lower, vault_lower)
        if matcher.ratio() > 0.4:
            # Sentences are about the same topic — return vault version
            return vault_result.matched_text

        # Strategy 3 — Fallback: Append inline warning instead of injecting raw PDF blob
        logger.debug("Strategy 3 fallback: appending inline warning to original text")
        # Ensure we don't duplicate warnings if the LLM generated multiple sentences
        warning = " [⚠️ Contradicts Vault]"
        if warning not in original_sentence:
            return f"{original_sentence}{warning}"
        return original_sentence

    def build_correction_payload(self, original: str,
                                  corrected: str, source: str) -> dict:
        """Build the correction metadata dict sent to frontend."""
        ratio = difflib.SequenceMatcher(None, original, corrected).ratio()
        return {
            "original":   original,
            "corrected":  corrected,
            "source":     source,
            "diff_ratio": round(ratio, 4),
        }


# Singleton
rewriter = ReverseRewriter()