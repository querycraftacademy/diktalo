"""Opcionális átirat-javítás MiniMax LLM-mel (felhős), biztonsági ellenőrzéssel.

Alapból ki van kapcsolva (config: "ai_correction": false). A kulcsot a
MINIMAX_API_KEY környezeti változóból olvassuk, fájlban nem tároljuk.

A javító csak akkor kap zöld utat, ha nem írja át a szavaidat: a kimenetet
összevetjük a nyers átirattal, és ha túl sok szó megváltozott, eldobjuk a
javítást, és a nyers szöveg megy tovább. Így az írásjeleket és az ékezeteket
megkapjuk, de a modell nem tud mást a szádba adni.
"""
import re
import unicodedata

import requests

LANG_NAMES = {"hu": "Hungarian", "en": "English", "de": "German", "": "the same language as the input"}

SYSTEM_PROMPT = """You are a post-processor for speech-to-text (Whisper) output. The user dictates in {lang_name}.

Your job:
- Fix punctuation, capitalization and diacritics.
- Fix obvious speech-recognition errors (misheard words, wrong word boundaries) — but ONLY when the surrounding context makes the intended word clear beyond reasonable doubt.
- Keep the meaning, wording and style of the speaker. Do NOT rephrase, shorten, expand or translate.
- If the transcript contains a spoken correction like "no wait, I mean X", apply it.
- Numbers, e-mail addresses and URLs should be written in their conventional written form.
{vocab}
Strict rules:
- When you are not sure a word is a recognition error, ALWAYS keep the original word. A wrong "correction" is worse than leaving a transcription error in place.
- Never substitute a name, brand or product with a different one that was not spoken, unless it is in the vocabulary list above and the transcript contains a near-homophone of it.
- The user may mix short English phrases into the text; keep them in English.
- If a "Context" block is provided, it is what the user dictated previously. Use it to resolve ambiguities, but NEVER include or repeat it in your output.

Output ONLY the corrected text of the new transcript, with no quotes, no explanation, no preamble."""

PUNCT_ONLY_PROMPT = """You restore punctuation in speech-to-text output. The user dictates in {lang_name}.

Allowed changes, nothing else:
- Add or fix punctuation (commas, periods, question marks, hyphens, apostrophes).
- Fix capitalization (sentence starts, proper nouns).
- Fix missing or wrong diacritics (accents) on words.
- Join or split a word only where the language requires it (compound words).

Forbidden: replacing a word with a different word, adding words, deleting words,
rephrasing, translating, or "improving" the style. Keep every word the speaker said,
even if it looks like a recognition error.
{vocab}
Output ONLY the resulting text, with no quotes, no explanation, no preamble."""


def _cmp_tokens(text: str) -> list[str]:
    """Szavak összehasonlító alakja: kisbetű, ékezet és írásjel nélkül."""
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]", " ", s).split()


def _word_edits(a: list[str], b: list[str]) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


class Corrector:
    def __init__(
        self,
        cfg: dict,
        language: str = "hu",
        vocabulary: list[str] | None = None,
    ):
        self.api_key = cfg.get("api_key", "")
        self.base_url = cfg.get("base_url", "https://api.minimax.io/anthropic").rstrip("/")
        self.model = cfg.get("model", "MiniMax-M3")
        # "anthropic": /v1/messages + thinking kikapcsolva (MiniMax-M3: ~1-2 s)
        # "openai":    /chat/completions (más szolgáltatókhoz, pl. Groq)
        self.api_style = cfg.get("api_style", "anthropic")
        self.timeout = cfg.get("timeout", 45)
        # "balanced": szócserét is engedünk, de csak a szavak max_word_change részéig
        # "punctuation_only": a szavak nem változhatnak, csak írásjel/ékezet/nagybetű
        self.mode = cfg.get("mode", "balanced")
        self.max_word_change = cfg.get("max_word_change", 0.25)
        vocab = ""
        if vocabulary:
            vocab = (
                "- The following names/terms occur often and are frequently misrecognized; "
                "prefer them when the transcript contains something similar: "
                + ", ".join(vocabulary)
                + "\n"
            )
        lang_name = LANG_NAMES.get(language, language or "the input language")
        template = PUNCT_ONLY_PROMPT if self.mode == "punctuation_only" else SYSTEM_PROMPT
        self.system_prompt = template.format(lang_name=lang_name, vocab=vocab)

    def _accept(self, raw: str, fixed: str) -> tuple[bool, str]:
        """Elfogadható-e a javítás? (igen/nem, indok)"""
        a, b = _cmp_tokens(raw), _cmp_tokens(fixed)
        if a == b:
            return True, "csak írásjel/ékezet"
        if self.mode == "punctuation_only":
            return False, "szavakat írt át, pedig csak írásjel lenne szabad"
        if not a:
            return False, "üres bemenet"
        changed = _word_edits(a, b)
        ratio = changed / len(a)
        if ratio > self.max_word_change:
            return False, f"a szavak {ratio:.0%}-át átírta (határ {self.max_word_change:.0%})"
        return True, f"{changed} szó változott"

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def correct(self, text: str, context: str = "") -> str:
        """Ellenőrzött javítást ad vissza; hiba vagy gyanús átírás esetén az eredetit."""
        if not text.strip() or not self.available:
            return text
        fixed = self._raw_call(text, context)
        if not fixed:
            return text
        ok, reason = self._accept(text, fixed)
        if not ok:
            print(f"[MiniMax] Javítás ELDOBVA ({reason}), a nyers átirat megy tovább.")
            return text
        return fixed

    def _raw_call(self, text: str, context: str = "", system: str | None = None) -> str:
        """A modell nyers válasza, ellenőrzés nélkül (üres string hiba esetén)."""
        prompt = system or self.system_prompt
        message = text
        if context:
            message = f"Context (previous dictation, do not repeat):\n{context}\n\nNew transcript to correct:\n{text}"
        try:
            if self.api_style == "anthropic":
                resp = requests.post(
                    f"{self.base_url}/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "system": prompt,
                        "messages": [{"role": "user", "content": message}],
                        "max_tokens": 4096,
                        "temperature": 0.2,
                        # gondolkodás kikapcsolva = 1-2 s valaszido (MiniMax-M3 tamogatja)
                        "thinking": {"type": "disabled"},
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = "".join(
                    block.get("text", "")
                    for block in data.get("content", [])
                    if block.get("type") == "text"
                )
            else:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": message},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 4096,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
            # Gondolkodó modellek <think>...</think> blokkjat el kell dobni
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            return content.strip().strip('"').strip()
        except Exception as e:
            print(f"[MiniMax] Javítás sikertelen, nyers átirat használva: {e}")
            return ""
