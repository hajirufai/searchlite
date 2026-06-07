"""
Query parser — transforms query strings into Query AST nodes.

Supports:
  - Simple terms: python data
  - Boolean operators: python AND data, python OR java, NOT javascript
  - Phrases: "machine learning"
  - Field-specific: title:python
  - Wildcards: pyth*
  - Grouping: (python OR java) AND backend
  - Boosting: python^2.0

Grammar (simplified):
  query     → or_expr
  or_expr   → and_expr (OR and_expr)*
  and_expr  → not_expr (AND not_expr)*
  not_expr  → NOT? primary
  primary   → '(' query ')' | phrase | field_term | wildcard | term
  phrase    → '"' terms '"'
  field_term→ WORD ':' (phrase | term)
  wildcard  → WORD '*'
  term      → WORD ('^' NUMBER)?
"""

import re
from searchlite.query import (
    Query, TermQuery, PhraseQuery, BoolQuery,
    WildcardQuery, MatchAllQuery,
)


class QueryParseError(Exception):
    pass


class QueryParser:
    """Recursive descent parser for search queries."""

    # Token patterns
    _TOKEN_RE = re.compile(
        r"""
        (?P<LPAREN>\()
        |(?P<RPAREN>\))
        |(?P<PHRASE>"[^"]*")
        |(?P<AND>\bAND\b)
        |(?P<OR>\bOR\b)
        |(?P<NOT>\bNOT\b)
        |(?P<BOOST>\^[0-9]+(?:\.[0-9]+)?)
        |(?P<WORD>[^\s()"^]+)
        """,
        re.VERBOSE,
    )

    def __init__(self, default_fields: list[str] | None = None):
        self.default_fields = default_fields or []

    def parse(self, query_string: str) -> Query:
        """Parse a query string into a Query AST."""
        query_string = query_string.strip()
        if not query_string or query_string == "*":
            return MatchAllQuery()

        self._tokens = list(self._tokenize(query_string))
        self._pos = 0
        result = self._parse_or_expr()

        return result

    def _tokenize(self, text: str):
        for match in self._TOKEN_RE.finditer(text):
            kind = match.lastgroup
            value = match.group()
            yield (kind, value)

    def _peek(self):
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, expected_kind=None):
        token = self._peek()
        if token is None:
            return None
        if expected_kind and token[0] != expected_kind:
            return None
        self._pos += 1
        return token

    def _parse_or_expr(self) -> Query:
        left = self._parse_and_expr()

        while self._peek() and self._peek()[0] == "OR":
            self._consume("OR")
            right = self._parse_and_expr()
            if isinstance(left, BoolQuery) and left.should and not left.must and not left.must_not:
                left.should.append(right)
            else:
                left = BoolQuery(should=[left, right])

        return left

    def _parse_and_expr(self) -> Query:
        left = self._parse_not_expr()

        while True:
            peeked = self._peek()
            if peeked and peeked[0] == "AND":
                self._consume("AND")
                right = self._parse_not_expr()
            elif peeked and peeked[0] not in ("OR", "RPAREN", None):
                # Implicit AND
                right = self._parse_not_expr()
            else:
                break

            if isinstance(left, BoolQuery) and left.must and not left.should:
                left.must.append(right)
            else:
                left = BoolQuery(must=[left, right])

        return left

    def _parse_not_expr(self) -> Query:
        if self._peek() and self._peek()[0] == "NOT":
            self._consume("NOT")
            expr = self._parse_primary()
            return BoolQuery(must_not=[expr])

        return self._parse_primary()

    def _parse_primary(self) -> Query:
        token = self._peek()
        if token is None:
            raise QueryParseError("Unexpected end of query")

        # Parenthesized group
        if token[0] == "LPAREN":
            self._consume("LPAREN")
            expr = self._parse_or_expr()
            self._consume("RPAREN")
            return expr

        # Quoted phrase
        if token[0] == "PHRASE":
            return self._parse_phrase()

        # Word — could be field:term, wildcard, or plain term
        if token[0] == "WORD":
            return self._parse_word()

        raise QueryParseError(f"Unexpected token: {token}")

    def _parse_phrase(self) -> Query:
        token = self._consume("PHRASE")
        text = token[1][1:-1]  # strip quotes
        terms = text.lower().split()
        if not terms:
            return MatchAllQuery()

        boost = self._try_boost()
        return PhraseQuery(terms=terms, boost=boost)

    def _parse_word(self) -> Query:
        token = self._consume("WORD")
        word = token[1]

        # Check for field:value or field: followed by phrase
        if ":" in word and not word.startswith(":"):
            field_name, value = word.split(":", 1)

            # field:"phrase" — colon at end, phrase follows as next token
            if not value and self._peek() and self._peek()[0] == "PHRASE":
                phrase_token = self._consume("PHRASE")
                text = phrase_token[1][1:-1]
                terms = text.lower().split()
                boost = self._try_boost()
                return PhraseQuery(terms=terms, field=field_name, boost=boost)

            # field:value followed by "phrase"
            if value and self._peek() and self._peek()[0] == "PHRASE":
                phrase_token = self._consume("PHRASE")
                text = phrase_token[1][1:-1]
                terms = text.lower().split()
                boost = self._try_boost()
                return PhraseQuery(terms=terms, field=field_name, boost=boost)

            if not value:
                # Bare field: with no value — treat as regular term
                boost = self._try_boost()
                return TermQuery(term=word.lower(), boost=boost)

            # field:wildcard*
            if value.endswith("*"):
                boost = self._try_boost()
                return WildcardQuery(pattern=value.lower(), field=field_name, boost=boost)

            # field:term
            boost = self._try_boost()
            return TermQuery(term=value.lower(), field=field_name, boost=boost)

        # Wildcard: pyth*
        if word.endswith("*"):
            boost = self._try_boost()
            return WildcardQuery(pattern=word[:-1].lower() + "*", boost=boost)

        # Plain term
        boost = self._try_boost()
        return TermQuery(term=word.lower(), boost=boost)

    def _try_boost(self) -> float:
        if self._peek() and self._peek()[0] == "BOOST":
            token = self._consume("BOOST")
            return float(token[1][1:])  # strip ^
        return 1.0
