"""AST/token ambiguity detection and intent-comment linting."""

from __future__ import annotations
import ast, io, tokenize


def normalized_ast(source: str) -> str:
    return ast.dump(ast.parse(source), annotate_fields=False, include_attributes=False)


def token_similarity(a: str, b: str) -> float:
    def tokens(text):
        return [tok.string for tok in tokenize.generate_tokens(io.StringIO(text).readline)
                if tok.type not in {tokenize.COMMENT,tokenize.NL,tokenize.NEWLINE,
                                    tokenize.INDENT,tokenize.DEDENT,tokenize.ENDMARKER}]
    x,y=tokens(a),tokens(b)
    if not x and not y:return 1.0
    matches=sum(left==right for left,right in zip(x,y))
    return matches/max(len(x),len(y))


def ambiguous(a: str, b: str, threshold=.9) -> bool:
    return normalized_ast(a)!=normalized_ast(b) and token_similarity(a,b)>=threshold


def intent_comments(source: str) -> tuple[str,...]:
    return tuple(tok.string[1:].strip()[len("intent:"):].strip() for tok in
                 tokenize.generate_tokens(io.StringIO(source).readline)
                 if tok.type==tokenize.COMMENT and tok.string[1:].strip().lower().startswith("intent:"))


def lint_intent_comments(source: str, required: tuple[str,...]=()) -> tuple[str,...]:
    comments=" ".join(intent_comments(source)).casefold()
    return tuple(f"missing-intent:{item}" for item in required if item.casefold() not in comments)


def inherit_comments(parent: str, child: str) -> str:
    notes=[line for line in parent.splitlines() if line.lstrip().startswith("# intent:")]
    existing=set(child.splitlines())
    return "\n".join(notes+[child] if not all(n in existing for n in notes)
                     else [child])
