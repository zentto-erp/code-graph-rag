"""Calls that name their target in a string literal.

Some call sites do not name their callee syntactically: they pass its name as a
string to a dispatcher.

    const rows = await callSp('usp_invoice_list', params)   # TypeScript -> PL/pgSQL
    cursor.callproc('usp_invoice_list')                     # Python DB-API
    conn.prepareCall("{call usp_invoice_list(?)}")          # Java JDBC

No parser resolves those as calls, so the graph stops at the dispatcher and the
routine looks unreachable: dead-code reports it, and "what breaks if I change
this procedure" has no answer. That gap is widest exactly where it hurts most,
in codebases that keep business logic in the database.

The dispatcher is project-specific, so it is declared rather than guessed, in a
`.cgr.toml` at the repository root:

    [[string_calls]]
    callee = "callSp"      # dispatcher; matched on the last name segment
    arg_index = 0          # which argument carries the name

Nothing is inferred: without a declaration this module does nothing at all.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from tree_sitter import Node

CONFIG_FILENAME = ".cgr.toml"
CONFIG_SECTION = "string_calls"
# Points at a config file outside the repository, for indexing a checkout that
# cannot carry one (a read-only mount, a third-party repo).
CONFIG_PATH_ENV = "CGR_CONFIG_PATH"
_QUOTES = "\"'`"


@dataclass(frozen=True)
class StringCallSpec:
    """A dispatcher whose argument names the real callee."""

    callee: str
    arg_index: int = 0


def load_string_call_specs(repo_root: Path) -> tuple[StringCallSpec, ...]:
    """Read `[[string_calls]]` from the repository's `.cgr.toml`.

    A missing or malformed file disables the feature instead of failing the
    ingestion: this is an optional refinement, never a prerequisite.
    """
    override = os.environ.get(CONFIG_PATH_ENV)
    config_path = Path(override) if override else repo_root / CONFIG_FILENAME
    if not config_path.is_file():
        return ()
    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning(f"Ignoring unreadable {CONFIG_FILENAME}: {exc}")
        return ()

    specs: list[StringCallSpec] = []
    for entry in data.get(CONFIG_SECTION, ()):
        if not isinstance(entry, dict):
            continue
        callee = entry.get("callee")
        if not isinstance(callee, str) or not callee:
            continue
        arg_index = entry.get("arg_index", 0)
        if not isinstance(arg_index, int) or arg_index < 0:
            arg_index = 0
        specs.append(StringCallSpec(callee=callee, arg_index=arg_index))

    if specs:
        names = ", ".join(spec.callee for spec in specs)
        logger.info(f"String-call dispatchers declared in {CONFIG_FILENAME}: {names}")
    return tuple(specs)


def _string_literal_value(node: Node) -> str | None:
    """The text of a string literal, or None when the argument is not one.

    A non-literal argument (a variable, a template with substitutions) names a
    target only at runtime; guessing there would invent edges.
    """
    text_bytes = node.text
    if not text_bytes:
        return None
    text = text_bytes.decode("utf-8", "replace").strip()
    if len(text) < 2 or text[0] not in _QUOTES or text[-1] != text[0]:
        return None
    value = text[1:-1].strip()
    if not value or "${" in value or "%s" in value:
        return None
    return value


def string_call_target(
    call_node: Node, call_name: str, specs: tuple[StringCallSpec, ...]
) -> str | None:
    """The routine named by a declared dispatcher's argument, if any.

    `call_name` is matched on its LAST segment so a dispatcher reached through
    a namespace or an object (`db.callSp`, `this.callSp`) still matches.
    Schema-qualified targets (`app.usp_x`) reduce to the routine name, which is
    how the definition side registers it.
    """
    if not specs or not call_name:
        return None
    # The call name can arrive decorated by the surrounding expression
    # (`await callSp`, `new Dispatcher`); the callee is its last token, and the
    # last dotted segment of that.
    tokens = call_name.split()
    if not tokens:
        return None
    last_segment = tokens[-1].rsplit(".", 1)[-1]
    for spec in specs:
        if last_segment != spec.callee:
            continue
        # `type_arguments` also contains "argument", and in a generic call it
        # comes FIRST: `callSpOut<UserRow>("usp_users_list")` puts the type list
        # before the value list. Taking the first match picked the types, looked
        # for a string literal among them, found `UserRow`, and returned None — so
        # every TYPED dispatcher call produced no edge at all.
        #
        # Typing the row is the ordinary way to call such a helper in TypeScript,
        # so the loss is not marginal: on one ERP it was 842 of roughly 1.100
        # route -> stored-procedure edges. The procedures they reach then look
        # unreachable — dead-code analysis accuses them, and "what breaks if I
        # change this procedure" answers nothing, which is the exact question
        # declaring the dispatcher was meant to answer.
        #
        # Skipping `type_*` keeps this language-agnostic: the value list is
        # `arguments` in TS/Rust and `argument_list` in Python, and neither starts
        # with `type_`.
        arguments = next(
            (
                child
                for child in call_node.named_children
                if "argument" in child.type and not child.type.startswith("type_")
            ),
            None,
        )
        if arguments is None:
            return None
        args = [
            child
            for child in arguments.named_children
            if child.type not in {"comment", "line_comment", "block_comment"}
        ]
        if spec.arg_index >= len(args):
            return None
        value = _string_literal_value(args[spec.arg_index])
        if not value:
            return None
        return value.rsplit(".", 1)[-1]
    return None
