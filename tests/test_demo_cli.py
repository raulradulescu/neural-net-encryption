from __future__ import annotations

from src.demo_cli import parse_args, prompt_missing_inputs


def test_demo_cli_allows_prompted_plaintext() -> None:
    args = parse_args(["--key", "secret"])

    assert args.text is None
    assert args.key == "secret"


def test_demo_cli_prompts_for_missing_text_and_key(monkeypatch) -> None:
    answers = iter(["hello from keyboard", "secret"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    args = parse_args([])
    prompt_missing_inputs(args)

    assert args.text == "hello from keyboard"
    assert args.key == "secret"
