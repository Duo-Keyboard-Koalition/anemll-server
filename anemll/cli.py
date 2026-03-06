"""CLI interface for anemll - like ollama/llama.cpp."""

import argparse
import sys
import time

from . import __version__


def cmd_pull(args):
    from .models import pull_model
    pull_model(args.model)


def cmd_list(args):
    from .models import list_models
    models = list_models()
    if not models:
        print("No models installed. Run: anemll pull <model>")
        return
    print(f"{'NAME':<55} {'CTX':>6} {'BATCH':>6}")
    for m in models:
        print(f"{m['name']:<55} {m['context_length'] or '?':>6} {m['batch_size'] or '?':>6}")


def cmd_rm(args):
    from .models import remove_model
    remove_model(args.model)


def cmd_run(args):
    from .models import resolve_model
    from .engine import ANEEngine

    model_path = resolve_model(args.model)
    print(f"Loading {model_path.name}...")
    engine = ANEEngine(model_path)

    if not args.no_warmup:
        print("Warming up...")
        engine.warmup()
        engine.warmup()

    if args.prompt:
        # Single prompt mode
        messages = [{"role": "user", "content": args.prompt}]
        for token in engine.generate(messages, temperature=args.temperature):
            print(token, end="", flush=True)
        print()
        return

    # Interactive chat
    print(f"Model: {engine.model_name} (ctx={engine.context_length})")
    print("Type /bye to exit.\n")

    conversation = []
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/bye":
            break

        conversation.append({"role": "user", "content": user_input})

        # Trim history if needed
        input_ids = engine.tokenize_chat(conversation)
        while input_ids.size(1) > engine.context_length - 100 and len(conversation) > 1:
            conversation = conversation[2:]  # Remove oldest pair
            input_ids = engine.tokenize_chat(conversation)

        response = ""
        start = time.time()
        token_count = 0
        for token in engine.generate(conversation, temperature=args.temperature):
            print(token, end="", flush=True)
            response += token
            token_count += 1

        elapsed = time.time() - start
        tps = token_count / elapsed if elapsed > 0 else 0
        print(f"\n[{token_count} tokens, {tps:.1f} t/s]\n")

        conversation.append({"role": "assistant", "content": response})


def cmd_serve(args):
    from .models import resolve_model
    from .server import serve_model

    model_path = resolve_model(args.model)
    serve_model(str(model_path), host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(
        prog="anemll",
        description="Run LLMs on Apple Neural Engine",
    )
    parser.add_argument("--version", action="version", version=f"anemll {__version__}")
    sub = parser.add_subparsers(dest="command")

    # pull
    p = sub.add_parser("pull", help="Download a model from HuggingFace")
    p.add_argument("model", help="Model name or HF repo id (e.g. anemll/anemll-Meta-Llama-3.2-1B-ctx2048_0.1.2)")

    # list
    sub.add_parser("list", help="List downloaded models")

    # rm
    p = sub.add_parser("rm", help="Remove a downloaded model")
    p.add_argument("model", help="Model name to remove")

    # run
    p = sub.add_parser("run", help="Interactive chat with a model")
    p.add_argument("model", help="Model name")
    p.add_argument("-p", "--prompt", help="Single prompt (non-interactive)")
    p.add_argument("-t", "--temperature", type=float, default=0.7)
    p.add_argument("--no-warmup", action="store_true", help="Skip warmup")

    # serve
    p = sub.add_parser("serve", help="Start OpenAI-compatible API server")
    p.add_argument("model", help="Model name")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "pull": cmd_pull,
        "list": cmd_list,
        "rm": cmd_rm,
        "run": cmd_run,
        "serve": cmd_serve,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
