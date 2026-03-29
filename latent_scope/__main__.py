"""Entry point: python -m latent_scope [--port PORT] [--host HOST] [--model MODEL_ID]"""

import argparse
import threading
import webbrowser

from latent_scope.runtime import SharedModelRuntime
from latent_scope.app import create_app

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def main():
    parser = argparse.ArgumentParser(description="Latent Scope — activation space explorer")
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to load on startup")
    parser.add_argument("--no-model", action="store_true", help="Start without loading a model")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    runtime = SharedModelRuntime()

    model_id = None if args.no_model else args.model
    if model_id:
        print(f"  Loading {model_id}…")
        try:
            runtime.load_model(model_id)
            print(f"  Model ready.\n")
        except Exception as e:
            print(f"  Warning: could not load model: {e}\n")

    app = create_app(runtime)

    url = f"http://{args.host}:{args.port}"
    print(f"  Latent Scope  →  {url}\n")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
