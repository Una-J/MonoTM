"""Serve a Hugging Face chat model on CPU through a loopback HTTP endpoint.

Requests are processed sequentially. Requires MonoTM's embeddings extra.
Model files are stored in the Hugging Face cache.
"""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import time

LABEL_FORMAT = "Return only one short line: FINAL: <4-10 words naming one concept>. No explanations, paragraphs, or lists."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum generated tokens per request")
    args = parser.parse_args()
    if args.threads < 1 or args.max_new_tokens < 1:
        parser.error("threads and max-new-tokens must be positive")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                attn_implementation="eager").to("cpu").eval()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send_json(200 if self.path == "/health" else 404,
                           {"model": args.model, "status": "ready", "max_new_tokens": args.max_new_tokens})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": "Unknown path"})
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 2_000_000:
                    raise ValueError("Expected a JSON body under 2 MB")
                payload = json.loads(self.rfile.read(size))
                if payload.get("model") != args.model:
                    raise ValueError(f"The configured model is {args.model}")
                limit = min(int(payload.get("max_tokens", 128)), args.max_new_tokens)
                if limit < 1:
                    raise ValueError("max_tokens must be positive")
                guidance = None if "PREDICTION:" in payload["messages"][-1]["content"] else LABEL_FORMAT
                messages = ([{"role": "system", "content": guidance}] if guidance else []) + payload["messages"]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt")
                if inputs.input_ids.shape[1] + limit > model.config.max_position_embeddings:
                    raise ValueError("Prompt exceeds this model's context length")
            except (ValueError, TypeError, KeyError) as exc:
                self.send_json(400, {"error": str(exc)})
                return
            start = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
            new_tokens = output[0, inputs.input_ids.shape[1]:]
            content = tokenizer.decode(new_tokens, skip_special_tokens=True)
            finish = "length" if len(new_tokens) >= limit else "stop"
            logging.info("Generated %s tokens in %.2fs (%s)", len(new_tokens), time.monotonic() - start, finish)
            self.send_json(200, {"model": args.model, "choices": [{"message": {"role": "assistant", "content": content},
                                                               "finish_reason": finish}],
                                 "usage": {"prompt_tokens": int(inputs.input_ids.shape[1]), "completion_tokens": len(new_tokens)},
                                 "smoke_server": {"system_prompt": guidance, "output_token_cap": args.max_new_tokens}})

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    logging.info("Ready at http://127.0.0.1:%s/v1 (CPU; max output %s tokens)", args.port, args.max_new_tokens)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
