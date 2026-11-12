"""Private resident encoder for the one inactive complete-path capacity test."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
import math
from pathlib import Path
import secrets
import sys
import time

spec = importlib.util.spec_from_file_location(
    "resident_capacity", Path(__file__).with_name("bidirectional_support_capacity.py")
)
capacity = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capacity
spec.loader.exec_module(capacity)


def prepare(root, model_root):
    tokenizer, binding = capacity.tokenizer_at(model_root)
    cases = json.loads((root / "cases.json").read_bytes())
    frames = []
    for case in cases:
        cohort = case["cohort"]
        encode = lambda limit=8192: capacity.frame_request(
            cohort, tokenizer, binding, max_tokens=limit
        )
        if case["name"].startswith("token-"):
            source = cohort["candidates"][-1]["sources"][-1]
            original = source["excerpt"]
            repeats = 8192 - len(encode().input_ids)
            for _ in range(8):
                source["excerpt"] = original + " x" * repeats
                actual = len(encode(8448).input_ids)
                if actual == 8192:
                    break
                repeats += 8192 - actual
            assert len(encode().input_ids) == 8192
        if case["name"] == "byte-bound":
            source = cohort["candidates"][-1]["sources"][-1]
            source["excerpt"] += " " * (64000 - len(encode().payload_json.encode()))
            assert len(encode().payload_json.encode()) == 64000
        framed = encode()
        frames.append(
            dict(
                name=case["name"],
                frame_sha256=framed.frame_sha256,
                tokens=len(framed.input_ids),
                payload_bytes=len(framed.payload_json.encode()),
                target_ids=list(framed.target_ids),
            )
        )
    capacity.save(root / "cases.json", cases)
    capacity.save(root / "frames.json", frames)


def serve(root, model_root):
    import torch
    import transformers
    from transformers import AutoModel

    config = json.loads(sys.stdin.readline())
    token = config["token"]
    assert len(token) >= 32
    assert torch.__version__ == "2.10.0+cu128" and transformers.__version__ == "4.57.6"
    assert capacity.sha(model_root / "model/model.safetensors") == capacity.WEIGHT_SHA
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4080"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(capacity.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer, binding = capacity.tokenizer_at(model_root)
    started = time.perf_counter()
    model, loading = AutoModel.from_pretrained(
        model_root / "model",
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        reference_compile=False,
        output_loading_info=True,
    )
    assert (
        not loading["missing_keys"] and not loading["mismatched_keys"] and not loading["error_msgs"]
    )
    assert all(k.startswith(("head.", "classifier.")) for k in loading["unexpected_keys"])
    model = model.to("cuda:0").eval()
    head = (
        torch.nn.Linear(model.config.hidden_size, 1).to(device="cuda:0", dtype=torch.float16).eval()
    )
    assert model.config.num_hidden_layers == 22 and model.config.reference_compile is False
    calls = {"forward": 0, "requests": 0}

    def forward(_module, _inputs):
        calls["forward"] += 1

    model.register_forward_pre_hook(forward)
    frames = {c["name"]: c for c in json.loads((root / "frames.json").read_bytes())}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                if self.path != "/infer" or not secrets.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + token
                ):
                    raise ValueError("Unauthorized")
                size = int(self.headers["Content-Length"])
                if not 0 < size <= 300000:
                    raise ValueError("Invalid message size")
                value = json.loads(self.rfile.read(size))
                expected = frames[value["name"]]
                framed = capacity.frame_request(value["cohort"], tokenizer, binding)
                if framed.frame_sha256 != expected["frame_sha256"]:
                    raise ValueError("Unexpected actual retrieval frame")
                values = []
                if framed.target_ids:
                    ids = torch.tensor([framed.input_ids], device="cuda:0", dtype=torch.long)
                    with torch.inference_mode():
                        output = model(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            output_hidden_states=False,
                            output_attentions=False,
                            return_dict=True,
                        )
                        assert output.last_hidden_state.shape == (
                            1,
                            len(framed.input_ids),
                            model.config.hidden_size,
                        )
                        positions = torch.tensor(framed.decision_positions, device="cuda:0")
                        values = (
                            head(output.last_hidden_state[0, positions])
                            .squeeze(-1)
                            .float()
                            .cpu()
                            .tolist()
                        )
                torch.cuda.synchronize()
                if len(values) != len(framed.target_ids) or not all(
                    math.isfinite(v) for v in values
                ):
                    raise ValueError("Invalid readback")
                calls["requests"] += 1
                result = dict(
                    target_ids=list(framed.target_ids),
                    values=values,
                    tokens=len(framed.input_ids),
                    frame_sha256=framed.frame_sha256,
                    calls=dict(calls),
                )
                data = json.dumps(result, allow_nan=False).encode()
                self.send_response(200)
            except BaseException as exc:
                data = json.dumps(dict(error_type=type(exc).__name__)).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    print(
        json.dumps(
            dict(
                port=server.server_port,
                load_ms=(time.perf_counter() - started) * 1000,
                versions=capacity.runtime_versions(),
                layers=model.config.num_hidden_layers,
                parameters=sum(p.numel() for p in model.parameters()),
                tokenizer_binding=binding,
            )
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    mode, root, model_root = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
    if mode == "prepare":
        prepare(root, model_root)
    elif mode == "serve":
        serve(root, model_root)
    else:
        raise ValueError("Unknown operation")
