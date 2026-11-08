# Token-score semantics and runtime identity

Source inspection supports treating the tested scores as pre-sampling model probabilities, not probabilities normalized over the offered labels. This is a source-based interpretation tied to reported runtime identities, not a reproducible-build proof or a calibration result.

The capability log identifies the launched `llama-server.exe` and build commit `a37cc4610`. A read-only version command on the installed executable reports that same commit, build 1, and Clang 18.1.8 for Windows AMD64. The current executable and inference DLL hashes are retained in the private receipt. No model was loaded during this inspection.

In [Ollama 0.40.9](https://github.com/ollama/ollama/blob/v0.40.9/llm/llama_server.go), the completion request maps the requested alternative count to `n_probs` and does not set `post_sampling_probs`. The response converter forwards the log-probability values without normalizing over offered answers. Its alternate post-sampling probability branch is not the path inferred for this request.

At the [reported runner commit](https://github.com/ggml-org/llama.cpp/blob/a37cc4610/tools/server/server-task.h), `post_sampling_probs` defaults to false. The [server context](https://github.com/ggml-org/llama.cpp/blob/a37cc4610/tools/server/server-context.cpp) disables backend sampling when pre-sampling logits are needed and obtains token probabilities from logits rather than the sampler's filtered candidate distribution.

The [probability helper](https://github.com/ggml-org/llama.cpp/blob/a37cc4610/tools/server/server-common.cpp) normalizes across its full available logits vector, then the caller returns the requested leading alternatives. The [serializer](https://github.com/ggml-org/llama.cpp/blob/a37cc4610/tools/server/server-task.cpp) takes the logarithm, representing an underflowed zero with a finite negative sentinel. Single-precision arithmetic can round the dominant probability to one while retaining very small alternatives, consistent with the recorded response. This is not evidence that a probability of one means certainty about a real-world answer.

The decoder documentation now says it preserves reported mass and requires callers to verify provider semantics. It makes no universal vocabulary-normalization claim about arbitrary providers. The capability archive retains the exact previous decoder, so this clarification does not alter its frozen result.

The original capability freeze hashed Ollama and model assets but omitted the separate runner executable and DLLs. The saved log binds the reported runner commit and launch path; current hashes cannot retroactively prove those binaries were unchanged during that run. Future experiments must bind the executable and loaded runtime libraries before and after execution, in addition to the model, request, decoder, and settings. The present receipt records this gap explicitly.

Token-score support remains development-only. No selection threshold, support policy, production default, or semantic acceptance claim follows from this inspection. A prospective selector must preserve complete source and temporal conditions, distinguish unavailable responses from abstention, and pass the original quality and full latency gates on separate evaluation data.
