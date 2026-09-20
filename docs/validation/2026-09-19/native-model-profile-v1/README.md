# Fixed native-host sampling profile

Ollama 0.34.1 created `mnemos-native-qwen35-9b:greedy-v1` from the existing
Qwen3.5:9b model. The original model digest and complete show response are
unchanged. The derived model shares its weight blob, template, model metadata,
capabilities, tensors, and license. Only temperature 0, presence penalty 0, and
seed 42 were specified; inherited top-k 20 and top-p 0.95 remain. The base
reported temperature 1 and presence penalty 1.5. No inference occurred during
profile creation, and no live Hermes configuration was changed.

This uses Ollama's documented [create-model API](https://docs.ollama.com/api/create)
and [model parameters](https://docs.ollama.com/modelfile), checked September 19,
2026. The API's before/after records and executed helper are retained with hashes.
The effect on identifier copying and the native lifecycle is unmeasured until
the separate prospective v3 cohort runs. Neither a fixed seed nor temperature
zero guarantees a correct response or identical outputs for different prompts.
