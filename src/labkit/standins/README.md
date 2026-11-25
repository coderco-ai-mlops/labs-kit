# Practice versions of tools

Some tools cannot be installed in a browser: PyTorch, TensorFlow, Airflow, dbt, Chroma, LangGraph. For those, the labs site loads a small practice version from this folder that behaves the same as the real tool for everything the lab uses.

A lab lists them in `lab.toml`:

```toml
standins = ["torch"]
```

Rules:
- A practice version is importable under the real name (`torch.py` or a `torch/` package here).
- The same checks and the same solution must pass with the real tool and with the practice version. Prove both.
- Anything the practice version does not support must fail with a plain message that names the feature and says it is in the real tool, never with a confusing error.
- It is never used in a terminal. There the real tool is installed. Authors can force it for testing with `LAB_STANDINS=1`.
