# Architecture

```text
User / Analyst
      |
      v
+--------------------+
|    Streamlit UI    |
+---------+----------+
          |
          +------------------------------+
          |                              |
          v                              v
+------------------------+      +----------------------+
| Deterministic ML Layer |      |    Agentic Layer     |
|------------------------|      |----------------------|
| Validation             |      | LangChain create_agent|
| Cleaning               |      | Structured tools     |
| Preprocessing          |      | OpenAI chat model    |
| Imbalance handling     |      +----------+-----------+
| Training               |                 |
| Evaluation             |     compact verified outputs
| Risk scoring           |                 |
| Explainability         |<----------------+
+------------------------+
```

The LLM is not responsible for training, scoring, or calculating metrics.
