# Router (`Router`)

`Router` picks which model to use based on query complexity.

This is for **cost governance**: simple questions route to cheap models, complex ones to expensive models.

## Usage

```python
from contextbuddy import Router, Pipeline

router = Router([
    {"max_complexity": 0.3, "model": "gpt-4o-mini"},
    {"max_complexity": 1.0, "model": "gpt-4o"},
])

pipeline = Pipeline.from_directory("./docs/", router=router)

answer = pipeline.query(
    "Summarize the contract",
    llm_calls={
        "gpt-4o-mini": cheap_llm,
        "gpt-4o": expensive_llm,
    },
)
```

## Notes

- Complexity scoring is offline and deterministic.
- Router is optional; compression works without it.

