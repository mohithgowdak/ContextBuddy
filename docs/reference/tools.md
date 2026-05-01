# Agent tools (`contextbuddy.tools`)

ContextBuddy can generate OpenAI-compatible tool schemas so agents can:

- search documents
- compress context

## Usage

```python
from contextbuddy.tools import make_search_tool, make_compress_tool, handle_tool_call

tools = [
    make_search_tool(store),
    make_compress_tool(engine),
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

for tc in response.choices[0].message.tool_calls:
    result = handle_tool_call(tc, tools)
```

## Notes

- Tools are optional.
- They don’t change the core compressor behavior.

