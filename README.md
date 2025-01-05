Some simple LangChain and/or LangGraph applications created while I learned them. 
Currently what's in `apps/` is mostly a simple replication of the 
[LangGraph tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction), 
but the hardcoded messages in the original tutorial were rewritten into more flexible
workflows that allow arbitrary inputs. For example, in `apps\chatbot_with_human_supervision.py`
the user may use the `"override_ai"` prompt to override the AI response, or use
`"override_tool_call"` to override the queries in the search tool call. Also, different
from the tutorial which uses Claude, I used the OpenAI API. In LangChain it is just a
drop-in replacement between `ChatOpenAI` and `ChatAnthropic`.

To use with your own LLM API key, create a directory `../../api_keys` relative to `apps/`
and put there your API keys as text files, *e.g.* `openai.txt`.
