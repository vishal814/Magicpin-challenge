# magicpin AI Challenge — Team Alpha (Vera)

## Approach
This project is built using **Python and FastAPI** to provide a highly concurrent, reliable HTTP web server that satisfies the 5 required judging endpoints. 

For the core intelligence, we use **OpenAI's GPT-4o** via the official Python SDK. The system utilizes structured JSON output mode to guarantee that the bot returns perfectly formatted JSON schema objects to the judge during the `/v1/tick` and `/v1/reply` phases.

### Prompting Strategy
- **Context Injection**: For proactive messages (`/v1/tick`), the LLM is provided the raw JSON context encompassing the Category, Merchant, Trigger, and Customer profiles. 
- **System Constraints**: The system prompt forces strict adherence to the judging rubric: zero hallucinations, mandatory specificity (using exact numbers/dates from the payload), and dynamic tone matching (e.g., using a clinical tone for Dentists).
- **Reactive Routing**: During `/v1/reply`, the LLM evaluates the merchant's reply against the conversation history and selects the optimal action. Explicit rules guide it to instantly drop into `end` mode if an auto-reply is detected, or immediately transition to `action` mode if the merchant agrees to an offer.

## Tradeoffs Made
- **Stateless Prompts vs Memory Cost**: Instead of using LangChain or complex memory databases, the FastAPI server relies on simple in-memory python dictionaries for fast context retrieval. If the server scales, we would transition this to a Redis cluster.
- **Synchronous Generation**: The LLM calls are executed synchronously. While GPT-4o is fast enough to hit the 30-second window, heavy loads could risk timeouts without moving to async/background task queues.

## Additional Context Needed
It would be highly beneficial to receive a list of "successful past conversions" in the Category payload, allowing the LLM to model its phrasing off of verified real-world wins.
