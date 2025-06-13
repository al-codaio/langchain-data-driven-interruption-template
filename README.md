# LangGraph Data-Driven Interruption Template

## Features

* **LangGraph Core:** Demonstrates a common pattern for dynamic state validation and user interruption based on missing required fields.
* **Orchestrator Node:** Uses an LLM to dynamically route the graph flow based on the conversation state and a "plan".
* **FastAPI & LangServe:** Exposes your LangGraph application via a REST API for local testing and potential deployment.
* **Pydantic for State:** Uses Pydantic `BaseModel` for robust state management.
* **Centralized Requirements:** Defines node requirements in a single place (`models.py`) to avoid boilerplate.
* **LLM-Powered User Prompts:** Dynamically generates user-friendly messages for missing information.
