# Wikipedia Agentic RAG

Simple example of a retrieval agent using TensorZero and Google Gemini API.

## Requirements

- Google Gemini API
- Python 3

## Setup

1. Get your API in [Google AI Studio](https://aistudio.google.com/apikey)
2. Duplicate `.env.example` file and rename the copy to `.env`
3. Create a [Python virtual environment](https://docs.python.org/3/library/venv.html) and activate it
4. Add the API key in `GOOGLE_AI_STUDIO_API_KEY` on `.env` file
5. Install the dependencies `pip install -r requirements.txt`

## Running

```bash
python main.py
```

## Known issues

- This project is created from [simple agentic-rag](https://github.com/tensorzero/tensorzero/tree/main/examples/rag-retrieval-augmented-generation/simple-agentic-rag) tutorial, so it's not realiable
- Sometimes the agent is not capable to answer the questions (changing the model or ai provider can be better)

## Links

- [TensorZero Documentation](https://www.tensorzero.com/docs/quickstart/)
- [TensorZero Examples](https://github.com/tensorzero/tensorzero/tree/main/examples)

## License

This project is under MIT License, enjoy it!
