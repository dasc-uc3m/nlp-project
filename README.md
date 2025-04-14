# ChatBot Project
Duarte Moura, Alejandro Merino, Sandra Eizaguerri and Carlos Garijo.

## Installation
1. Open an empty folder and run
```sh
git@github.com:dasc-uc3m/nlp-project.git
```

2. Create your virtual environment. I use virtualenv for this (in Linux) e.g.:
```sh
virtualenv -p python3 .venv
source .venv/bin/activate
```
but you can use `conda`, `uv` or whatever you want!

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

## LLM service
In `llm/` folder there is the definition of the LLM and a REST API that run in the docker to allow LLM inference.
These files must not be modified! Think of them as if they were an external LLM to which we will make requests, but
with everything coded outside of it. All the logic and coding (RAG, prompt formatting, chatbot structure and methods...)
are allocated in the `src/` folder. The LLM loaded in the docker just receives a requests and returns an answer!

To start the LLM run in a terminal:

```sh
docker compose up --build
```

Once it starts, it is possible to make inference to the LLM from wherever you want. To make inference, the port defined in `docker-compose.yml` is exposed and sending data is possible by making POST requests to the `/generate` method, but you don't have to worry about this as an interface to this API is coded in the `LocalLLM` class in `src/chatbot.py`.


