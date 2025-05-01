import sys
sys.path.append(".")

import requests
import html
from typing import Dict
from src.db import VectorDB

from llm.model import LocalLLM

import nltk
from nltk.corpus import wordnet
from sentence_transformers import CrossEncoder

class Memory:
    def __init__(self, max_messages_count=10):
        self._memory = []
        self.max_messages_count = max_messages_count
    def update_memory(self, human_msg: str, ai_msg: str):
        self._memory.append(
            {"role": "user", "content": human_msg}
        )
        self._memory.append(
            {"role": "assistant", "content": ai_msg}
        )

    def reset_memory(self):
        self._memory = []

    @property
    def history(self):
        return self._memory[-self.max_messages_count:]


class ChatBot:
    def __init__(self):
        
        try: # To ensure WordNet is available
            wordnet.synsets('test')
        except LookupError:
            nltk.download('wordnet')
            
        self.llm = LocalLLM() # Attribute pointing to the LLM to send messages.
        self.memory = Memory() # Memory object that holds a history of the conversation that has been taken.

        self.re_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Download language detection model
        try:
            from langdetect import detect
            self.detect_language = detect
        except ImportError:
            print("Installing langdetect...")
            import subprocess
            subprocess.check_call(["pip", "install", "langdetect"])
            from langdetect import detect
            self.detect_language = detect

        # System prompt. This is like some 'general rules' that will be passed to the LLM to behave in a certain way.
        self.system_prompt = (
            "You are an AI assistant ChatBot engaged in a conversation with a user. "
            "You will be provided with context extracted from documents the user provides. "
            "Always provide useful answers, taking into account the given context. "
            "Whenever possible, refer to the provided context and cite the text from the documents to enrich your answers. "
            "IMPORTANT LANGUAGE RULE: You MUST ALWAYS respond in the EXACT SAME LANGUAGE as the user's question. "
            "This is a strict requirement that must be followed for every response. "
            "If the user's language is unclear, ask for clarification."
        )

        # Context prompt. This is the template of the message that will be sent to the LLM where we provide it with context.
        self.context_prompt = (
            "Below is some context extracted from documents, specified between [START_OF_CONTEXT] and [END_OF_CONTEXT] tags.\n"
            "Use this context to answer all future questions, referring to and citing it whenever necessary. "
            "Please read carefully and remember this context for our conversation. "
            "IMPORTANT LANGUAGE RULE: You MUST ALWAYS respond in the EXACT SAME LANGUAGE as the user's question. "
            "This is a strict requirement that must be followed for every response.\n"
            "The context:\n"
            "[START_OF_CONTEXT]\n"
            "{context}\n"
            "[END_OF_CONTEXT]"
        )
    
    def initialize_context(self, context: str):
        """
        Method that loads the context in the ChatBot object to use it for the conversation.
        """
        self.context = context

    def remove_context(self):
        """
        Resets the context.
        """
        del self.context

    def has_context(self):
        return hasattr(self, "context")

    def build_prompt(self, context: str, user_query: str):
        """
        This method builds the context following the llm's chat template that huggingface models use.
        For more info see: https://huggingface.co/docs/transformers/chat_templating.
        
        Parameters
        ----------
        context : str
            String that contains the context formed by the chunks of the different documents, holded in `self.context`.
        user_query : str
            New user query in string format.
        
        Returns
        -------
        list[dict]
            List of messages following the chat template defined by huggingface.
        """

        messages = [
            # We add the system prompt message.
            {"role": "system", "content": self.system_prompt},

            # We add the context by completing the context message template. We also add an AI answer that can help the
            # LLM enroll better in the conversation.
            {"role": "user", "content": self.context_prompt.format(context=context)},
            {"role": "assistant", "content": (
                "OK. I have the context. Please, start asking questions and I will answer you based on the "
                "context you have provided me.")
            },

            # We add the history of messages.
            *self.memory.history,

            # Finally, we add the last given user query.
            {"role": "user", "content": user_query}
        ]
        return messages

    def infer(self, message: str, expand: bool = False):
        
        # Detect language of the message
        try:
            detected_lang = self.detect_language(message)
            # Add language instruction to the message
            message_with_lang = f"[LANGUAGE: {detected_lang.upper()}] {message}"
        except:
            message_with_lang = message
            
        if expand:
            expanded_message = self.expand_query(message_with_lang)
        else:
            expanded_message = message_with_lang
            
        print(f"DEBUG: {expanded_message}")
        
        # If ChatBot has no attribute "context" (context hasn't been provided) it prints an error and returns an empty string.
        if not hasattr(self, "context"):
            prompt = [
                {"role": "system", "content": self.system_prompt},
                *self.memory.history,
                {"role": "user", "content": expanded_message}
            ]
            sources = []
        else:
            prompt = self.build_prompt(context=self.context, user_query=expanded_message)
            sources = self.current_sources
        
        answer = self.llm(prompt)
        flag = "</think>"
        if flag in answer:
            start_idx = answer.find(flag)
            if start_idx != -1:
                answer = answer[start_idx + len(flag):]
        self.memory.update_memory(human_msg=message, ai_msg=answer)
        
        return answer, sources

    def expand_query(self, query: str) -> str:
        """
        Expand the user query by replacing words with their most common synonyms using WordNet.
        """
        words = query.split()
        expanded_words = []

        for word in words:
            synonyms = wordnet.synsets(word)
            if synonyms:
                # Cogemos el primer sinónimo (el más habitual)
                lemmas = synonyms[0].lemma_names()
                if lemmas:
                    # Usamos el primer lemma como expansión si es diferente
                    synonym = lemmas[0].replace('_', ' ')
                    expanded_words.append(synonym)
                else:
                    expanded_words.append(word)
            else:
                expanded_words.append(word)

        expanded_query = " ".join(expanded_words)
        return expanded_query

    def retrieve_context_from_db(self, query, vector_db: VectorDB, k=3):
        context, sources = vector_db.retrieve_context(query, k=k)
        if len(context) > 0:
            self.initialize_context(context=context)
            self.current_sources = sources  # Store sources for later use
            message = "Successfully loaded context."
        else:
            message = "No context found in the Database."
            self.current_sources = []
        print(message)
        return message, sources
        
    def retrieve_context_from_db_with_reranking(self, query, vector_db: VectorDB, k=3):
        
        initial_results = vector_db.retrieve_context(query, k=10) # Esto recupera k documentos

        if not initial_results: # Esto por si te dice que nanai de la china no hay docs
            print("No context found in the Database.")
            return "No context found in the Database."

        pairs = [(query, doc) for doc in initial_results] # Hacemos los pares query doc para re-rankear

        scores = self.re_ranker.predict(pairs) # Usamos el cross-encoder model que está cargado en __init__ para sacar los scores

        ranked_docs = [doc for _, doc in sorted(zip(scores, initial_results), key=lambda x: x[0], reverse=True)] # Ordemos los docs por relevancia en base al score

        selected_contexts = ranked_docs[:k] # Cogemos los k mejores docs (3 en este caso)

        context = "\n\n".join(selected_contexts) # Los unimos como un sólo contexto

        self.initialize_context(context=context) # Cargamos el contexto nuevo en el chatbot
        print("Succesfully loaded context after re-ranking.")

        return "Succesfully loaded context after re-ranking."

    def __call__(self, message: str):
        self.infer(message=message)

if __name__=="__main__":
    cb = ChatBot()
    cb.infer("Hola")