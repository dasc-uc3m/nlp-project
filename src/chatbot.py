import sys
sys.path.append(".")

import requests
import html
from typing import Dict
from src.db import VectorDB

from llm.model import LocalLLM

import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

ST_MODEL = SentenceTransformer("all-mpnet-base-v2") 

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
            "If the user's language is unclear, ask for clarification.\n"
            "If the user's query isn't related with healthcare, you MUST indicate that you are an assistant chatbot for healthcare, "
            "specialized in maternity and feminine health areas and you are not the indicated assistant to solve that question."
        )

        # Context prompt. This is the template of the message that will be sent to the LLM where we provide it with context.
        self.context_prompt = (
            "Below is some context extracted from documents, specified between [START_OF_CONTEXT] and [END_OF_CONTEXT] tags.\n"
            "Use this context to answer all future questions, referring to and citing it whenever necessary. "
            "Please read carefully and remember this context for our conversation. "
            "IMPORTANT LANGUAGE RULE: You MUST ALWAYS respond in the EXACT SAME LANGUAGE as the user's question. "
            "This is a strict requirement that must be followed for every response.\n"
            "Whenever is possible, cite and refer to the content provided in the following context in your answers.\n"
            "If the context isn't directly related with the query the user has provided or the information provided "
            "is not enough for answering the question, please, indicate it saying explicitly that you can't answer "
            "to the request with the provided context.\n"
            "If the user's query isn't related with healthcare, you MUST indicate that you are an assistant chatbot for healthcare, "
            "specialized in maternity and feminine health areas and you are not the indicated assistant to solve that question.\n"
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

    def expand_query(self,query: str,max_expansions: int = 3, min_sim: float = 0.85) -> list[str]:
        """
        1) Desambiguamos con Lesk + WordNet para generar variantes.
        2) Calculamos emb de todas en batch.
        3) Ordenamos por similitud con la query original.
        4) Devolvemos solo las top‐K que superen min_sim.
        """

        # 1) variantes con Lesk
        variants = {query}
        for w in query.split():
            syn = lesk(query.split(), w)
            if not syn:
                continue
            for lemma in syn.lemma_names():
                alt = lemma.replace("_", " ")
                if alt.lower() == w.lower():
                    continue
                variants.add(query.replace(w, alt))

        variants = list(variants)

        # 2) emb y similitudes en batch
        orig_emb = ST_MODEL.encode(query, convert_to_tensor=True)
        cand_embs = ST_MODEL.encode(variants, convert_to_tensor=True)
        sims = util.cos_sim(orig_emb, cand_embs)[0]  # tensor de forma (len(variants),)

        # 3) emparejamos y ordenamos
        scored = list(zip(variants, sims.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)

        # 4) filtrado por umbral + top‐K
        final = []
        for text, score in scored:
            if score < min_sim:
                break
            final.append(text)
            if len(final) >= max_expansions + 1:  # +1 porque contamos la original
                break

        return final

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
        
    def retrieve_context_from_db_with_reranking(self, query: str, vector_db: VectorDB, k_initial: int = 5, k_final: int = 3):
            """
            1) Expandimos la query original (WordNet + Lesk) → lista de sub-queries.
            2) Por cada sub-query llamamos a vector_db.retrieve_context(q, k_initial),
            que nos devuelve (context_string, sources_list).
            3) Recolectamos TODOS los `sources_list` (que son listas de dicts {'source', 'content'}).
            4) Desduplicamos por el campo 'content'.
            5) Re-rankeamos con CrossEncoder usando SIEMPRE la query original.
            6) Cogemos los k_final primeros y construimos
            — el `context` final concatenando sus `.content`,
            — la lista de `sources`.
            7) Igual que tu función “normal”, guardamos en self.context y self.current_sources.
            """

            # 1) Generar todas las queries
            expanded_queries = self.expand_query(query)

            # 2) Recuperar los chunks ‘raw’ de cada sub-query
            all_chunks = []  # aquí vamos a meter dicts {'source':…, 'content':…}
            for q in expanded_queries:
                context_str, chunks = vector_db.retrieve_context(q, k=k_initial)
                # chunks es LISTA de dicts {'source','content'}
                all_chunks.extend(chunks)

            if not all_chunks:
                print("No context found in the Database.")
                return "", []

            # 3) Desduplicar por texto puro
            seen = set()
            unique_chunks = []
            for chunk in all_chunks:
                txt = chunk["content"]
                if txt not in seen:
                    seen.add(txt)
                    unique_chunks.append(chunk)
            # Limitar candidatos a un máximo razonable
            candidates = unique_chunks[:50]

            # 4) Preparar los pares para el CrossEncoder (query original, texto chunk)
            pairs = [(query, chunk["content"]) for chunk in candidates]

            # 5) Re-rank
            scores = self.re_ranker.predict(pairs)

            # 6) Ordenar por score y quedarnos con los best k_final
            ranked = [
                chunk
                for _, chunk in sorted(zip(scores, candidates),
                                    key=lambda x: x[0],
                                    reverse=True)
            ]
            topk = ranked[:k_final]

            # 7) Construir context final y lista de fuentes
            context_pieces = [c["content"] for c in topk]
            sources        = [c["source"]  for c in topk]

            final_context = "\n\n".join(context_pieces)

            # 8) Guardamos igual que en la versión sin reranking
            self.initialize_context(final_context)
            self.current_sources = sources

            print("Successfully loaded context after re-ranking.")
            return final_context, sources

    def __call__(self, message: str):
        self.infer(message=message)

if __name__=="__main__":
    cb = ChatBot()
    cb.infer("Hola")