# Import libraries
import sys
sys.path.append(".")
from src.db import VectorDB
from llm.model import LocalLLM
import nltk
from nltk.corpus import wordnet, stopwords
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
from deep_translator import GoogleTranslator

# Memory class to store and manage the chat history
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

# Main chatbot class
class ChatBot:
    def __init__(self):
        # Ensure WordNet and stopwords are available
        try:
            wordnet.synsets('test')
        except LookupError:
            nltk.download('wordnet')
        try:
            stopwords("english")
        except:
            nltk.download("stopwords")
            
        self.llm = LocalLLM() # Attribute pointing to the LLM to send messages.
        self.memory = Memory() # Memory object that holds a history of the conversation that has been taken.

        # Translator (auto to English)
        try:
            self.translator = GoogleTranslator(source='auto', target='en')
        except Exception as e:
            print("[WARN] Translator not loaded:", e)
            self.translator = None
        
        # Sentence embedding model
        self.ST_MODEL = SentenceTransformer("all-mpnet-base-v2")
        
        # Load English stopwords
        try:
            _stopwords = set(stopwords.words("english"))
        except:
            nltk.download("stopwords")
            _stopwords = set(stopwords.words("english"))
        self.STOPWORDS_EN = _stopwords
        
        # CrossEncoder model for re-ranking
        self.re_ranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        
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

    def infer(self, message: str):
        """
        Sends the user message to the LLM along with system prompt, context and memory.

        Parameters:
            message (str): User input.

        Returns:
            Tuple[str, list]: The model's response and the current sources used.
        """
        # Detect language of the message
        try:
            detected_lang = self.detect_language(message)
            # Add language instruction to the message
            message_with_lang = f"[LANGUAGE: {detected_lang.upper()}] {message}"
        except:
            message_with_lang = message

        # If ChatBot has no attribute "context" (context hasn't been provided) it prints an error and returns an empty string.
        if not hasattr(self, "context"):
            prompt = [
                {"role": "system", "content": self.system_prompt},
                *self.memory.history,
                {"role": "user", "content": message_with_lang}
            ]
            sources = []
        else:
            prompt = self.build_prompt(context=self.context, user_query=message_with_lang)
            sources = self.current_sources
        
        answer = self.llm(prompt)
        flag = "</think>"
        if flag in answer:
            start_idx = answer.find(flag)
            if start_idx != -1:
                answer = answer[start_idx + len(flag):]
        self.memory.update_memory(human_msg=message, ai_msg=answer)
        
        return answer, sources
    
    def retrieve_context_from_db(self, query, vector_db: VectorDB, k=3):
        """
        Retrieves relevant context chunks from the vector DB.

        Parameters:
            query (str): User query.
            vector_db (VectorDB): The vector database object.
            k (int): Number of top-k chunks to retrieve.

        Returns:
            Tuple[str, list]: Context string and list of sources.
        """
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
        
    def expand_query(self, query_en: str, max_expansions: int = 5, min_sim: float = 0.7) -> list[str]:
        """
        Expands a query using synonyms from WordNet and filters them based on cosine similarity.

        Parameters:
            query_en (str): Query in English.
            max_expansions (int): Max number of expansions to return.
            min_sim (float): Minimum cosine similarity threshold.

        Returns:
            list[str]: Top similar expansions of the query.
        """
        STOP = self.STOPWORDS_EN
        tokens = [t.strip(".,¡¿?;:()[]").lower() for t in query_en.split()]
        targets = []

        for w in tokens:
            if len(w) >= 3 and w.isalpha() and w not in STOP:
                if wordnet.synsets(w, pos=wordnet.NOUN, lang="eng"):
                    targets.append((w, wordnet.NOUN))
                if wordnet.synsets(w, pos=wordnet.VERB, lang="eng"):
                    targets.append((w, wordnet.VERB))

        variants = {query_en}

        for word, pos in targets:
            synsets = wordnet.synsets(word, pos=pos, lang="eng")[:3]
            for syn in synsets:
                for lemma in syn.lemma_names("eng"):
                    lemma = lemma.replace("_", " ")
                    if lemma.lower() == word:
                        continue
                    if pos == wordnet.NOUN and lemma.endswith("s") and lemma[:-1] == word:
                        continue
                    if not lemma.isalpha() or lemma in STOP:
                        continue
                    variants.add(query_en.replace(word, lemma))

        orig_emb = self.ST_MODEL.encode(query_en, convert_to_tensor=True)
        var_list = list(variants)
        var_embs = self.ST_MODEL.encode(var_list, convert_to_tensor=True)
        sims = util.cos_sim(orig_emb, var_embs)[0].tolist()
        scored = sorted(zip(var_list, sims), key=lambda x: x[1], reverse=True)

        final = []
        for text, sim in scored:
            if sim < min_sim:
                break
            final.append(text)
            if len(final) >= max_expansions:
                break

        print("Expanded queries:", final)
        return final
    
    def translate_to_english(self, text: str) -> str:
        """Translates any input text to English using GoogleTranslator."""
        if self.translator:
            try:
                return self.translator.translate(text)
            except:
                pass
        return text
    
    def retrieve_context_from_db_with_reranking(self, query: str, vector_db: VectorDB, k_initial: int = 5, k_final: int = 3):
        """
        Expands and reranks context chunks for a query using semantic similarity.

        Parameters:
            query (str): Original user query.
            vector_db (VectorDB): Vector store object.
            k_initial (int): Chunks to retrieve per expansion.
            k_final (int): Final chunks to keep after reranking.

        Returns:
            Tuple[str, list]: Final context and selected top documents.
        """
        query_en = self.translate_to_english(query)
        expanded_queries = self.expand_query(query_en)

        all_chunks = []
        for q in expanded_queries:
            _, chunks = vector_db.retrieve_context(q, k=k_initial)
            all_chunks.extend(chunks)

        if not all_chunks:
            print("No context found in the Database.")
            self.current_sources = []
            return "", []

        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            txt = chunk["content"]
            if txt not in seen:
                seen.add(txt)
                unique_chunks.append(chunk)

        candidates = unique_chunks[:50]
        pairs = [(query_en, chunk["content"]) for chunk in candidates]
        scores = self.re_ranker.predict(pairs)

        ranked = [chunk for _, chunk in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
        topk = ranked[:k_final]

        joined_chunks = []
        for doc in topk:
            joined_chunks.append(vector_db._search_nearby_chunks(doc, 4))
        final_context = "\n\n---\n\n".join(joined_chunks)

        self.initialize_context(final_context)
        self.current_sources = topk

        print("Successfully loaded context after re-ranking.")
        return final_context, topk  

    def __call__(self, message: str):
        self.infer(message=message)

if __name__=="__main__":
    cb = ChatBot()
    cb.infer("Hola")