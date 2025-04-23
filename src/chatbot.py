import sys
sys.path.append(".")

import requests
import html
from typing import Dict
from src.db import VectorDB

from llm.model import LocalLLM

class Memory:
    def __init__(self):
        self._memory = []
    
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
        return self._memory


class ChatBot:
    def __init__(self):
        self.llm = LocalLLM() # Attribute pointing to the LLM to send messages.
        self.memory = Memory() # Memory object that holds a history of the conversation that has been taken.

        # System prompt. This is like some 'general rules' that will be passed to the LLM to behave in a certain way.
        self.system_prompt = (
            "You are an AI assistant ChatBot that will be enrolled in a conversation with a certain user. "
            "You will be provided with some context extracted from some documents that the user will provide. "
            "You must provide useful answers always taking into account the given context. "
            "Whenever it is possible, refer to the provided context to answer the user requests and, if necessary "
            "cite the text in the documents to enrich your answers. "
            "Answer always in the same language that the user is using."
        )

        # Context prompt. This is the template of the message that will be sent to the LLM where we provide it with context.
        self.context_prompt = (
            "Below I will provide you with some context that will be used to have a conversation between us. "
            "The context is extracted from some documents, and it is specified between [START_OF_CONTEXT] and [END_OF_CONTEXT] "
            "tags.\n"
            "Answer all the future questions using this context, refering and citing it whenever it is necessary. "
            "Please, read carefully and remember this context while we have this conversation. The context:\n"
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
            self.memory.history,

            # Finally, we add the last given user query.
            {"role": "user", "content": user_query}
        ]
        return messages

    def infer(self, message: str):
        # If ChatBot has no attribute "context" (context hasn't been provided) it prints an error and returns an empty string.
        if not hasattr(self, "context"):
            print("ERROR: You are asking the LLM but it hasn't got its context loaded.")
            return ""
            # prompt = self.default_prompt.format(context="", input=message, history=self.memory.compile())
        else:
            prompt = self.build_prompt(context=self.context, user_query=message)
        
        answer = self.llm(prompt)
        self.memory.update_memory(human_msg=message, ai_msg=answer)

        return answer

    def retrieve_context_from_db(self, query, vector_db: VectorDB, k=3):
        context = vector_db.retrieve_context(query, k=k)
        if len(context) > 0:
            self.initialize_context(context=context)
            message = "Succesfully loaded context."
        else:
            message = "No context found in the Database."
        print(message)
        return message
        

    def __call__(self, message: str):
        self.infer(message=message)

if __name__=="__main__":
    cb = ChatBot()
    cb.infer("Hola")