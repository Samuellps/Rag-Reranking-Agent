import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI as OpenAICoreClient
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from typing import TypedDict, Annotated
import operator
from langchain_core.chat_history import BaseChatMessageHistory, ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from .simple_vectorDB import SimpleVectorDB
from .document_processor import ContextGenerator
from .configs import USE_THRESHOLD, USE_RERANK, SIMILARITY_THRESHOLD, USE_HYDE

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

AGENT_SYSTEM_PROMPT_TEXT = """Você é um assistente de IA especializado no conteúdo de um documento fornecido. Seu objetivo é responder às perguntas do usuário com base nesse documento.
Para fazer isso, você TEM ACESSO A UMA FERRAMENTA chamada 'search_text'.

Fluxo de trabalho:
1. Quando o usuário fizer uma pergunta, avalie se você precisa de informações do documento para respondê-la. Se for uma saudação ou conversa geral, responda apropriadamente sem usar a ferramenta.
2. Se precisar de informações do documento para responder à pergunta, VOCÊ DEVE usar a ferramenta 'search_text' com uma query (termo de busca) apropriada e concisa, derivada da pergunta do usuário.
3. Após receber os resultados da ferramenta 'search_text':
    a. Se os resultados da ferramenta indicarem explicitamente "Nenhum resultado relevante encontrado na base de conhecimento." ou algo similar, então responda ao usuário: "Desculpe, não encontrei informações suficientemente relevantes no documento para responder à sua pergunta com confiança neste momento."
    b. Caso contrário, use os trechos e contextos fornecidos pela ferramenta para formular uma resposta precisa e concisa à pergunta original do usuário. Integre a informação naturalmente na sua resposta.
4.  Lembre-se do histórico da conversa para manter o contexto em respostas subsequentes.
5.  Responda em português brasileiro.
"""

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

class Agent:
    def __init__(self, doc_path: str):
        self.doc_path = doc_path
        self.llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0.0)
        self.hyde_rag = ChatOpenAI(
            model="gpt-4o-mini",
            n=1,
            max_tokens=50,
            temperature=0.0,
            openai_api_key=openai_api_key
        )
        self.tools = [self.search_text]
        self.context_generator = ContextGenerator(doc_source=self.doc_path)
        self.graph = self._build_graph()

    def search_text(self, query: str, k: int = 3) -> str:
        """
        A ferramenta search_text é responsável por realizar buscas no banco de conhecimento do agente, 
        garantindo que informações relevantes sejam recuperadas antes da geração de uma resposta. 
        Essa ferramenta é essencial para aprimorar a precisão do agente, especialmente quando a pergunta é muito específica.
        
        query: str
            Tipo: string
            Descrição: Representa o termo ou a frase que o usuário deseja buscar no banco de dados vetorial.

        k: int = 3
            Tipo: inteiro
            Descrição: Define o número máximo de resultados mais relevantes que serão retornados pela busca.
        """
        json_path = self.context_generator.generate_contexts()
        with open(json_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)

        name = Path(self.doc_path).stem
        vector_db = SimpleVectorDB(name=name, api_key=openai_api_key)
        vector_db.load_data(json_data)

        search_query = query
        if USE_HYDE:
            hyde_prompt = f"""
            Com base na query: {query}, gere uma única frase afirmativas imulando uma resposta, 
            use seu contexto de treinamento para simular essa resposta.
            Ex.:

            Query: Qual é a espessura do compensado?
            Simulação da respsotas: O compensado de MDF geralmente tem a espessura 0.75
            """
            response = self.hyde_rag.invoke([SystemMessage(content=hyde_prompt)])
            hypothetical_doc = response.content.strip().split("\n")[0]
            search_query = hypothetical_doc
        
        results = vector_db.search(search_query, k=k, use_rerank=USE_RERANK, rerank_top_n=k)
        
        if not results:
            return "Nenhum resultado relevante encontrado na base de conhecimento."
        
        formatted_results = "\n\n".join([
            f"Trecho {idx+1} (Similaridade: {res['similarity']:.2f}):\n{res['chunk']}\nContexto Gerado: {res['context']}"
            for idx, res in enumerate(results)
        ])
        return formatted_results

    def assistant_node(self, state: AgentState):
        current_messages = state["messages"]
        processed_messages_for_llm = []
        has_system_prompt = False

        if current_messages and isinstance(current_messages[0], SystemMessage) and current_messages[0].content == AGENT_SYSTEM_PROMPT_TEXT:
            has_system_prompt = True
            processed_messages_for_llm = current_messages[:]
        else:
            processed_messages_for_llm.append(SystemMessage(content=AGENT_SYSTEM_PROMPT_TEXT))
            for msg in current_messages:
                if isinstance(msg, SystemMessage) and msg.content == AGENT_SYSTEM_PROMPT_TEXT:
                    continue
                processed_messages_for_llm.append(msg)
        
        llm_with_tools = self.llm.bind_tools(self.tools)
        response_message = llm_with_tools.invoke(processed_messages_for_llm)
        
        return {"messages": [response_message]}

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("assistant", self.assistant_node)
        builder.add_node("tools", ToolNode(self.tools))
        builder.set_entry_point("assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
            {"tools": "tools", END: END}
        )
        builder.add_edge("tools", "assistant")
        return builder.compile()

_SESSION_STORE = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]

if __name__ == "__main__":
    print("Iniciando Agente Conversacional de Documentos...")
    document_path = "data/Dom_Casmurro.txt" 

    if not Path(document_path).exists():
        print(f"ERRO: Arquivo de documento '{document_path}' não encontrado.")
        exit()

    try:
        agent_executor = Agent(doc_path=document_path)
        agent_with_memory = RunnableWithMessageHistory(
            agent_executor.graph,
            get_session_history,
            input_messages_key="input_query_text", 
            history_messages_key="messages", 
        )
        session_id = "conversa_com_memoria_e_prompts_embutidos_vClean" 
        print(f"\nAgente pronto. Sessão ID: {session_id}. Digite 'sair' para terminar.")

        initial_history = get_session_history(session_id)
        if not initial_history.messages:
             initial_history.add_message(SystemMessage(content=AGENT_SYSTEM_PROMPT_TEXT))

        while True:
            user_input = input("Você: ")
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Agente: Encerrando...")
                break
            if not user_input.strip():
                continue

            print("Agente: Processando...")
            response_state = agent_with_memory.invoke(
                {"input_query_text": user_input}, 
                config={"configurable": {"session_id": session_id}}
            )
            
            if response_state and 'messages' in response_state and response_state['messages']:
                ai_response_message = response_state['messages'][-1]
                if isinstance(ai_response_message, AIMessage):
                    if ai_response_message.tool_calls:
                        print("Agente: (decidiu usar ferramenta, o ciclo continua internamente)")
                    else:
                        print(f"Agente: {ai_response_message.content}")
                else: 
                    ai_response_message.pretty_print()
            else:
                print("Agente: Não houve resposta estruturada do agente.")

    except Exception as e:
        import traceback
        print(f"Um erro inesperado ocorreu no Agente: {e}")
        traceback.print_exc()