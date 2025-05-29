import os
import json
import pickle 
import numpy as np 
import openai 
import cohere
from pathlib import Path
from tqdm import tqdm 
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage 
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display 


from .simple_vectorDB import SimpleVectorDB
from .document_processor import ContextGenerator
from .configs import USE_THRESHOLD, USE_RERANK, SIMILARITY_THRESHOLD, USE_HYDE


# Carregar variáveis do ambiente
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class Agent: 
    def __init__(self, doc_path: str): 
        self.doc_path = doc_path 
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.hyde_rag = ChatOpenAI(model="gpt-4o-mini",
                                        n=1,
                                        max_tokens=50, 
                                        temperature=0.0)
        self.tools = [self.search_text]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.context_generator = ContextGenerator(doc_source=self.doc_path)
    
    def search_text(self, query: str, k: int = 3):
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
        
        vector_db = SimpleVectorDB(name=name, api_key=os.getenv("OPENAI_API_KEY"))
        
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
            hypothetical_doc = response.content.strip()
            hypothetical_doc = hypothetical_doc.split("\n")[0]
            search_query = hypothetical_doc
        
    
        results = vector_db.search(search_query, k=k, use_rerank=USE_RERANK, rerank_top_n=k)
        return results # Retorna a lista de dicionários de resultados
    
    def assistant(self, state: MessagesState):
        query = state["messages"][-1].content
        results = self.search_text(query) 

        if USE_THRESHOLD and (not results or results[0]["similarity"] < SIMILARITY_THRESHOLD):
            return {"messages": [HumanMessage(content="""Desculpe, não encontrei contexto suficiente para responder. 
    Você poderia reformular sua pergunta com mais detalhes?""")]}

        context_msgs = [
            SystemMessage(content=(
                f"[sim={r['similarity']:.2f}]\n"
                f"Chunk: {r['chunk']}\n"
                f"Contexto Gerado: {r['context']}" 
            ))
            for r in results
        ]
        
        sys_msg = SystemMessage(content= """

    Você é  um assistente que usa documentos como fonte.

    Busque informações chamando a ferramenta 'search_text'.

    Baseie sua resposta exclusivamente nas informações do contexto acima, 

    e utilize-o somente se ele realmente ajudar a responder à pergunta; 

    caso contrário, não invente nada. 

    Não responda a pergunta se o contexto não tiver nada haver com a pergunta""")
        
        prompt_messages = [sys_msg] + context_msgs + state["messages"]

        return {"messages": [self.llm_with_tools.invoke(prompt_messages)]} 
    
    def build_graph(self):
        builder = StateGraph(MessagesState)
        builder.add_node("assistant", self.assistant)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        return builder.compile()
    
    def run_query(self, query):
        graph = self.build_graph()
        
        final_state = graph.invoke({"messages": [HumanMessage(content=query)]})
        
        
        if final_state and 'messages' in final_state and final_state['messages']:
            final_state['messages'][-1].pretty_print()
        
        return final_state

if __name__ == "__main__":
    try:
        print("Iniciando Agente (versão de query única)...")
        document_path = "src/data/Dom_Casmurro.txt"

        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent 
        document_path_abs = project_root / document_path 


        if not document_path_abs.exists():
            original_path_attempt = Path(document_path)
            if not original_path_attempt.exists():
                 print(f"ERRO: Arquivo de documento '{document_path_abs}' (ou '{original_path_attempt}') não encontrado.")
                 exit()
            else:
                document_path_abs = original_path_attempt


        agent_executor = Agent(doc_path=str(document_path_abs))
        
        user_query = input("Você: ")
        if user_query.strip():
            print("Agente: Processando...")
            agent_executor.run_query(user_query)
        else:
            print("Nenhuma query fornecida.")

    except ValueError as e:
        print(f"Erro de Valor: {e}")
    except FileNotFoundError as e: # 
        print(f"Erro de Arquivo não Encontrado: {e}")
    except Exception as e:
        import traceback
        print(f"Um erro inesperado ocorreu: {e}")
        traceback.print_exc()