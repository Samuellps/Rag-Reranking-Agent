import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from langchain_text_splitters import TokenTextSplitter
from configs import CHUNK_SIZE

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class DocumentProcessor:
    def __init__(self, doc_source: str):
        self.doc_source = doc_source
        self.max_tokens_per_chunk = CHUNK_SIZE
        self.text_splitter = TokenTextSplitter(
            encoding_name="cl100k_base",
            chunk_size=self.max_tokens_per_chunk,
            chunk_overlap=int(self.max_tokens_per_chunk * 0.5)
        )

    def get_chunks(self):
        file_path = Path(self.doc_source)
        if not file_path.is_file() or file_path.suffix.lower() != ".txt":
            raise ValueError(f"Arquivo de origem não é .txt válido: {self.doc_source}")

        with open(file_path, 'r', encoding='utf-8') as f:
            full_text_content = f.read()

        chunk_texts = self.text_splitter.split_text(full_text_content)
        return chunk_texts, full_text_content

class ContextGenerator:
    def __init__(self, doc_source: str):
        self.doc_source = doc_source
        self.client = OpenAI(api_key=openai_api_key)

    def situate_context(self, current_chunk: str, prev_chunk: str = None, next_chunk: str = None) -> str:
        prompt_parts = []

        if prev_chunk:
            prompt_parts.append(f"Contexto do chunk Anterior:\n<chunk_anterior>\n{prev_chunk}\n</chunk_anterior>\n")
        
        prompt_parts.append(f"chunk principal (para o qual o contexto é desejado):\n<chunk_principal>\n{current_chunk}\n</chunk_principal>\n")

        if next_chunk:
            prompt_parts.append(f"Contexto do chunk Seguinte:\n<chunk_seguinte>\n{next_chunk}\n</chunk_seguinte>\n")

        instruction = """INSTRUÇÃO PARA VOCÊ, IA:
Com base nos trechos fornecidos (Anterior, Principal, Seguinte), sua tarefa é gerar uma ÚNICA frase curta e informativa que sirva como um "contexto de ponte" para o TRECHO PRINCIPAL.
Esta frase deve:
1. Destacar a função ou o principal desenvolvimento que o TRECHO PRINCIPAL introduz, continua ou conclui, em relação aos seus vizinhos.
2. Ser concisa e direta (idealmente entre 15 e 30 palavras).
3. **EVITAR OBRIGATORIAMENTE** iniciar com expressões como "Este trecho...", "O Pedaço Principal...", "No trecho principal...", "O trecho apresentado...", "Neste chunk...", etc. Comece diretamente com a informação que situa o trecho.
4. O objetivo é criar um metadado que melhore a relevância da busca para o TRECHO PRINCIPAL, conectando-o ao fluxo narrativo/informativo de forma distintiva.

Veja exemplos de como fazer e como NÃO fazer:

---
Exemplo 1 (COMO NÃO FAZER - Exemplo Ruim):
Dado:
TRECHO ANTERIOR: "O cientista preparou o experimento com cuidado."
TRECHO PRINCIPAL: "Ele misturou as substâncias A e B no tubo de ensaio."
TRECHO SEGUINTE: "Uma fumaça colorida começou a sair do recipiente."

Saída Ruim (Evitar este estilo): "No Trecho Principal, o cientista mistura as substâncias A e B, dando continuidade ao seu experimento."
(Motivo do erro no exemplo ruim: Começa com "No Trecho Principal..." e é uma descrição simples do chunk principal, sem focar na transição ou função de ponte de forma distintiva.)
---
Exemplo 2 (COMO FAZER - Exemplo Ideal):
Dado:
TRECHO ANTERIOR: "O cientista preparou o experimento com cuidado."
TRECHO PRINCIPAL: "Ele misturou as substâncias A e B no tubo de ensaio."
TRECHO SEGUINTE: "Uma fumaça colorida começou a sair do recipiente."

Saída Ideal (Seguir este estilo): "Após preparar o experimento, a mistura das substâncias A e B pelo cientista desencadeia uma reação visível e colorida."
(Motivo do acerto no exemplo ideal: Conecta com o anterior, descreve a ação principal e seu resultado imediato que leva ao próximo, sem usar frases de abertura proibidas e focando na progressão.)
---

Responda APENAS com a frase contextualizadora para o TRECHO PRINCIPAL que você está analisando AGORA (os trechos no topo deste prompt). Não adicione nenhuma outra palavra ou explicação.
"""
        final_prompt = "\n".join(prompt_parts) + instruction
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": final_prompt},
            ],
            max_tokens=150, # Reduzido, pois o contexto de entrada é menor
            temperature=0.0
        )
        return response.choices[0].message.content

    def _json_path(self) -> str:
        stem = Path(self.doc_source).stem
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        return str(data_dir / f"{stem}_chunks_with_context_adj.json") # Nome do arquivo alterado
    
    def generate_contexts(self) -> str:
        json_path = self._json_path()
        if os.path.exists(json_path): # Verifica se o JSON com novo nome já existe
            return json_path
        
        processor = DocumentProcessor(self.doc_source)
        chunk_texts_list, _ = processor.get_chunks() # Não precisamos mais do full_document_text aqui
        results = []
        
        num_chunks_total = len(chunk_texts_list)

        for i, current_chunk_text in enumerate(chunk_texts_list):
            prev_chunk_text = chunk_texts_list[i-1] if i > 0 else None
            next_chunk_text = chunk_texts_list[i+1] if i < (num_chunks_total - 1) else None
            
            contextualized_text = self.situate_context(
                current_chunk=current_chunk_text,
                prev_chunk=prev_chunk_text,
                next_chunk=next_chunk_text
            )
            results.append({
                "chunk": current_chunk_text,
                "context": contextualized_text
            })
            with open(json_path, 'w', encoding='utf-8') as fp:
                json.dump(results, fp, ensure_ascii=False, indent=4)
        return json_path

if __name__ == "__main__":
    try:
        # Teste da geração de contexto com chunks adjacentes
        context_gen = ContextGenerator(doc_source="data/Dom_Casmurro.txt")
        json_file = context_gen.generate_contexts()
        print(f"Contextos (com chunks adjacentes) gerados em: {json_file}")

    except ValueError as e:
        print(f"Erro de Valor: {e}")
    except FileNotFoundError:
        print(f"Erro: Arquivo 'data/Dom_Casmurro.txt' não encontrado.")
    except Exception as e:
        print(f"Erro inesperado: {e}")