# Agente RAG com Memória Conversacional e Reranking

## 1. Introdução e Cenário

Este projeto implementa um Agente conversacional capaz de responder a perguntas sobre um documento específico utilizando LLM. O sistema emprega técnicas de RAG , incluindo processamento de texto, geração de "contextos de ponte" para chunks de texto, criação de embeddings, busca vetorial, reranking opcional e. O objetivo é demonstrar a aplicação prática de conceitos de LLMs e Agentes na solução de problemas de busca e resposta em documentos.

## 2. Documento Utilizado

Para este projeto, o documento de texto escolhido como base de conhecimento é **"Dom Casmurro" de Machado de Assis**.

* **Localização Esperada:** O arquivo de texto do documento (ex: `Dom_Casmurro.txt`) deve ser colocado dentro do diretório `src/data/`. O agente está configurado por padrão para procurar por `src/data/Dom_Casmurro.txt` através de um caminho relativo à sua própria localização.

## 3. Estrutura de Pastas do Projeto

A estrutura de pastas do projeto é organizada da seguinte forma:

```text
RAG-RERANKING-AGENT/
├── .venv/
├── src/
│   ├── __pycache__/
│   ├── data/
│   │   └── Dom_Casmurro.txt
│   ├── __init__.py
│   ├── agent.py
│   ├── configs.py
│   ├── document_processor.py
│   └── simple_vectorDB.py
├── .env
├── .gitignore
├── pyproject.toml
└── README.md

## 4. Configuração do Ambiente

### 4.1. Pré-requisitos
* Python 3.9 ou superior.
* `uv` (ferramenta de gerenciamento de pacotes Python). Se não tiver, instale com `pip install uv` ou conforme as instruções oficiais.

### 4.2. Passos de Instalação
1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO_NO_GITHUB>
    cd RAG-RERANKING-AGENT 
    ```
2.  **Crie e ative um ambiente virtual usando `uv`:**
    ```bash
    uv venv
    ```
    Para ativar (Linux/macOS):
    ```bash
    source .venv/bin/activate
    ```
    Para ativar (Windows):
    ```bash
    .venv\Scripts\activate
    ```
3.  **Instale as dependências a partir do `pyproject.toml`:**
    ```bash
    uv pip install -e .
    ```
    Isso instalará todas as bibliotecas listadas no `pyproject.toml`, incluindo `langchain`, `langgraph`, `openai`, `cohere`, `numpy`, etc.

4.  **Configure as Variáveis de Ambiente:**
    * Crie um arquivo chamado `.env` na raiz do projeto (`RAG-RERANKING-AGENT/.env`).
    * Adicione suas chaves de API:
        ```env
        OPENAI_API_KEY="sk-suaChaveDaOpenAIaqui"
        COHERE_API_KEY="suaChaveDaCohereAqui"
        ```

## 5. Funcionalidades Principais e Scripts

* **`document_processor.py`:**
    * `DocumentProcessor`: Lê o arquivo `.txt` e o divide em chunks usando `TokenTextSplitter` do LangChain. A configuração atual em `configs.py` define `CHUNK_SIZE` (padrão 500 tokens) e o `document_processor.py` aplica uma **sobreposição (overlap) de 50%** entre os chunks.
    * `ContextGenerator`: Para cada chunk, gera um "contexto de ponte" utilizando `gpt-4o-mini`. Este modelo recebe o chunk atual, o anterior e o próximo, e é guiado por um prompt detalhado com exemplos de "como fazer" e "como não fazer" para criar um resumo contextualizador. Os resultados são salvos em um arquivo JSON.

* **`simple_vectorDB.py`:**
    * `SimpleVectorDB`: Carrega os dados do JSON (chunks e seus contextos de ponte).
    * Cria um **texto combinado** (`chunk original + contexto de ponte gerado`) para cada entrada.
    * Gera **embeddings** para esses textos combinados usando `text-embedding-3-small` da OpenAI.
    * Armazena os embeddings e metadados (incluindo o chunk original e o contexto de ponte separadamente) em um arquivo Pickle (`.pkl`) para persistência.
    * Realiza buscas por similaridade e oferece **reranking opcional** dos resultados (usando o texto original do chunk) com `rerank-multilingual-v3.0` da Cohere. A lógica de retorno para buscas sem reranking foi corrigida para respeitar o parâmetro `k`.

* **`agent.py`:**
    * `Agent`: Orquestra o pipeline RAG usando LangGraph.
    * Implementa **memória conversacional** usando `RunnableWithMessageHistory`, permitindo que o agente lembre de interações anteriores dentro de uma mesma sessão.
    * Utiliza um **prompt de sistema principal (`AGENT_SYSTEM_PROMPT_TEXT`) embutido** que guia o `gpt-4o-mini` sobre seu papel, quando e como usar a ferramenta `search_text`, e como lidar com a ausência de resultados.
    * Opcionalmente aplica a técnica **HyDE** para refinar a query de busca.
    * Chama a ferramenta `search_text` (que interage com `ContextGenerator` e `SimpleVectorDB`) para buscar informações.
    * Opcionalmente aplica um **threshold de similaridade** aos resultados (lógica agora mais integrada ao fluxo de decisão do LLM com base no output da ferramenta).
    * Gera a resposta final ao usuário com base no contexto recuperado e no histórico da conversa.

* **`configs.py`:**
    * Centraliza flags e parâmetros como `USE_RERANK` (padrão `False`), `USE_THRESHOLD` (padrão `True`), `SIMILARITY_THRESHOLD` (padrão `0.5`), `USE_HYDE` (padrão `False`), e `CHUNK_SIZE` (padrão `500` tokens). A variável `MODEL_EMBED` foi removida por não estar em uso.

## 6. Como Usar o Agente

Após a configuração completa:

1.  **Verifique o Documento:** Certifique-se de que o arquivo de texto (ex: `Dom_Casmurro.txt`) está no diretório `src/data/`.
2.  **Execute o Script do Agente:**
    No terminal, a partir do diretório raiz do projeto (`RAG-RERANKING-AGENT/`), execute:
    ```bash
    python -m src.agent
    ```
3.  **Interaja:**
    * O script iniciará e mostrará: "Iniciando Agente Conversacional de Documentos..." seguido por "Agente pronto. Sessão ID: ... Digite 'sair' para terminar."
    * Digite suas perguntas sobre o documento no prompt "Você: " e pressione Enter.
    * Para encerrar a conversa, digite `sair`.

    **Primeira Execução para um Novo Documento:**
    Se for a primeira vez que o agente é executado com um novo `doc_path` (ou se os arquivos processados em `src/data/` forem removidos), o sistema realizará todo o pré-processamento:
    * Geração de contextos de ponte para todos os chunks (pode levar tempo e consumir tokens da API OpenAI).
    * Geração de embeddings para todos os chunks combinados (também consome tokens da API OpenAI).
    As execuções subsequentes para o mesmo documento carregarão os dados processados do disco, tornando o início e as buscas muito mais rápidos.

## 7. Engenharia de Prompt Aplicada

A engenharia de prompt é um componente chave deste projeto:

1.  **Geração de "Contexto de Ponte" (`document_processor.py` -> `ContextGenerator.situate_context`):**
    * O prompt para `gpt-4o-mini` é estruturado para receber o chunk anterior, o principal e o seguinte.
    * Contém instruções detalhadas sobre o objetivo (criar uma frase curta e informativa que sirva como "contexto de ponte"), o formato da saída (concisão, idealmente 15-30 palavras), e restrições importantes (como **EVITAR OBRIGATORIAMENTE** frases de abertura formulaicas como "Este trecho...").
    * Inclui exemplos positivos e negativos (`COMO FAZER` e `COMO NÃO FAZER`) para ilustrar o comportamento desejado. As linhas "Motivo do erro/acerto..." estão presentes no prompt atual enviado ao LLM; para uma versão final, essas poderiam ser removidas do prompt literal, pois o LLM aprende pelo padrão entrada/saída dos exemplos.

2.  **Prompt de Sistema Principal do Agente (`agent.py` -> `AGENT_SYSTEM_PROMPT_TEXT`):**
    * Este prompt é embutido diretamente no `agent.py` e define a persona e o fluxo de trabalho do agente `gpt-4o-mini`.
    * Instrui o LLM a se considerar um especialista no documento.
    * Especifica que ele **DEVE OBRIGATORIAMENTE** usar a ferramenta `search_text` para perguntas que busquem informações no documento.
    * Detalha como o LLM deve formular a query para a ferramenta `search_text` (baseada na pergunta original do usuário).
    * Instrui sobre como agir com base no resultado da ferramenta `search_text`, incluindo como responder se a ferramenta retornar "Nenhum resultado relevante encontrado...".
    * Reforça a necessidade de usar apenas informações recuperadas, lembrar do histórico e responder em português.

3.  **HyDE (`agent.py` -> `search_text`):**
    * Se `USE_HYDE = True`, um prompt específico é usado para que o `gpt-4o-mini` gere uma resposta/documento hipotético com base na query do usuário, e essa resposta hipotética é usada para a busca vetorial.

## 8. Mitigação de Viés (Bias)

A mitigação de viés em LLMs é um esforço contínuo. As abordagens neste projeto que contribuem para reduzir seu impacto incluem:

1.  **Fundamentação no Documento (RAG):** A principal estratégia é o RAG. O `AGENT_SYSTEM_PROMPT_TEXT` instrui o LLM a basear suas respostas *exclusivamente* nos trechos recuperados do documento, o que limita a geração de informações baseadas apenas em seus dados de treinamento e potenciais vieses neles contidos.
2.  **Contextualização Focada:** A geração de "contextos de ponte" usa um escopo limitado (chunks adjacentes), o que pode reduzir a chance de o LLM introduzir vieses amplos ao interpretar o documento.
3.  **Prompts Objetivos:** As instruções nos prompts são formuladas para serem diretas e focadas na tarefa, buscando respostas baseadas em fatos do texto.
4.  **Escolha do Modelo:** Utiliza-se um modelo de fundação (`gpt-4o-mini`) sem fine-tuning específico neste projeto, o que significa que os vieses seriam os inerentes ao modelo pré-treinado.

É importante notar que o viés pode estar presente no próprio documento fonte, e o agente, por ser baseado nele, pode refletir esse viés. A mitigação completa exigiria técnicas mais avançadas, avaliação contínua e, possivelmente, curadoria de dados e fine-tuning específico, que estão fora do escopo deste projeto experimental.

## 9. Configurações (`configs.py`)

O arquivo `src/configs.py` permite ajustar rapidamente alguns comportamentos do agente:
* `USE_RERANK`: Ativa/desativa o reranking com Cohere.
* `USE_THRESHOLD`: Ativa/desativa a filtragem por similaridade na busca.
* `SIMILARITY_THRESHOLD`: O limiar de similaridade para considerar um chunk relevante.
* `USE_HYDE`: Ativa/desativa a busca com HyDE.
* `CHUNK_SIZE`: O tamanho dos chunks em tokens.

Experimente com esses valores para otimizar o desempenho para diferentes documentos ou tipos de query.

---