# Agente RAG com Reranking para Perguntas e Respostas sobre Documentos

## 1. Introdução e Cenário

Este projeto foi desenvolvido como parte de um teste técnico e demonstra a criação de um Agente capaz de responder a perguntas sobre um documento específico utilizando um Modelo de Linguagem Grande (LLM). [cite: 1] O sistema emprega técnicas de RAG (Retrieval Augmented Generation), processamento de texto, geração de embeddings, busca vetorial e reranking para fornecer respostas contextualmente relevantes e baseadas no conteúdo do documento fornecido. O objetivo é demonstrar a aplicação prática de conceitos de LLMs e Agentes na solução de problemas reais. [cite: 9, 10]

## 2. Documento Utilizado

Para este projeto, o documento de texto escolhido foi **"Dom Casmurro" de Machado de Assis**, uma obra em domínio público. [cite: 2]

* **Localização Esperada:** O arquivo de texto do documento (ex: `Dom_Casmurro.txt`) deve ser colocado dentro do diretório `src/data/`. O agente está configurado por padrão para procurar por `src/data/Dom_Casmurro.txt`.

## 3. Estrutura de Pastas do Projeto

A estrutura de pastas do projeto é organizada da seguinte forma:

RAG-RERANKING-AGENT/
├── .venv/                     # Diretório do ambiente virtual (criado pelo uv)
├── src/                       # Código fonte principal do projeto
│   ├── __pycache__/          # Cache do Python
│   ├── data/                 # Diretório para armazenar documentos de texto e dados gerados (JSON, PKL)
│   │   └── Dom_Casmurro.txt  # Exemplo de documento de texto
│   ├── __init__.py         # Torna 'src' um pacote Python
│   ├── agent.py              # Define a classe Agent e o fluxo principal do RAG
│   ├── configs.py            # Configurações do projeto (flags, thresholds, etc.)
│   ├── document_processor.py # Classes DocumentProcessor e ContextGenerator
│   ├── simple_vectorDB.py    # Classe SimpleVectorDB para embeddings e busca
│   └── prompts.py            # (Necessário se agent.py importar prompts daqui)
├── .env                       # Arquivo para variáveis de ambiente (chaves de API)
├── .gitignore                 # Especifica arquivos não rastreados pelo Git
├── pyproject.toml             # Define metadados do projeto e dependências
└── README.md                  # Este arquivo

*(Baseado na imagem fornecida e nas importações dos scripts)*

## 4. Configuração do Ambiente

Siga os passos abaixo para configurar e executar o projeto:

### 4.1. Pré-requisitos
* Python 3.9 ou superior.
* `uv` (ferramenta de gerenciamento de pacotes Python). Se não tiver, instale com `pip install uv`.

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
4.  **Configure as Variáveis de Ambiente:**
    * Crie um arquivo chamado `.env` na raiz do projeto (`RAG-RERANKING-AGENT/.env`).
    * Adicione suas chaves de API neste arquivo. Ele deve ter o seguinte formato:
        ```env
        OPENAI_API_KEY="sk-suaChaveDaOpenAIaqui"
        COHERE_API_KEY="suaChaveDaCohereAqui"
        ```
    * Substitua pelos seus valores reais.

## 5. Funcionalidades Principais e Scripts

O projeto é modularizado nos seguintes scripts dentro do diretório `src/`:

* **`document_processor.py`:**
    * `DocumentProcessor`: Lê o arquivo `.txt`, divide o texto em chunks usando `TokenTextSplitter` do LangChain (com `CHUNK_SIZE` de `configs.py` e 50% de overlap).
    * `ContextGenerator`: Para cada chunk, gera um "contexto de ponte" usando `gpt-4o-mini` e os chunks adjacentes. Salva os chunks e seus contextos em um arquivo JSON (ex: `src/data/NomeDoDoc_chunks_with_context_adj.json`).
* **`simple_vectorDB.py`:**
    * `SimpleVectorDB`: Carrega os dados do JSON. Cria um texto combinado (chunk original + contexto de ponte) para cada entrada e gera embeddings usando `text-embedding-3-small` da OpenAI. Armazena embeddings e metadados em um arquivo Pickle (`.pkl`) para persistência. Realiza buscas por similaridade e oferece reranking opcional dos resultados (usando o texto original do chunk) com `rerank-multilingual-v3.0` da Cohere.
* **`agent.py`:**
    * `Agent`: Orquestra o pipeline RAG usando LangGraph.
        * Recebe a query do usuário.
        * Utiliza o `ContextGenerator` e `SimpleVectorDB` (através da ferramenta `search_text`) para buscar informações relevantes.
        * Opcionalmente aplica HyDE para refinar a query de busca.
        * Opcionalmente aplica um threshold de similaridade aos resultados.
        * Injeta os chunks recuperados e seus contextos de ponte em um prompt para o `gpt-4o-mini`.
        * Gera a resposta final ao usuário.
        * Implementa memória conversacional usando `RunnableWithMessageHistory` para manter o contexto entre múltiplas interações na mesma sessão.
* **`configs.py`:**
    * Centraliza flags e parâmetros como `USE_RERANK`, `USE_THRESHOLD`, `SIMILARITY_THRESHOLD`, `USE_HYDE`, e `CHUNK_SIZE`.
* **`prompts.py` (Presumido):**
    * O script `agent.py` fornecido ainda importa `NOT_ENOUGH_CONTEXT_MSG` e `SIMPLE_PROMPT` de `from .prompts import ...`. Portanto, este arquivo deve existir em `src/` e conter estas constantes de string, ou os prompts precisam ser embutidos diretamente em `agent.py` (conforme discutimos anteriormente).
        * *(Nota: A última versão do `agent.py` que refinei para você já embutia o `AGENT_SYSTEM_PROMPT_TEXT`. Se você estiver usando essa versão, o `prompts.py` pode não ser mais necessário para essas constantes específicas.)*

## 6. Como Usar o Agente

Após a configuração e instalação das dependências:

1.  **Certifique-se de que o documento de texto** (ex: `Dom_Casmurro.txt`) está no diretório `src/data/`.
2.  **Execute o script do agente:**
    Navegue até o diretório raiz do projeto (`RAG-RERANKING-AGENT/`) e execute o agente como um módulo (isso garante que as importações relativas funcionem corretamente):
    ```bash
    python -m src.agent 
    ```
    Alternativamente, se estiver dentro do diretório `src/`:
    ```bash
    python agent.py
    ```
3.  **Interaja com o Agente:**
    * O script iniciará um loop de chat no terminal.
    * Você verá uma mensagem como "Agente pronto. Sessão ID: ... Digite 'sair' para terminar."
    * Digite sua pergunta e pressione Enter.
    * Para encerrar a conversa, digite `sair`, `exit` ou `quit`.

    **Observação sobre a Primeira Execução:** A primeira vez que você fizer uma pergunta sobre um novo documento, o sistema precisará:
    * Gerar os contextos de ponte para todos os chunks (envolvendo múltiplas chamadas à API do `gpt-4o-mini`).
    * Gerar os embeddings para todos os chunks (envolvendo múltiplas chamadas à API `text-embedding-3-small`).
    Isso pode levar um tempo considerável dependendo do tamanho do documento. Nas execuções subsequentes para o mesmo documento, os dados processados (JSON e `.pkl`) serão carregados do disco, tornando as respostas muito mais rápidas.

## 7. Engenharia de Prompt Aplicada [cite: 3]

A engenharia de prompt foi crucial em duas áreas principais:

1.  **Geração de "Contexto de Ponte" (`document_processor.py` -> `ContextGenerator.situate_context`):**
    * **Objetivo:** Criar um resumo curto e informativo para cada chunk que o situe em relação aos seus vizinhos (anterior e próximo), melhorando a relevância para buscas.
    * **Técnicas Aplicadas:**
        * **Prompt Detalhado com Instruções Claras:** O prompt especifica o formato da saída (frase única, concisa), o que deve ser destacado (função/desenvolvimento do chunk principal em relação aos vizinhos), e o que deve ser evitado.
        * **Instruções Negativas Explícitas:** A instrução "**EVITAR OBRIGATORIAMENTE** iniciar com expressões como 'Este trecho...', 'O Pedaço Principal...'" foi adicionada para combater respostas formulaicas.
        * **Exemplos (Few-Shot/One-Shot com Negativo):** O prompt inclui exemplos de "COMO NÃO FAZER" (Saída Ruim) e "COMO FAZER" (Saída Ideal), demonstrando o estilo de resposta esperado e os erros a serem evitados. Isso ajuda o modelo a entender melhor a tarefa.
        * *(Observação: Na versão atual do `document_processor.py` fornecida, as linhas "Motivo do erro/acerto..." nos exemplos estão incluídas no prompt enviado à IA. Para otimização, essas linhas de "motivo" poderiam ser removidas do prompt final, pois são mais para entendimento humano).*

2.  **Geração da Resposta Final do Agente (`agent.py` -> `AGENT_SYSTEM_PROMPT_TEXT`):**
    * **Objetivo:** Instruir o LLM principal sobre como se comportar, como usar a ferramenta de busca (`search_text`), e como formular respostas baseadas no contexto recuperado e no histórico da conversa.
    * **Técnicas Aplicadas:**
        * **Definição de Persona e Objetivo:** "Você é um assistente de IA especializado no conteúdo de um documento fornecido..."
        * **Instrução de Uso de Ferramenta:** Explica claramente quando e como o LLM DEVE usar a ferramenta `search_text`.
        * **Tratamento de Casos Específicos:** Instrui o LLM sobre como responder se a ferramenta retornar "Nenhum resultado relevante encontrado...".
        * **Foco na Qualidade da Resposta:** Pede respostas precisas, concisas e que integrem naturalmente a informação.
        * **Consciência do Histórico:** Lembra o LLM de considerar o histórico da conversa.

3.  **HyDE (Hypothetical Document Embeddings) (`agent.py` -> `search_text`):**
    * Se ativado (`USE_HYDE = True`), um prompt específico é usado para fazer o `gpt-4o-mini` gerar uma resposta hipotética à pergunta do usuário. Essa resposta hipotética, que se espera estar semanticamente mais próxima dos chunks relevantes, é então usada para a busca vetorial.

## 8. Mitigação de Viés (Bias) [cite: 5]

Mitigar completamente o viés em LLMs é um desafio complexo e contínuo. Neste projeto, as seguintes abordagens contribuem para reduzir o impacto de vieses potenciais e promover respostas mais neutras e baseadas em fatos do documento:

1.  **Fundamentação no Documento (RAG):** A principal estratégia é o próprio RAG. Ao forçar o LLM a basear suas respostas em trechos recuperados do documento fonte, reduz-se a chance de ele gerar informações baseadas apenas em seus dados de treinamento internos, que podem conter vieses. O `AGENT_SYSTEM_PROMPT_TEXT` reforça: "use os trechos e contextos fornecidos pela ferramenta para formular uma resposta precisa e concisa".
2.  **Contextualização Local:** A geração de "contextos de ponte" usando apenas chunks adjacentes foca em informações locais, em vez de tentar uma interpretação do documento inteiro que poderia ser mais suscetível a vieses do LLM ao preencher lacunas.
3.  **Prompts Específicos e Neutros:**
    * As instruções nos prompts (tanto para gerar o contexto de ponte quanto para a resposta final do agente) são formuladas para serem o mais objetivas possível, focando na tarefa e no conteúdo.
    * Evitar linguagem carregada ou que possa induzir o LLM a respostas tendenciosas.
4.  **Ausência de Fine-tuning Específico:** Este projeto utiliza modelos pré-treinados (`gpt-4o-mini`). Se fosse realizado um fine-tuning, a seleção e curadoria de um dataset de fine-tuning diverso e imparcial seria uma etapa crucial para mitigação de viés.
5.  **Transparência (Implícita):** Ao apresentar os trechos recuperados (com score de similaridade) junto à resposta, o sistema oferece uma forma (ainda que não direta ao usuário final no output do chat) de verificar a base da resposta do LLM. Em sistemas mais avançados, citar as fontes diretamente na resposta ao usuário é uma boa prática.

**Limitações e Próximos Passos para Mitigação de Viés:**
* O viés pode estar presente no documento fonte. O agente refletirá o conteúdo do documento.
* Modelos LLM, mesmo os mais avançados, podem exibir vieses inerentes aos seus dados de treinamento.
* **Melhorias Futuras:** Poderiam incluir técnicas como a solicitação explícita ao LLM para considerar múltiplas perspectivas, evitar generalizações indevidas, ou a utilização de ferramentas de análise de viés nas respostas geradas.

## 9. Configurações Adicionais (`configs.py`)

O arquivo `src/configs.py` permite ajustar o comportamento do agente:
* `USE_RERANK`: Ativa/desativa o reranking com Cohere.
* `USE_THRESHOLD`: Ativa/desativa a filtragem por similaridade.
* `SIMILARITY_THRESHOLD`: Define o limiar de similaridade.
* `USE_HYDE`: Ativa/desativa a busca com HyDE.
* `CHUNK_SIZE`: Define o tamanho dos chunks em tokens.

Consulte e modifique este arquivo para experimentar diferentes configurações.

---