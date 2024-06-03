# Arquitetura

A arquitetura foi desenvolvida com base no [planejamento](docs/planning.md). O objetivo é criar um projeto que auxilie os usuários no aprimoramento de seus conhecimentos acadêmicos. Foi desenvolvido um modelo LLM que integra diferentes dados educacionais, entendendo as dificuldades dos usuários e sugerindo as melhores soluções de estudo.

Como se trata de um produto de dados, o foco não está em desenvolver um ambiente escalável via Chainlit, mas sim em envelopar o pipeline do modelo em uma API, permitindo que desenvolvedores integrem em aplicativos e sites. Este projeto possui dois fluxos principais:

1. **Inicialização do Banco Vetorizado e Modelo**:
   - Iniciar o banco vetorizado.
   - Iniciar o modelo.
   - Conectar ambos para disponibilizar a API e o site.

2. **Pipeline de Dados**:
   - Receber dados brutos.
   - Transformar os dados para o banco no fluxo de ETL.
   - Este pipeline deveria estar em outro projeto e integrado a orquestrações como Airflow ou Databricks.

# Biblioteca

Este projeto utiliza as bibliotecas listadas em [requirements.txt](requirements.txt). Algumas das principais soluções utilizadas foram:
- **OpenAI**: para utilização do modelo.
- **ChromaDB**: para armazenar os dados vetorizados de educação.
- **LangChain**: para integrar o modelo com os dados.
- **Chainlit**: para prototipação do uso do site.

# Instalação

Instale as dependências em `requirements.txt`.

## Instruções de instalação para jq

### Para Windows

1. Baixe o arquivo `jq` wheel para Python 3.11 [aqui](http://jeffreyknockel.com/jq/jq-1.4.0-cp311-cp311-win_amd64.whl).
2. Mova o arquivo para a pasta raiz do projeto.
3. Execute o seguinte comando no ambiente virtual:
    ```bash
    pip install jq-1.4.0-cp311-cp311-win_amd64.whl
    ```

### Para Linux

Execute o seguinte comando:
```bash
sudo apt-get install jq
```

## Instalação do Tesseract

Instale o pacote conforme seu sistema operacional seguindo as instruções [aqui](https://tesseract-ocr.github.io/tessdoc/Installation.html). Se estiver usando Windows, adicione o caminho do diretório de instalação ao `PATH` das variáveis de ambiente.

## Configuração do Pre-commit

Execute o comando para instalar os hooks de código:
```bash
pre-commit install
```

# Credenciais

Crie um arquivo `.env` com as credenciais conforme o `env.example`.

# Uso

- Testes e exploração inicial via notebook.
- Protótipo para apresentação interna via Chainlit.
- Modelo em produção com API e Docker.

O modelo pode ser executado via API com o comando:
```bash
uvicorn src.api.llm_apy:app --reload
```

Para testar o site, utilize o comando:
```bash
chainlit run src/webapp.py --port 8001
```

Com a API em execução a documentação está disponível em: [http://localhost:8000/docs](http://localhost:8000/docs)

# Construindo e Executando o Contêiner Docker

Construa a imagem Docker:
```bash
docker build -t study-journey-app .
```

Execute o contêiner Docker:
```bash
docker run -d -p 8000:8000 -p 8001:8001 --name study-journey-container study-journey-app
```

As portas para o Docker são as mesmas do ambiente local:
- [http://localhost:8001/](http://localhost:8001/) para o Chainlit.
- [http://localhost:8000/query](http://localhost:8000/query) para requisição de API.

# Conclusões

O projeto entrega todas as etapas inicialmente propostas no planejamento, como indexação dos dados, prompts para o modelo interagir com os dados e os usuários, e outputs como API e site de prototipação.

# Próximos Passos

Algumas considerações para pontos de melhorias incluem:
- **Tratamento de Dados**: Limpar as fontes de diferentes informações ajudará o modelo a identificar e entregar melhores respostas.
- **Feedback**: Coletar feedback sobre a qualidade das respostas, incluindo testes A/B para avaliar a eficácia das mudanças.
