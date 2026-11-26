# Chat com PDF

**App online:** https://chat-com-pdf-b8yvktjhbvsnvneupjmnfx.streamlit.app


Faça perguntas sobre qualquer PDF usando a API do Claude. As respostas citam a página do documento e, quando a informação não existe, o programa diz que não encontrou.

O projeto tem duas versões:

- `chat_com_pdf.py`: manda o PDF inteiro em cada pergunta. Simples, mas só funciona bem com documentos pequenos.
- `chat_rag.py`: usa RAG. Divide o PDF em trechos, gera embeddings locais (modelo multilíngue gratuito, sem API extra), busca os trechos mais parecidos com a pergunta e manda só esses ao Claude. Mostra quais páginas foram usadas em cada resposta.

## Como rodar

1. Instale o Python 3.10 ou mais novo.
2. Crie e ative o ambiente virtual: `python -m venv venv` e depois `venv\Scripts\Activate.ps1` (Windows).
3. Instale as dependências: `pip install -r requirements.txt`
4. Defina sua chave de API da Anthropic: `$env:ANTHROPIC_API_KEY="sua-chave"`
5. Rode uma das versões: `python chat_rag.py seu_arquivo.pdf`

Na primeira execução do RAG, o modelo de embeddings (cerca de 250 MB) é baixado automaticamente.

## Como foi testado

Testei as duas versões com 5 perguntas sobre um plano estratégico de 17 páginas: dois fatos explícitos, uma informação que juntava duas seções, uma sobre o meio do documento e uma cuja resposta não estava no PDF. As duas versões responderam corretamente às 5, inclusive dizendo que não encontraram a informação ausente.

## Limitações

- Funciona com PDFs que têm texto selecionável. PDFs escaneados (só imagem) não são lidos.
- Os testes foram feitos com um documento pequeno. Ainda não medi o desempenho com PDFs grandes.
- O RAG depende da busca: se ela não trouxer o trecho certo, o Claude não consegue responder.

## Próximos passos

Interface web com Streamlit e avaliação com um conjunto maior de perguntas.



