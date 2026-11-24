# Chat com PDF

Faça perguntas sobre qualquer PDF usando a API do Claude. As respostas citam a página do documento e, quando a informação não existe, o programa diz que não encontrou.

## Como rodar

1. Instale o Python 3.10 ou mais novo.
2. Crie e ative o ambiente virtual: python -m venv venv e depois venv\Scripts\Activate.ps1 (Windows).
3. Instale as dependências: pip install -r requirements.txt
4. Defina sua chave: $env:ANTHROPIC_API_KEY="sua-chave"
5. Rode: python chat_com_pdf.py seu_arquivo.pdf

## Como foi testado

Testei com 5 perguntas sobre um documento de plano estratégico: fatos explícitos, informação combinada de duas seções, e uma pergunta cuja resposta não estava no documento. Todas as respostas vieram corretas e com página citada.

## Limitações

Envia o PDF inteiro em cada pergunta, então só funciona bem com documentos pequenos. Próximo passo: RAG com embeddings para documentos grandes.
