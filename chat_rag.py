"""
Chat com PDF usando RAG (Retrieval-Augmented Generation).

Em vez de mandar o PDF inteiro a cada pergunta, o programa:
  1. divide o PDF em trechos (chunks), guardando a página de cada um;
  2. transforma cada trecho em um vetor (embedding) com um modelo local e gratuito;
  3. a cada pergunta, busca os trechos mais parecidos com ela;
  4. manda só esses trechos ao Claude.

Setup:
    pip install anthropic pypdf fastembed numpy

Uso:
    python chat_rag.py documento.pdf
"""
import sys

import anthropic
import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader

MODEL = "claude-sonnet-5-5"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800   # caracteres por trecho
OVERLAP = 100      # sobreposição entre trechos vizinhos
TOP_K = 4          # quantos trechos enviar ao Claude por pergunta


def ler_paginas(caminho: str) -> list[tuple[int, str]]:
    leitor = PdfReader(caminho)
    return [(i + 1, p.extract_text() or "") for i, p in enumerate(leitor.pages)]


def dividir_em_chunks(paginas: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Divide cada página em trechos, mantendo o número da página."""
    chunks = []
    for numero, texto in paginas:
        texto = " ".join(texto.split())
        inicio = 0
        while inicio < len(texto):
            fim = inicio + CHUNK_SIZE
            chunks.append((numero, texto[inicio:fim]))
            if fim >= len(texto):
                break
            inicio = fim - OVERLAP
    return chunks


def normalizar(vetores: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(vetores, axis=-1, keepdims=True)
    return vetores / np.clip(normas, 1e-12, None)


def buscar(pergunta: str, modelo: TextEmbedding, vetores: np.ndarray, chunks: list) -> list:
    q = normalizar(np.array(list(modelo.embed([pergunta]))))[0]
    similaridades = vetores @ q  # vetores já normalizados: produto escalar = cosseno
    melhores = np.argsort(similaridades)[::-1][:TOP_K]
    melhores = sorted(melhores)  # mantém a ordem do documento
    return [chunks[i] for i in melhores]


def main():
    if len(sys.argv) < 2:
        sys.exit("Uso: python chat_rag.py arquivo.pdf")

    paginas = ler_paginas(sys.argv[1])
    chunks = dividir_em_chunks(paginas)
    if not chunks:
        sys.exit("Não encontrei texto no PDF. Ele pode ser escaneado (só imagem).")

    print(f"{len(paginas)} páginas, {len(chunks)} trechos.")
    print("Carregando modelo de embeddings (na 1ª vez baixa algumas centenas de MB)...")
    modelo = TextEmbedding(EMBEDDING_MODEL)
    vetores = normalizar(np.array(list(modelo.embed([c[1] for c in chunks]))))

    client = anthropic.Anthropic()
    system = (
        "Você responde perguntas usando APENAS os trechos de documento fornecidos. "
        "Cite a página de onde tirou cada informação. "
        "Se a resposta não estiver nos trechos, diga que não encontrou no documento."
    )

    historico = []
    print("\nPergunte sobre o PDF (digite 'sair' para encerrar).")
    while True:
        pergunta = input("\nVocê: ").strip()
        if pergunta.lower() in {"sair", "exit"}:
            break
        if not pergunta:
            continue

        trechos = buscar(pergunta, modelo, vetores, chunks)
        contexto = "\n\n".join(f"[Página {n}]\n{t}" for n, t in trechos)
        paginas_usadas = sorted({n for n, _ in trechos})

        resposta = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=system,
            messages=historico
            + [{"role": "user", "content": f"<trechos>\n{contexto}\n</trechos>\n\nPergunta: {pergunta}"}],
        )
        texto = "".join(b.text for b in resposta.content if b.type == "text")

        # o histórico guarda só a pergunta e a resposta, não os trechos (economiza tokens)
        historico.append({"role": "user", "content": pergunta})
        historico.append({"role": "assistant", "content": texto})

        print(f"\n[trechos usados: páginas {paginas_usadas}]")
        print(f"\nClaude: {texto}")


if __name__ == "__main__":
    main()
