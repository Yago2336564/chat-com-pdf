"""
Chat com um PDF usando a API da Anthropic.

Setup:
    pip install anthropic pypdf
    export ANTHROPIC_API_KEY="sua-chave"      # Windows: set ANTHROPIC_API_KEY=sua-chave

Uso:
    python chat_com_pdf.py meu_arquivo.pdf
"""
import sys
import anthropic
from pypdf import PdfReader

MODEL = "claude-sonnet-5-5"


def ler_pdf(caminho: str) -> str:
    leitor = PdfReader(caminho)
    paginas = [f"[Página {i + 1}]\n{p.extract_text() or ''}" for i, p in enumerate(leitor.pages)]
    return "\n\n".join(paginas)


def main():
    if len(sys.argv) < 2:
        sys.exit("Uso: python chat_com_pdf.py arquivo.pdf")

    texto = ler_pdf(sys.argv[1])
    client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente

    system = (
        "Você responde perguntas usando APENAS o documento abaixo. "
        "Cite a página quando possível. Se a resposta não estiver no documento, diga que não encontrou.\n\n"
        f"<documento>\n{texto}\n</documento>"
    )

    historico = []
    print("Pergunte sobre o PDF (digite 'sair' para encerrar).")
    while True:
        pergunta = input("\nVocê: ").strip()
        if pergunta.lower() in {"sair", "exit"}:
            break
        if not pergunta:
            continue

        historico.append({"role": "user", "content": pergunta})
        resposta = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=system,
            messages=historico,
        )
        texto_resposta = "".join(b.text for b in resposta.content if b.type == "text")
        historico.append({"role": "assistant", "content": texto_resposta})
        print(f"\nClaude: {texto_resposta}")


if __name__ == "__main__":
    main()

