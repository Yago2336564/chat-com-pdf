"""
Interface web do Chat com PDF (RAG).

Rodar localmente:
    pip install streamlit
    streamlit run app.py
"""
import anthropic
import numpy as np
import streamlit as st
from fastembed import TextEmbedding

from chat_rag import (
    EMBEDDING_MODEL,
    MODEL,
    buscar,
    dividir_em_chunks,
    ler_paginas,
    normalizar,
)

MAX_PAGINAS = 150  # protege a memória do servidor gratuito

SYSTEM = (
    "Você responde perguntas usando APENAS os trechos de documento fornecidos. "
    "Cite a página de onde tirou cada informação. "
    "Se a resposta não estiver nos trechos, diga que não encontrou no documento."
)

st.set_page_config(page_title="Chat com PDF", page_icon="📄")
st.title("📄 Chat com PDF")
st.caption("Faça perguntas sobre um PDF. As respostas citam as páginas usadas.")


@st.cache_resource(show_spinner="Carregando modelo de embeddings (só na 1ª vez)...")
def carregar_modelo() -> TextEmbedding:
    return TextEmbedding(EMBEDDING_MODEL)


with st.sidebar:
    api_key = st.text_input(
        "Chave de API da Anthropic",
        type="password",
        help="Fica só na sua sessão. Crie uma em console.anthropic.com.",
    )
    st.markdown(
        "Seu PDF e sua chave não são salvos. "
        "O PDF é processado na memória durante a sessão."
    )

arquivo = st.file_uploader("Envie um PDF com texto selecionável", type="pdf")

if arquivo is None:
    st.info("Envie um PDF para começar.")
    st.stop()

# Indexa o PDF uma vez por arquivo
id_arquivo = (arquivo.name, arquivo.size)
if st.session_state.get("id_arquivo") != id_arquivo:
    paginas = ler_paginas(arquivo)
    if len(paginas) > MAX_PAGINAS:
        st.error(f"PDF com {len(paginas)} páginas. O limite é {MAX_PAGINAS}.")
        st.stop()
    chunks = dividir_em_chunks(paginas)
    if not chunks:
        st.error("Não encontrei texto no PDF. Ele pode ser escaneado (só imagem).")
        st.stop()
    modelo = carregar_modelo()
    with st.spinner(f"Indexando {len(paginas)} páginas..."):
        vetores = normalizar(np.array(list(modelo.embed([c[1] for c in chunks]))))
    st.session_state.update(
        id_arquivo=id_arquivo,
        chunks=chunks,
        vetores=vetores,
        n_paginas=len(paginas),
        mensagens=[],
    )

st.success(
    f"{st.session_state.n_paginas} páginas e {len(st.session_state.chunks)} trechos indexados."
)

# Histórico da conversa
for m in st.session_state.mensagens:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("paginas"):
            st.caption(f"Páginas usadas: {m['paginas']}")

pergunta = st.chat_input("Pergunte sobre o PDF")
if pergunta:
    if not api_key:
        st.warning("Cole sua chave de API da Anthropic na barra lateral.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(pergunta)

    modelo = carregar_modelo()
    trechos = buscar(pergunta, modelo, st.session_state.vetores, st.session_state.chunks)
    contexto = "\n\n".join(f"[Página {n}]\n{t}" for n, t in trechos)
    paginas_usadas = sorted({n for n, _ in trechos})

    historico = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.mensagens
    ]
    historico.append(
        {"role": "user", "content": f"<trechos>\n{contexto}\n</trechos>\n\nPergunta: {pergunta}"}
    )

    with st.chat_message("assistant"):
        try:
            client = anthropic.Anthropic(api_key=api_key)
            with st.spinner("Pensando..."):
                resposta = client.messages.create(
                    model=MODEL, max_tokens=4000, system=SYSTEM, messages=historico
                )
            texto = "".join(b.text for b in resposta.content if b.type == "text")
        except anthropic.AuthenticationError:
            st.error("Chave de API inválida. Confira a chave na barra lateral.")
            st.stop()
        except anthropic.APIError as e:
            st.error(f"Erro na API: {e}")
            st.stop()
        st.markdown(texto)
        st.caption(f"Páginas usadas: {paginas_usadas}")

    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    st.session_state.mensagens.append(
        {"role": "assistant", "content": texto, "paginas": paginas_usadas}
    )
