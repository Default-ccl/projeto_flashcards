import os
import re
import sys
import time
import threading
import traceback
import subprocess
import requests
from dotenv import load_dotenv
import customtkinter as ctk
from google import genai
from fpdf import FPDF
import unicodedata

# ---------------------------------------------------------------------------
# 1. CONFIGURAÇÕES DE AMBIENTE
# ---------------------------------------------------------------------------
load_dotenv()

CHAVE_API       = os.getenv("GEMINI_API_KEY")
PASTA_AULAS     = os.getenv("PASTA_AULAS",   "./aulas")
PASTA_SUPORTE   = os.getenv("PASTA_SUPORTE", "./livros_e_fontes")
PASTA_DB        = os.getenv("PASTA_DB",      "./chroma_db")
ANKI_URL        = os.getenv("ANKI_URL",      "http://localhost:8765")
ANKI_TIMEOUT    = int(os.getenv("ANKI_TIMEOUT",    "30"))
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE",      "1000"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP",   "150"))
SEARCH_K_AULA   = int(os.getenv("SEARCH_K_AULA",   "40"))
SEARCH_K_SUP    = int(os.getenv("SEARCH_K_SUP",    "15"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "1.35"))
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "models/gemini-flash-lite-latest")
EMBED_MODEL     = os.getenv("EMBED_MODEL",  "all-MiniLM-L6-v2")

QTD_MIN        = 1
QTD_MAX        = 200
LOG_MAX_LINHAS = 500
CONTEXTO_MINIMO = 500

# ---------------------------------------------------------------------------
# 2. PALETA — TECH BANNER BLUE
# ---------------------------------------------------------------------------
BG_VOID        = "#04080F"
BG_PANEL       = "#080F20"
BG_CARD        = "#0C1628"
BG_CARD_HOVER  = "#111E38"

BLUE_DEEP      = "#0D2580"
BLUE_MID       = "#1A4FD8"
BLUE_BRIGHT    = "#2563EB"
BLUE_GLOW      = "#3B82F6"
BLUE_LIGHT     = "#60A5FA"

BORDER_DIM     = "#1A3A6B"
BORDER_MID     = "#2553A8"
BORDER_BRIGHT  = "#3B82F6"

TXT_PRIMARY    = "#E0EAFF"
TXT_SECONDARY  = "#7BA3D4"
TXT_MUTED      = "#3A5A8A"
TXT_B          = "#CBD5E1"
TXT_M          = "#64748B"

RED_DIM        = "#EF4444"
RED_DARK       = "#3A0000"

SUCCESS        = "#22C55E"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------------------------------------------------------------------------
# 3. PADRÕES E HELPERS
# ---------------------------------------------------------------------------
_SPOILER_PATTERN = re.compile(
    r"\b(certo|errado|correto|incorreto|verdadeiro|falso|true|false)\b",
    re.IGNORECASE,
)
# Atualizado para incluir os níveis de Bloom
_PREFIXO_PATTERN = re.compile(
    r"^(Afirma[cç][aã]o|Resposta|Pergunta|Frente|Verso|P|R|Q|NÍVEL|LEMBRAR|ENTENDER|APLICAR|ANALISAR)[:\-\s]+",
    re.IGNORECASE,
)
_CTRL_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\ufeff]"
)

def _sanitizar(texto: str) -> str:
    texto = _CTRL_CHARS.sub(" ", str(texto))
    return " ".join(texto.split())

def _normalizar_nome(nome: str) -> str:
    nome = nome.lower()
    nome = unicodedata.normalize('NFKD', nome).encode('ASCII', 'ignore').decode('ASCII')
    nome = re.sub(r'\.pdf$', '', nome)
    nome = re.sub(r'\s+', ' ', nome).strip()
    return nome

def auditar_linha(linha: str):
    if ";" not in linha:
        motivo = "linha vazia" if not linha.strip() else "sem separador ';'"
        return None, None, motivo
    frente, verso = [x.strip() for x in linha.split(";", 1)]
    frente = _sanitizar(_PREFIXO_PATTERN.sub("", frente))
    verso  = _sanitizar(_PREFIXO_PATTERN.sub("", verso))
    if not frente or not verso:
        return None, None, "frente ou verso vazio após limpeza"
    # Remove qualquer indicação de nível entre colchetes (ex: [LEMBRAR], [NÍVEL 1], etc.)
    frente = re.sub(r"^\[.*?\]\s*", "", frente, flags=re.IGNORECASE)
    frente = _sanitizar(frente)
    if _SPOILER_PATTERN.search(frente):
        trecho = frente[:40] + ("..." if len(frente) > 40 else "")
        return None, None, f"spoiler na frente: '{trecho}'"
    if len(verso) < 15:
        return None, None, "verso muito curto (< 15 chars)"
    return frente, verso, None

def _validar_qtd(valor: str) -> int:
    if not valor.strip().isdigit():
        raise ValueError("Valor não numérico.")
    qtd = int(valor.strip())
    if not QTD_MIN <= qtd <= QTD_MAX:
        raise ValueError(f"Deve estar entre {QTD_MIN} e {QTD_MAX}.")
    return qtd

# ============================================================
# PROMPT PARA FLASHCARDS COM TAXONOMIA DE BLOOM + EXEMPLO FEYNMAN
# FOCO: TÓPICOS MAIS RELEVANTES PARA CONCURSOS PÚBLICOS E ESTUDO ACADÊMICO
# ============================================================
def montar_prompt_flashcards(qtd_alvo: int, ctx_aula: str, ctx_sup: str) -> str:
    # Distribui a quantidade entre os 4 níveis de Bloom
    niveis = ["LEMBRAR", "ENTENDER", "APLICAR", "ANALISAR"]
    qtd_por_nivel = qtd_alvo // 4
    distribuicao = {n: qtd_por_nivel for n in niveis}
    resto = qtd_alvo - (qtd_por_nivel * 4)
    for i in range(resto):
        distribuicao[niveis[i]] += 1

    return (
        f"Gere flashcards técnicos no formato PERGUNTA E RESPOSTA, distribuídos pelos 4 primeiros níveis da Taxonomia de Bloom:\n"
        f"- LEMBRAR: {distribuicao['LEMBRAR']} flashcards (fatos, definições, fórmulas)\n"
        f"- ENTENDER: {distribuicao['ENTENDER']} flashcards (explicação com suas próprias palavras)\n"
        f"- APLICAR: {distribuicao['APLICAR']} flashcards (casos práticos, uso da teoria)\n"
        f"- ANALISAR: {distribuicao['ANALISAR']} flashcards (comparações, contrastes, relações)\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "PRIORIZAÇÃO DE CONTEÚDO — REGRA MAIS IMPORTANTE:\n"
        "═══════════════════════════════════════════════════════════════\n"
        "Antes de gerar os flashcards, ANALISE o conteúdo da AULA e SELECIONE APENAS OS TÓPICOS MAIS RELEVANTES, seguindo estes critérios:\n"
        "1. **Alta incidência em concursos públicos**: priorize definições, classificações, exceções, prazos, princípios, competências, requisitos legais, fórmulas essenciais, nomenclaturas técnicas e conceitos que costumam ser cobrados em provas objetivas.\n"
        "2. **Relevância acadêmica**: priorize os conceitos estruturantes da disciplina, aqueles que servem de base para outros tópicos e que aparecem em provas, trabalhos e na prática profissional.\n"
        "3. **Descarte o irrelevante**: IGNORE exemplos meramente ilustrativos, digressões do professor, histórias paralelas, comentários informais, opiniões pessoais, repetições e qualquer conteúdo que não agregue valor direto ao aprendizado técnico ou à preparação para provas.\n"
        "4. **Foco no que cai em prova**: se o conteúdo mencionar números, prazos, artigos, leis, fórmulas, classificações ou listas, essas informações TÊM PRIORIDADE ABSOLUTA.\n"
        "5. **Densidade informativa**: cada flashcard deve carregar informação útil para prova ou estudo, nunca sendo genérico ou vago.\n\n"
        "FORMATO OBRIGATÓRIO PARA CADA FLASHCARD:\n"
        "[NÍVEL X] Pergunta aqui ; Resposta curta e direta aqui\n\n"
        "REGRAS CRÍTICAS — CUMPRIR TODAS:\n"
        "1. APENAS respostas. Zero texto introdutório ou conclusivo. Comece direto com os flashcards.\n"
        "2. Cada linha = UM flashcard. Use EXATAMENTE um ponto e vírgula (;) para separar pergunta da resposta.\n"
        "3. Inicie cada flashcard com [LEMBRAR], [ENTENDER], [APLICAR] ou [ANALISAR] conforme o nível.\n"
        "4. A pergunta deve ser formulada de acordo com o nível cognitivo:\n"
        "   - LEMBRAR: pergunta direta sobre fato, definição, fórmula (ex: 'Qual a fórmula da área do triângulo?')\n"
        "   - ENTENDER: peça para explicar com suas próprias palavras, justificar (ex: 'Explique por que a área do triângulo é base x altura / 2')\n"
        "   - APLICAR: problema prático, uso da teoria (ex: 'Calcule a quantidade de piso para uma sala triangular de 4m x 3m')\n"
        "   - ANALISAR: compare, diferencie, relacione (ex: 'Diferença entre calcular área de triângulo equilátero e retângulo')\n"
        "5. A resposta deve ser OBJETIVA, DIDÁTICA e DIRETA (máximo 2-3 frases). Evite longas explicações.\n"
        "6. A pergunta NUNCA contém as palavras: certo, errado, correto, incorreto, verdadeiro, falso, true, false.\n"
        "7. A resposta deve ter MÍNIMO 15 caracteres, baseada exclusivamente no conteúdo fornecido.\n"
        "8. Não repetir flashcards. Cada um deve ser único e bem fundamentado, sempre priorizando os tópicos mais relevantes listados acima.\n"
        "9. CRITÉRIO DE FIDELIDADE ABSOLUTA: Todas as perguntas e respostas DEVEM estar estritamente baseadas no conteúdo da AULA fornecida abaixo. "
        "Não invente fatos, não use conhecimento externo. Se algo não estiver explícito no texto da aula, não o inclua.\n"
        "10. Se o conteúdo da aula for insuficiente para gerar a quantidade solicitada, gere apenas o máximo possível — SEMPRE priorizando qualidade sobre quantidade.\n"
        "11. **EXIGÊNCIA FUNDAMENTAL**: CADA RESPOSTA DEVE CONTER, OBRIGATORIAMENTE, UM EXEMPLO PRÁTICO OU ANALÓGICO, MAS DE FORMA SIMPLES E ACESSÍVEL (técnica de Feynman). "
        "Isso significa que o exemplo deve ser explicado com linguagem coloquial, sem usar termos técnicos, como se estivesse ensinando uma criança de 10 anos. "
        "O exemplo deve vir APÓS a explicação técnica (Bloom) e ser precedido apenas por 'Exemplo:' ou 'Por exemplo:'. "
        "A ausência desse exemplo invalida completamente o flashcard. Portanto, NUNCA gere uma resposta sem o exemplo.\n"
        "12. A parte técnica da resposta (antes do exemplo) deve usar a terminologia adequada e ser concisa, conforme o nível de Bloom. "
        "O exemplo é um complemento para facilitar a compreensão, não substitui a explicação técnica, mas deve ser claramente identificado com 'Exemplo:' ou 'Por exemplo:'.\n"
        "13. **CHECKLIST PARA CADA FLASHCARD:**\n"
        "    ✓ Trata de um tópico relevante para concursos/academia (não é irrelevante).\n"
        "    ✓ Começa com [NÍVEL] correto.\n"
        "    ✓ Pergunta formulada conforme o nível.\n"
        "    ✓ Resposta técnica curta e direta (Bloom).\n"
        "    ✓ **Exemplo obrigatório** no final, com 'Exemplo:' ou 'Por exemplo:'.\n"
        "    ✓ O exemplo não contém jargões técnicos — apenas analogias do cotidiano.\n\n"
        "EXEMPLOS DE FORMATO CORRETO (sem rótulo 'Feynman', apenas 'Exemplo:'):\n"
        "[LEMBRAR] Qual é a fórmula da área do triângulo? ; A área do triângulo é calculada por A = (base × altura) / 2. Exemplo: imagine que você tem um retângulo de 4 metros de comprimento por 3 de largura; se cortar esse retângulo na diagonal, fica com dois triângulos iguais, cada um com área de 6 m².\n"
        "[ENTENDER] Explique com suas palavras por que dividimos a base vezes a altura por dois. ; A divisão por dois ocorre porque o triângulo equivale exatamente à metade de um retângulo ou paralelogramo com a mesma base e altura. Exemplo: pegue uma folha retangular de papel, desenhe uma diagonal e recorte os dois triângulos; você verá que cada um ocupa metade da folha.\n"
        "[APLICAR] Como calcular a quantidade de piso para uma sala triangular de 4m de base e 3m de altura? ; Basta aplicar a fórmula: (4×3)/2 = 6 m² de piso. Exemplo: se você for comprar piso para essa sala, peça 6 metros quadrados — é o mesmo que cobrir um retângulo de 4 por 1,5 metros.\n"
        "[ANALISAR] Qual a diferença crucial entre calcular a área de um triângulo equilátero e um retângulo? ; No retângulo, base e altura já são dados diretos; no triângulo equilátero, é necessário calcular a altura por Pitágoras antes de aplicar a fórmula geral. Exemplo: imagine um terreno retangular de 4×3, a área é 12 direto; já num terreno triangular de lados iguais, você precisa descobrir a altura com uma régua imaginária antes de multiplicar.\n\n"
        "LEMBRE-SE: (1) FOQUE NOS TÓPICOS MAIS COBRADOS E RELEVANTES; (2) O EXEMPLO É TÃO IMPORTANTE QUANTO A DEFINIÇÃO TÉCNICA — NÃO O PULE E NUNCA USE TERMOS TÉCNICOS NELE.\n\n"
        f"CONTEÚDO DA AULA (ÂNCORA - obrigatório):\n{ctx_aula[:15000]}\n\n"
        f"CONTEÚDO DE SUPORTE (COMPLEMENTAR - use apenas para enriquecer o que já está na aula):\n{ctx_sup[:5000]}\n"
    )

def montar_prompt_questoes(qtd: int, ctx_aula: str, ctx_sup: str) -> str:
    return (
        f"Gere EXATAMENTE {qtd} questões de múltipla escolha (com 5 alternativas cada) "
        "com base no conteúdo da AULA (principal) e, se relevante, complemente com o material de suporte. "
        "As questões devem avaliar o entendimento dos conceitos principais.\n\n"
        "Formato obrigatório para CADA questão:\n"
        "QUESTÃO X: [texto da pergunta]\n"
        "a) [primeira alternativa]\n"
        "b) [segunda alternativa]\n"
        "c) [terceira alternativa]\n"
        "d) [quarta alternativa]\n"
        "e) [quinta alternativa]\n\n"
        "Após TODAS as questões, adicione uma seção chamada '--- GABARITO ---' e, para cada questão, "
        "indique a letra correta e uma explicação detalhada do porquê aquela é a resposta correta, "
        "baseando-se no conteúdo fornecido.\n\n"
        "Exemplo de seção de gabarito:\n"
        "--- GABARITO ---\n"
        "1. Letra C - Explicação: ...\n"
        "2. Letra A - Explicação: ...\n\n"
        "IMPORTANTE: As alternativas devem ser apresentadas em linhas separadas. O gabarito deve vir apenas no final.\n\n"
        f"CONTEÚDO DA AULA (ÂNCORA):\n{ctx_aula[:15000]}\n\n"
        f"CONTEÚDO DE SUPORTE (COMPLEMENTAR):\n{ctx_sup[:5000]}\n"
    )

# ============================================================
# PROMPT PARA RESUMOS COM TAXONOMIA DE BLOOM (3 SEÇÕES)
# ============================================================
def montar_prompt_resumo(topico: str, ctx_aula: str, ctx_sup: str) -> str:
    return (
        f"Elabore um resumo técnico sobre '{topico}' seguindo a estrutura da Taxonomia de Bloom, dividido em três seções:\n\n"
        "--- SEÇÃO 1: BASE (O quê?) — Lembrar e Entender ---\n"
        "- Escreva uma síntese de 2 parágrafos explicando o assunto como se estivesse ensinando uma criança de 10 anos (Técnica Feynman).\n"
        "- Inclua os conceitos-chave, fórmulas, definições e regras fundamentais.\n\n"
        "--- SEÇÃO 2: INTERMEDIÁRIA (Como e Onde?) — Aplicar e Analisar ---\n"
        "- Crie uma tabela comparativa (se houver dois conceitos/teorias) apontando semelhanças e diferenças.\n"
        "- Adicione pelo menos 2 exemplos práticos de onde esse conhecimento é usado no dia a dia ou em problemas reais.\n\n"
        "--- SEÇÃO 3: AVANÇADA (E se?) — Avaliar e Criar ---\n"
        "- Responda a perguntas críticas como:\n"
        "   * Quais são os pontos fracos ou limitações dessa teoria/método?\n"
        "   * Como esse assunto se conecta com outros tópicos estudados?\n"
        "- Crie uma questão inédita de prova sobre esse assunto e escreva a resposta ideal.\n\n"
        "REQUISITOS DE QUALIDADE:\n"
        "- Seja DIDÁTICO e ESTRUTURADO, com títulos e subtítulos claros.\n"
        "- Use listas numeradas ou marcadores quando apropriado.\n"
        "- Mantenha fidelidade ao conteúdo da aula, sem inventar informações.\n"
        "- O resumo deve ser completo (mínimo 1500 palavras) e cobrir todos os níveis.\n\n"
        f"CONTEÚDO DA AULA (PRINCIPAL):\n{ctx_aula[:20000]}\n\n"
        f"CONTEÚDO DE SUPORTE (COMPLEMENTAR):\n{ctx_sup[:10000]}\n"
    )

# ---------------------------------------------------------------------------
# 4. SERVIÇOS
# ---------------------------------------------------------------------------
class VectorStoreService:
    def __init__(self):
        self._embeddings = None
        self._store      = None
        self._lock       = threading.Lock()
        self._cache_pdfs = None

    def _get_store(self):
        with self._lock:
            if self._store is None:
                from langchain_chroma import Chroma
                from langchain_huggingface import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                self._store = Chroma(persist_directory=PASTA_DB, embedding_function=self._embeddings)
        return self._store

    def indexar_pasta(self, pasta: str, folder_tag: str, log_fn) -> int:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        store    = self._get_store()
        dados    = store.get()
        indexados = {m.get("source_file") for m in dados["metadatas"] if m} if dados else set()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        count = 0
        os.makedirs(pasta, exist_ok=True)
        for nome in os.listdir(pasta):
            if not nome.endswith(".pdf") or nome in indexados:
                continue
            log_fn(f"[INDEX] {nome}")
            try:
                docs = splitter.split_documents(
                    PyPDFLoader(os.path.join(pasta, nome)).load()
                )
                for doc in docs:
                    doc.metadata["source_file"] = nome
                    doc.metadata["folder"]      = folder_tag

                batch_size = 1000
                total_docs = len(docs)
                for i in range(0, total_docs, batch_size):
                    batch = docs[i:i+batch_size]
                    store.add_documents(batch)
                    log_fn(f"[PROGRESSO] {nome}: {min(i+batch_size, total_docs)}/{total_docs} chunks indexados")
                count += 1
            except Exception as e:
                log_fn(f"[ERRO] {nome}:\n{traceback.format_exc()}")
        self._cache_pdfs = None
        return count

    def buscar_contexto_aula(self, pdf_nome: str) -> str:
        store = self._get_store()
        termo = re.sub(r"\.pdf$", "", pdf_nome, flags=re.IGNORECASE)
        docs = store.similarity_search(
            termo, k=SEARCH_K_AULA, filter={"source_file": pdf_nome}
        )
        return "\n".join(d.page_content for d in docs)

    def _carregar_todos_pdfs_cache(self):
        if self._cache_pdfs is not None:
            return self._cache_pdfs
        store = self._get_store()
        try:
            todos = store.get(include=["documents", "metadatas"])
            if not todos or not todos["documents"]:
                self._cache_pdfs = {}
                return {}
            pdf_conteudos = {}
            for doc, meta in zip(todos["documents"], todos["metadatas"]):
                if not meta:
                    continue
                nome_original = meta.get("source_file", "")
                if not nome_original:
                    continue
                key = _normalizar_nome(nome_original)
                if key not in pdf_conteudos:
                    pdf_conteudos[key] = {"nome_original": nome_original, "chunks": []}
                pdf_conteudos[key]["chunks"].append(doc)
            self._cache_pdfs = pdf_conteudos
            return self._cache_pdfs
        except Exception:
            return {}

    def extrair_texto_completo_pdf(self, pdf_nome: str) -> str:
        cache     = self._carregar_todos_pdfs_cache()
        nome_norm = _normalizar_nome(pdf_nome)
        for key, data in cache.items():
            if key == nome_norm:
                return "\n".join(data["chunks"])
        for key, data in cache.items():
            if nome_norm in key or key in nome_norm:
                return "\n".join(data["chunks"])
        return ""

    def listar_pdfs_indexados(self) -> list:
        cache = self._carregar_todos_pdfs_cache()
        return [data["nome_original"] for data in cache.values()]

    def buscar_contexto_suporte(self, termo: str, log_fn):
        store     = self._get_store()
        resultados = store.similarity_search_with_score(
            termo, k=SEARCH_K_SUP, filter={"folder": "suporte"}
        )
        ctx, fontes = "", set()
        for doc, score in resultados:
            if score < SCORE_THRESHOLD:
                ctx += doc.page_content + "\n"
                fontes.add(doc.metadata.get("source_file", "?"))
        if fontes:
            log_fn(f"[SUP] {', '.join(fontes)} ({len(resultados)} chunks)")
        else:
            log_fn("[SUP] Nenhum trecho relevante nos livros de suporte.")
        return ctx, fontes

    def buscar_contexto_por_topico(self, termo: str, k: int = 30) -> str:
        store = self._get_store()
        return "\n".join(d.page_content for d in store.similarity_search(termo, k=k))

    def aquecimento(self, log_fn):
        log_fn("[SYS] Inicializando motor de embeddings...")
        self._get_store()
        log_fn("[SYS] Motor pronto. Sistema operacional.")


class AnkiService:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback

    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def request(self, action: str, **params):
        try:
            resp = requests.post(
                ANKI_URL,
                json={"action": action, "version": 6, "params": params},
                timeout=ANKI_TIMEOUT,
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            return {"error": f"ConnectionError: {e}"}
        except requests.exceptions.Timeout:
            return {"error": "Timeout"}
        except Exception as e:
            return {"error": str(e)}

    def is_online(self, retries=3, delay=0.5) -> bool:
        for attempt in range(retries):
            result = self.request("version")
            if "error" not in result or result.get("error") is None:
                return True
            self._log(f"[DEBUG] Tentativa {attempt+1}: {result.get('error')}")
            time.sleep(delay)
        return False

    @staticmethod
    def launch_anki() -> bool:
        candidates = []
        if sys.platform == "win32":
            candidates = [
                r"C:\Program Files\Anki\anki.exe",
                r"C:\Program Files (x86)\Anki\anki.exe",
                os.path.expanduser(r"~\AppData\Local\Programs\Anki\anki.exe"),
            ]
        elif sys.platform == "darwin":
            candidates = ["/Applications/Anki.app/Contents/MacOS/anki"]
        else:
            candidates = ["anki", "/usr/bin/anki", "/usr/local/bin/anki"]

        for candidate in candidates:
            try:
                subprocess.Popen([candidate], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (FileNotFoundError, PermissionError, OSError):
                continue
        return False

    def garantir_deck(self, deck_path: str):
        return self.request("createDeck", deck=deck_path)

    def modelo_e_campos(self):
        modelos = self.request("modelNames").get("result", [])
        if not modelos:
            raise RuntimeError("AnkiConnect sem resposta.")
        modelo = next((m for m in modelos if m.lower().startswith("bas")), modelos[0])
        campos = self.request("modelFieldNames", modelName=modelo).get("result", [])
        if len(campos) < 2:
            raise RuntimeError(f"Modelo '{modelo}' precisa de >= 2 campos.")
        return modelo, campos

    def adicionar_nota(self, deck: str, modelo: str, campos, frente: str, verso: str):
        return self.request(
            "addNote",
            note={
                "deckName":  str(deck),
                "modelName": str(modelo),
                "fields": {str(campos[0]): str(frente), str(campos[1]): str(verso)},
            },
        )


# ---------------------------------------------------------------------------
# 5. WIDGETS — TAMANHOS REDUZIDOS E HARMÔNICOS
# ---------------------------------------------------------------------------
_FONT_MONO = "Consolas"

def _mono(size=11, weight="normal"):
    return ctk.CTkFont(family=_FONT_MONO, size=size, weight=weight)

def _sans(size=11, weight="normal"):
    return ctk.CTkFont(family="Segoe UI", size=size, weight=weight)


class PrimaryButton(ctk.CTkButton):
    def __init__(self, master, **kw):
        kw.setdefault("font",         _mono(12, "bold"))
        kw.setdefault("fg_color",     BLUE_BRIGHT)
        kw.setdefault("hover_color",  BLUE_MID)
        kw.setdefault("text_color",   "#FFFFFF")
        kw.setdefault("corner_radius", 6)
        kw.setdefault("height",        32)
        super().__init__(master, **kw)


class SecondaryButton(ctk.CTkButton):
    def __init__(self, master, **kw):
        kw.setdefault("font",         _mono(11, "bold"))
        kw.setdefault("fg_color",     BG_CARD)
        kw.setdefault("hover_color",  BG_CARD_HOVER)
        kw.setdefault("text_color",   BLUE_LIGHT)
        kw.setdefault("border_color", BORDER_MID)
        kw.setdefault("border_width", 1)
        kw.setdefault("corner_radius", 6)
        kw.setdefault("height",        30)
        super().__init__(master, **kw)


class DangerButton(ctk.CTkButton):
    def __init__(self, master, **kw):
        kw.setdefault("font",         _mono(12, "bold"))
        kw.setdefault("fg_color",     "#1E0808")
        kw.setdefault("hover_color",  RED_DARK)
        kw.setdefault("text_color",   RED_DIM)
        kw.setdefault("border_color", RED_DIM)
        kw.setdefault("border_width", 1)
        kw.setdefault("corner_radius", 6)
        kw.setdefault("height",        32)
        super().__init__(master, **kw)


class TechEntry(ctk.CTkEntry):
    def __init__(self, master, **kw):
        kw.setdefault("font",                   _mono(11))
        kw.setdefault("fg_color",               BG_CARD)
        kw.setdefault("text_color",             TXT_PRIMARY)
        kw.setdefault("placeholder_text_color", TXT_MUTED)
        kw.setdefault("border_color",           BORDER_DIM)
        kw.setdefault("border_width",           1)
        kw.setdefault("corner_radius",          6)
        kw.setdefault("height",                 32)
        super().__init__(master, **kw)


def section_label(parent, text: str) -> ctk.CTkLabel:
    return ctk.CTkLabel(
        parent, text=text,
        font=_mono(9, "bold"),
        text_color=TXT_MUTED,
        anchor="w",
    )


def card_frame(parent, **kw) -> ctk.CTkFrame:
    kw.setdefault("fg_color",     BG_CARD)
    kw.setdefault("border_width", 1)
    kw.setdefault("border_color", BORDER_DIM)
    kw.setdefault("corner_radius", 6)
    return ctk.CTkFrame(parent, **kw)


# ---------------------------------------------------------------------------
# 6. APLICAÇÃO PRINCIPAL
# ---------------------------------------------------------------------------
class AppAnki(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FlashCard AI  ·  v4.3 (Bloom + Feynman)")
        self.geometry("1020x750")
        self.minsize(860, 600)
        self.resizable(True, True)
        self.configure(fg_color=BG_VOID)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)

        os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

        if not CHAVE_API:
            print("[FATAL] GEMINI_API_KEY ausente no .env")
            sys.exit(1)

        self._gemini             = genai.Client(api_key=CHAVE_API)
        self._vs                 = VectorStoreService()
        self._anki               = AnkiService(log_callback=self._log_ui)
        self._todos_pdfs         = self._obter_pdfs()
        self._pronto             = False
        self._resumo_texto       = ""
        self._questoes_texto     = ""
        self.pdf_check_vars      = {}
        self._pdfs_selecionados_persistentes = set()
        self._misturar_conteudos = False
        self._cancel_event       = threading.Event()
        self._debounce_id        = None
        self._debounce_id_flash  = None
        self.pdf_selecionado_flash = None

        self._setup_ui()
        self.progressbar.start()
        threading.Thread(target=self._aquecimento, daemon=True).start()

    # ------------------------------------------------------------------ boot
    def _aquecimento(self):
        try:
            self._vs.aquecimento(self._log_ui)
            self._pronto = True
            self.after(0, self._habilitar_controles)
        except Exception:
            self._log_ui(f"[ERRO] Falha no boot:\n{traceback.format_exc()}")

    def _habilitar_controles(self):
        for btn in [
            self.btn_indexar, self.btn_gerar,
            self.btn_cancelar_flash, self.btn_gerar_resumo,
            self.btn_gerar_questoes, self.btn_cancelar_resumo,
        ]:
            btn.configure(state="normal")
        self.status_dot.configure(text_color=SUCCESS)
        self.status_label.configure(text="ONLINE", text_color=SUCCESS)
        self.progressbar.stop()
        self.progressbar.set(0)

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        self._build_header()
        ctk.CTkFrame(self, fg_color=BORDER_DIM, height=1, corner_radius=0).grid(
            row=1, column=0, sticky="ew")
        self.scroll_main = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
            scrollbar_button_color=BORDER_DIM,
            scrollbar_button_hover_color=BLUE_MID,
        )
        self.scroll_main.grid(row=2, column=0, sticky="nsew")
        self.scroll_main.grid_columnconfigure(0, weight=1)
        self._bind_mousewheel_to_scrollable(self.scroll_main)
        self._build_tabs()
        self._build_progressbar()
        self._build_log()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=52)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, padx=20, pady=0, sticky="w")

        ctk.CTkLabel(
            title_frame,
            text="◈  FLASHCARD",
            font=_mono(15, "bold"),
            text_color=BLUE_LIGHT,
        ).pack(side="left")
        ctk.CTkLabel(
            title_frame,
            text="  AI  ·  v4.3 (Bloom + Feynman)",
            font=_mono(10),
            text_color=TXT_MUTED,
        ).pack(side="left", pady=(4, 0))

        status_frame = ctk.CTkFrame(header, fg_color="transparent")
        status_frame.grid(row=0, column=2, padx=20, sticky="e")

        self.status_dot = ctk.CTkLabel(
            status_frame, text="●",
            font=_mono(12),
            text_color=TXT_MUTED,
        )
        self.status_dot.pack(side="left", padx=(0, 5))

        self.status_label = ctk.CTkLabel(
            status_frame, text="BOOTING",
            font=_mono(9, "bold"),
            text_color=TXT_MUTED,
        )
        self.status_label.pack(side="left")

    def _bind_mousewheel_to_scrollable(self, scrollable):
        def on_mousewheel(event):
            delta = 0
            if hasattr(event, 'delta') and event.delta != 0:
                delta = -1 * (event.delta // abs(event.delta))
            elif hasattr(event, 'num'):
                if event.num == 4:
                    delta = -1
                elif event.num == 5:
                    delta = 1
            if delta != 0:
                canvas = getattr(scrollable, '_parent_canvas', None)
                if canvas:
                    canvas.yview_scroll(delta, "units")
        self.bind_all("<MouseWheel>", on_mousewheel, add=True)
        self.bind_all("<Button-4>", on_mousewheel, add=True)
        self.bind_all("<Button-5>", on_mousewheel, add=True)

    def _build_tabs(self):
        self.tabview = ctk.CTkTabview(
            self.scroll_main,
            fg_color=BG_PANEL,
            segmented_button_fg_color=BG_VOID,
            segmented_button_selected_color=BLUE_BRIGHT,
            segmented_button_unselected_color=BG_PANEL,
            segmented_button_selected_hover_color=BLUE_MID,
            segmented_button_unselected_hover_color=BG_CARD_HOVER,
            text_color=TXT_PRIMARY,
            text_color_disabled=TXT_MUTED,
            corner_radius=6,
            border_width=1,
            border_color=BORDER_DIM,
        )
        self.tabview.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        self.tabview.add("  FLASHCARDS  ")
        self.tabview.add("  RESUMO & QUESTÕES  ")
        self.tabview.set("  FLASHCARDS  ")

        self._build_aba_flashcards(self.tabview.tab("  FLASHCARDS  "))
        self._build_aba_resumo(self.tabview.tab("  RESUMO & QUESTÕES  "))

    def _build_progressbar(self):
        self.progressbar = ctk.CTkProgressBar(
            self.scroll_main,
            mode="indeterminate",
            height=2,
            corner_radius=0,
            fg_color=BG_PANEL,
            progress_color=BLUE_BRIGHT,
        )
        self.progressbar.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 4))
        self.progressbar.set(0)
        self.progressbar.stop()

    def _build_log(self):
        log_header = ctk.CTkFrame(self.scroll_main, fg_color="transparent")
        log_header.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 2))
        section_label(log_header, "// CONSOLE").pack(side="left")
        SecondaryButton(
            log_header, text="Limpar log", width=80, height=26,
            command=lambda: self.txt_log.delete("1.0", "end"),
        ).pack(side="right")

        self.txt_log = ctk.CTkTextbox(
            self.scroll_main,
            corner_radius=6,
            fg_color=BG_PANEL,
            text_color=TXT_SECONDARY,
            font=(_FONT_MONO, 10),
            border_width=1,
            border_color=BORDER_DIM,
            scrollbar_button_color=BORDER_DIM,
            scrollbar_button_hover_color=BLUE_MID,
            height=120,
        )
        self.txt_log.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 12))
        self._log("Sistema iniciado. Aguardando boot...")

    # ------------------------------------------------------------------ aba flashcards
    def _build_aba_flashcards(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        PAD = dict(padx=10, pady=4)

        c_deck = card_frame(parent)
        c_deck.grid(row=0, column=0, sticky="ew", **PAD)
        c_deck.grid_columnconfigure(0, weight=1)
        section_label(c_deck, "// TARGET DECK").grid(row=0, column=0, padx=10, pady=(6, 0), sticky="w")
        self.entry_subdeck = TechEntry(c_deck, placeholder_text="ex: Engenharia_de_Requisitos", height=30)
        self.entry_subdeck.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        self.entry_subdeck.bind("<KeyRelease>", self._sincronizar_pdf_por_deck)

        c_pdf = card_frame(parent)
        c_pdf.grid(row=1, column=0, sticky="ew", **PAD)
        c_pdf.grid_columnconfigure(0, weight=1)
        c_pdf.grid_rowconfigure(2, weight=1)

        section_label(c_pdf, "// ARQUIVO PDF  (deixe em branco para usar tópico)").grid(
            row=0, column=0, padx=10, pady=(6, 0), sticky="w")

        search_flash_frame = ctk.CTkFrame(c_pdf, fg_color="transparent")
        search_flash_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))
        search_flash_frame.grid_columnconfigure(0, weight=1)

        self.entry_search_flash = TechEntry(search_flash_frame, placeholder_text="🔍  Filtrar PDFs...", height=30)
        self.entry_search_flash.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.entry_search_flash.bind("<KeyRelease>", self._filtrar_pdfs_flash_debounce)

        btn_atualizar_flash = SecondaryButton(search_flash_frame, text="↻", width=30, height=30,
                                              command=self._atualizar_lista_flash)
        btn_atualizar_flash.grid(row=0, column=1, padx=(0, 0))

        self.btn_indexar = SecondaryButton(
            search_flash_frame, text="Indexar PDFs", width=100, height=30,
            state="disabled", command=self._thread_indexar)
        self.btn_indexar.grid(row=0, column=2, padx=(4, 0))

        self.pdf_list_flash = ctk.CTkScrollableFrame(
            c_pdf, fg_color=BG_VOID,
            border_width=1, border_color=BORDER_DIM,
            scrollbar_button_color=BORDER_DIM,
            scrollbar_button_hover_color=BLUE_MID,
            height=120,
        )
        self.pdf_list_flash.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 8))
        self._bind_mousewheel_to_widget(self.pdf_list_flash)
        self._popular_lista_flash()

        c_topico = card_frame(parent)
        c_topico.grid(row=2, column=0, sticky="ew", **PAD)
        c_topico.grid_columnconfigure(0, weight=1)
        section_label(c_topico, "// TÓPICO ESPECÍFICO  (opcional — sobrepõe o PDF)").grid(
            row=0, column=0, padx=10, pady=(6, 0), sticky="w")
        self.entry_topico_flash = TechEntry(c_topico, placeholder_text="ex: redes neurais, transformadores", height=30)
        self.entry_topico_flash.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

        c_acao = card_frame(parent)
        c_acao.grid(row=3, column=0, sticky="ew", **PAD)
        c_acao.grid_columnconfigure(1, weight=1)

        section_label(c_acao, "// QUANTIDADE DE CARDS").grid(
            row=0, column=0, columnspan=3, padx=10, pady=(6, 2), sticky="w")

        self.entry_qtd = TechEntry(c_acao, width=70, justify="center", height=30)
        self.entry_qtd.insert(0, "10")
        self.entry_qtd.grid(row=1, column=0, padx=(10, 6), pady=(0, 8), sticky="w")

        self.btn_gerar = PrimaryButton(
            c_acao, text="▶  GERAR FLASHCARDS",
            state="disabled", command=self._iniciar_thread_gerar)
        self.btn_gerar.grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=(0, 8))

        self.btn_cancelar_flash = DangerButton(
            c_acao, text="✕", width=40,
            state="disabled", command=self._cancelar_operacao)
        self.btn_cancelar_flash.grid(row=1, column=2, padx=(0, 10), pady=(0, 8))

        parent.grid_rowconfigure(4, weight=1)

    # ------------------------------------------------------------------ aba resumo
    def _build_aba_resumo(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        PAD = dict(padx=10, pady=2)

        c_top = card_frame(parent)
        c_top.grid(row=0, column=0, sticky="ew", **PAD)
        c_top.grid_columnconfigure(0, weight=1)
        section_label(c_top, "// TÓPICO ESPECÍFICO  (ignora PDFs se preenchido)").grid(
            row=0, column=0, padx=10, pady=(4, 0), sticky="w")
        self.entry_topico_resumo = TechEntry(c_top, placeholder_text="ex: inteligência artificial, legislação trabalhista", height=30)
        self.entry_topico_resumo.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))

        c_pdfs = card_frame(parent)
        c_pdfs.grid(row=1, column=0, sticky="ew", **PAD)
        c_pdfs.grid_columnconfigure(0, weight=1)
        c_pdfs.grid_rowconfigure(2, weight=1)

        pdf_head = ctk.CTkFrame(c_pdfs, fg_color="transparent")
        pdf_head.grid(row=0, column=0, sticky="ew", padx=10, pady=(4, 2))
        pdf_head.grid_columnconfigure(0, weight=1)
        section_label(pdf_head, "// SELECIONAR PDFs").grid(row=0, column=0, sticky="w")

        sel_frame = ctk.CTkFrame(pdf_head, fg_color="transparent")
        sel_frame.grid(row=0, column=1, sticky="e")
        SecondaryButton(sel_frame, text="✓ Marcar todos", width=100, height=26,
                        command=self._marcar_todos_pdfs).pack(side="left", padx=(0, 4))
        SecondaryButton(sel_frame, text="✗ Desmarcar", width=90, height=26,
                        command=self._desmarcar_todos_pdfs).pack(side="left")

        self.search_entry = TechEntry(c_pdfs, placeholder_text="🔍  Filtrar PDFs por nome...", height=30)
        self.search_entry.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))
        self.search_entry.bind("<KeyRelease>", self._filtrar_pdfs_debounce)

        self.pdf_listbox = ctk.CTkScrollableFrame(
            c_pdfs, fg_color=BG_VOID,
            border_width=1, border_color=BORDER_DIM,
            scrollbar_button_color=BORDER_DIM,
            scrollbar_button_hover_color=BLUE_MID,
            height=200,
        )
        self.pdf_listbox.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 4))
        self._bind_mousewheel_to_widget(self.pdf_listbox)

        self.check_misturar_var = ctk.BooleanVar(value=False)
        self.check_misturar = ctk.CTkCheckBox(
            c_pdfs,
            text="Misturar conteúdos  —  consolida todos os PDFs num único resumo/questão",
            variable=self.check_misturar_var,
            fg_color=BLUE_BRIGHT,
            hover_color=BLUE_MID,
            border_color=BORDER_MID,
            text_color=TXT_SECONDARY,
            font=_mono(10),
            command=self._toggle_misturar_conteudos,
        )
        self.check_misturar.grid(row=3, column=0, sticky="w", padx=12, pady=(0, 6))

        c_conf = card_frame(parent)
        c_conf.grid(row=2, column=0, sticky="ew", **PAD)
        c_conf.grid_columnconfigure(1, weight=1)
        section_label(c_conf, "// CONFIGURAÇÕES").grid(
            row=0, column=0, columnspan=3, padx=10, pady=(4, 2), sticky="w")

        ctk.CTkLabel(c_conf, text="Número de questões:", font=_mono(11), text_color=TXT_PRIMARY).grid(
            row=1, column=0, padx=(10, 6), pady=(0, 6), sticky="w")
        self.spin_questoes = TechEntry(c_conf, width=70, justify="center", height=30)
        self.spin_questoes.insert(0, "5")
        self.spin_questoes.grid(row=1, column=1, sticky="w", padx=(0, 6), pady=(0, 6))

        SecondaryButton(c_conf, text="📄  Exportar PDF", width=130, height=30,
                        command=self._exportar_pdf).grid(
            row=1, column=2, sticky="e", padx=(0, 10), pady=(0, 6))

        c_acoes = card_frame(parent)
        c_acoes.grid(row=3, column=0, sticky="ew", **PAD)
        c_acoes.grid_columnconfigure((0, 1), weight=1)
        section_label(c_acoes, "// AÇÕES").grid(
            row=0, column=0, columnspan=4, padx=10, pady=(4, 2), sticky="w")

        self.btn_gerar_resumo = PrimaryButton(
            c_acoes, text="▶  Gerar Resumo",
            state="disabled", command=self._iniciar_thread_resumo)
        self.btn_gerar_resumo.grid(row=1, column=0, sticky="ew", padx=(10, 4), pady=(0, 6))

        self.btn_gerar_questoes = PrimaryButton(
            c_acoes, text="▶  Gerar Questões",
            state="disabled", command=self._iniciar_thread_questoes)
        self.btn_gerar_questoes.grid(row=1, column=1, sticky="ew", padx=(4, 4), pady=(0, 6))

        SecondaryButton(c_acoes, text="Limpar", width=70,
                        command=self._limpar_conteudo).grid(
            row=1, column=2, padx=(4, 4), pady=(0, 6))

        self.btn_cancelar_resumo = DangerButton(
            c_acoes, text="✕", width=40,
            state="disabled", command=self._cancelar_operacao)
        self.btn_cancelar_resumo.grid(row=1, column=3, padx=(0, 10), pady=(0, 6))

        section_label(parent, "// RESULTADO").grid(
            row=4, column=0, sticky="w", padx=12, pady=(2, 2))

        self.text_resumo_questoes = ctk.CTkTextbox(
            parent,
            fg_color=BG_PANEL,
            text_color=TXT_PRIMARY,
            font=(_FONT_MONO, 10),
            border_width=1,
            border_color=BORDER_DIM,
            corner_radius=6,
            scrollbar_button_color=BORDER_DIM,
            scrollbar_button_hover_color=BLUE_MID,
            height=180,
        )
        self.text_resumo_questoes.grid(row=5, column=0, sticky="nsew", padx=10, pady=(0, 8))

        self._popular_lista_pdfs()

    # ------------------------------------------------------------------ helpers UI
    def _bind_mousewheel_to_widget(self, widget):
        def on_mousewheel(event):
            x, y = event.x_root, event.y_root
            wx, wy = widget.winfo_rootx(), widget.winfo_rooty()
            if wx <= x <= wx + widget.winfo_width() and wy <= y <= wy + widget.winfo_height():
                delta = 0
                if hasattr(event, 'delta') and event.delta != 0:
                    delta = -1 * (event.delta // abs(event.delta))
                elif hasattr(event, 'num') and event.num == 4:
                    delta = -1
                elif hasattr(event, 'num') and event.num == 5:
                    delta = 1
                if delta != 0:
                    canvas = getattr(widget, '_parent_canvas', None) or getattr(widget, '_canvas', None)
                    if canvas:
                        canvas.yview_scroll(delta, "units")
        self.bind_all("<MouseWheel>", on_mousewheel, add=True)
        self.bind_all("<Button-4>",   on_mousewheel, add=True)
        self.bind_all("<Button-5>",   on_mousewheel, add=True)

    def _marcar_todos_pdfs(self):
        for pdf in self._todos_pdfs:
            self._pdfs_selecionados_persistentes.add(pdf)
        for pdf, var in self.pdf_check_vars.items():
            if pdf in self._pdfs_selecionados_persistentes:
                var.set(True)
        self._log("[SYS] Todos os PDFs selecionados.")

    def _desmarcar_todos_pdfs(self):
        self._pdfs_selecionados_persistentes.clear()
        for var in self.pdf_check_vars.values():
            var.set(False)
        self._log("[SYS] PDFs desmarcados.")

    def _toggle_misturar_conteudos(self):
        self._misturar_conteudos = self.check_misturar_var.get()
        modo = "consolidado" if self._misturar_conteudos else "separado por PDF"
        self._log(f"[SYS] Modo: {modo}")

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.txt_log.insert("end", f"[{ts}]  {msg}\n")
        self.txt_log.see("end")
        linhas = int(self.txt_log.index("end-1c").split(".")[0])
        if linhas > LOG_MAX_LINHAS:
            self.txt_log.delete("1.0", f"{linhas - LOG_MAX_LINHAS}.0")

    def _log_ui(self, msg: str):
        self.after(0, self._log, msg)

    def _set_busy(self, widget, label_busy: str, busy: bool):
        if busy:
            widget._original_text = widget.cget("text")
            widget.configure(state="disabled", text=label_busy)
            self.progressbar.start()
        else:
            original = getattr(widget, "_original_text", None)
            widget.configure(state="normal", text=original or widget.cget("text"))
            if original:
                try:
                    del widget._original_text
                except AttributeError:
                    pass
            self.progressbar.stop()
            self.progressbar.set(0)

    def _set_busy_ui(self, widget, label_busy: str, busy: bool):
        self.after(0, self._set_busy, widget, label_busy, busy)

    def _cancelar_operacao(self):
        self._cancel_event.set()
        self._log("[SYS] Cancelamento solicitado. Aguardando fim da operação atual...")

    def _sincronizar_pdf_por_deck(self, event=None):
        termo = self.entry_subdeck.get().strip().lower()
        if not termo:
            return
        for pdf in self._todos_pdfs:
            if termo in pdf.lower():
                self.pdf_selecionado_flash = pdf
                if hasattr(self, '_pdf_selecionado_var'):
                    self._pdf_selecionado_var.set(pdf)
                self._filtrar_pdfs_flash()
                break

    def _obter_pdfs(self) -> list:
        os.makedirs(PASTA_AULAS, exist_ok=True)
        return sorted(f for f in os.listdir(PASTA_AULAS) if f.endswith(".pdf"))

    def _atualizar_dropdown(self):
        self._todos_pdfs = self._obter_pdfs()
        self._popular_lista_pdfs()
        self._popular_lista_flash()
        self._log("[SYS] Listas de PDFs atualizadas.")

    # ------------------------------------------------------------------ indexação
    def _thread_indexar(self):
        self._set_busy(self.btn_indexar, "Indexando...", busy=True)
        self._cancel_event.clear()
        threading.Thread(target=self._indexar_arquivos, daemon=True).start()

    def _indexar_arquivos(self):
        try:
            self._log_ui("[SYS] Varrendo diretórios...")
            total  = self._vs.indexar_pasta(PASTA_AULAS,   "aula",    self._log_ui)
            if self._cancel_event.is_set():
                self._log_ui("[SYS] Indexação cancelada pelo usuário.")
                return
            total += self._vs.indexar_pasta(PASTA_SUPORTE, "suporte", self._log_ui)
            if self._cancel_event.is_set():
                self._log_ui("[SYS] Indexação cancelada pelo usuário.")
                return
            self._todos_pdfs = self._obter_pdfs()
            self.after(0, self._atualizar_dropdown)
            self._log_ui(f"[OK] {total} arquivo(s) indexado(s).")
        except Exception:
            self._log_ui(f"[ERR] Falha na indexação.\n{traceback.format_exc()}")
        finally:
            self._set_busy_ui(self.btn_indexar, "Indexando...", False)
            self._cancel_event.clear()

    # ------------------------------------------------------------------ flashcards (com validação de contexto)
    def _iniciar_thread_gerar(self):
        try:
            qtd_alvo = _validar_qtd(self.entry_qtd.get())
            self._log_ui(f"[INFO] Quantidade solicitada: {qtd_alvo} flashcards.")
        except ValueError as e:
            self._log(f"[ERRO] Quantidade inválida: {e}")
            return
        self._set_busy(self.btn_gerar, "  Gerando...", busy=True)
        self._cancel_event.clear()
        threading.Thread(target=self._processo_geracao, args=(qtd_alvo,), daemon=True).start()

    def _processo_geracao(self, qtd_alvo: int):
        try:
            if self._cancel_event.is_set():
                self._log_ui("[SYS] Geração cancelada pelo usuário.")
                return

            self._log_ui(f"[INFO] Verificando Anki em {ANKI_URL}...")
            if not self._anki.is_online():
                self._log_ui("[WARN] Anki offline. Tentando abrir o Anki...")
                if AnkiService.launch_anki():
                    self._log_ui("[INFO] Anki iniciado. Aguardando 5 segundos para o AnkiConnect...")
                    time.sleep(5)
                    if not self._anki.is_online():
                        self._log_ui("[ERRO] Anki não responde mesmo após iniciar. Verifique a instalação.")
                        return
                    else:
                        self._log_ui("[OK] Anki conectado com sucesso!")
                else:
                    self._log_ui("[ERRO] Não foi possível localizar o Anki. Instale o Anki e configure o caminho.")
                    return
            else:
                self._log_ui("[OK] Anki já está online.")

            sub_deck  = self.entry_subdeck.get().strip() or "Geral"
            deck_path = f"NOVOS::{sub_deck}"
            topico    = self.entry_topico_flash.get().strip()
            self._log_ui(f"[INFO] Deck destino: {deck_path}")

            if topico:
                self._log_ui(f"[RAG] Buscando contexto para o tópico: '{topico}'")
                ctx_aula = self._vs.buscar_contexto_por_topico(topico, k=SEARCH_K_AULA)
                ctx_sup, _ = self._vs.buscar_contexto_suporte(topico, self._log_ui)
                if not ctx_aula.strip() or len(ctx_aula) < CONTEXTO_MINIMO:
                    self._log_ui("[WARN] Conteúdo insuficiente para o tópico. Tente selecionar um PDF específico.")
                    return
                self._log_ui(f"[RAG] Contexto do tópico obtido (aula: {len(ctx_aula)} chars, suporte: {len(ctx_sup)} chars).")
            else:
                pdf_nome = getattr(self, 'pdf_selecionado_flash', None)
                if not pdf_nome or not pdf_nome.endswith(".pdf"):
                    self._log_ui("[WARN] Nenhum PDF selecionado.")
                    return
                self._log_ui(f"[RAG] Buscando contexto do PDF âncora: {pdf_nome}")
                ctx_aula = self._vs.buscar_contexto_aula(pdf_nome)
                if not ctx_aula.strip() or len(ctx_aula) < CONTEXTO_MINIMO:
                    self._log_ui("[ERRO] Contexto do PDF muito curto ou inexistente. Certifique-se de que o PDF foi indexado corretamente.")
                    return
                self._log_ui(f"[RAG] Contexto da âncora: {len(ctx_aula)} caracteres.")
                self._log_ui(f"[RAG] Buscando complemento nos livros de suporte...")
                ctx_sup, _ = self._vs.buscar_contexto_suporte(pdf_nome, self._log_ui)
                self._log_ui(f"[RAG] Contexto de suporte: {len(ctx_sup)} caracteres.")

            if self._cancel_event.is_set():
                self._log_ui("[SYS] Cancelamento detectado antes da chamada à API.")
                return

            self._log_ui(f"[AI] Chamando modelo {GEMINI_MODEL} para gerar flashcards (Bloom + Feynman)...")
            prompt = montar_prompt_flashcards(qtd_alvo, ctx_aula, ctx_sup)
            resp   = self._gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            if self._cancel_event.is_set():
                self._log_ui("[SYS] Cancelamento detectado após resposta da API.")
                return
            texto = getattr(resp, "text", "")
            if not texto:
                self._log_ui("[WARN] Resposta vazia ou bloqueada pelo safety filter.")
                return
            linhas = [l for l in texto.strip().split("\n") if l.strip()]
            self._log_ui(f"[AI] Resposta recebida: {len(linhas)} linhas brutas.")

            self._log_ui("[AUDIT] Enviando flashcards para o Anki...")
            self._anki.garantir_deck(deck_path)
            modelo, campos = self._anki.modelo_e_campos()
            count = self._injetar_flashcards(linhas, qtd_alvo, deck_path, modelo, campos)
            if not self._cancel_event.is_set():
                self._log_ui(f"[OK] {count} card(s) injetado(s) → {deck_path}")
                if count < qtd_alvo:
                    self._log_ui(f"[AVISO] Gerados apenas {count} flashcards (conteúdo insuficiente para {qtd_alvo}).")
        except Exception as e:
            self._log_ui(f"[ERRO] {e}")
        finally:
            self._set_busy_ui(self.btn_gerar, "  Gerando...", False)
            self._cancel_event.clear()

    def _injetar_flashcards(self, linhas, qtd_alvo, deck_path, modelo, campos):
        count = 0
        rejeitados = 0
        for linha in linhas:
            if self._cancel_event.is_set() or count >= qtd_alvo:
                break
            frente, verso, motivo = auditar_linha(linha)
            if motivo:
                self._log_ui(f"[SKIP] {motivo} → {linha[:60]}...")
                rejeitados += 1
                continue
            res = self._anki.adicionar_nota(deck_path, modelo, campos, frente, verso)
            if res.get("error"):
                if "duplicate" in str(res["error"]).lower():
                    self._log_ui(f"[DUP] Flashcard duplicado: {frente[:50]}...")
                else:
                    self._log_ui(f"[ERR] Anki: {res['error']}")
            else:
                count += 1
        if rejeitados:
            self._log_ui(f"[AUDIT] {rejeitados} linha(s) rejeitadas por auditoria.")
        return count

    # ------------------------------------------------------------------ lista PDFs (aba resumo)
    def _popular_lista_pdfs(self):
        self._todos_pdfs = self._obter_pdfs()
        self.search_entry.delete(0, "end")
        self._filtrar_pdfs()

    def _filtrar_pdfs_debounce(self, event=None):
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(250, self._filtrar_pdfs)

    def _filtrar_pdfs(self, event=None):
        termo = self.search_entry.get().strip().lower()
        pdfs  = [p for p in self._todos_pdfs if termo in p.lower()] if termo else list(self._todos_pdfs)

        for w in self.pdf_listbox.winfo_children():
            w.destroy()
        self.pdf_check_vars.clear()

        if not pdfs:
            ctk.CTkLabel(self.pdf_listbox, text="Nenhum PDF encontrado.",
                         text_color=RED_DIM, font=_mono(10)).pack(padx=10, pady=10)
            return

        for pdf in pdfs:
            is_selected = pdf in self._pdfs_selecionados_persistentes
            var = ctk.BooleanVar(value=is_selected)

            def on_check(pdf=pdf, var=var):
                if var.get():
                    self._pdfs_selecionados_persistentes.add(pdf)
                else:
                    self._pdfs_selecionados_persistentes.discard(pdf)

            chk = ctk.CTkCheckBox(
                self.pdf_listbox, text=pdf, variable=var,
                fg_color=BLUE_BRIGHT, hover_color=BLUE_MID,
                border_color=BORDER_DIM,
                text_color=TXT_PRIMARY, font=_mono(10),
                command=on_check,
            )
            chk.pack(anchor="w", padx=10, pady=2)
            self.pdf_check_vars[pdf] = var

    def _obter_pdfs_selecionados(self) -> list:
        return list(self._pdfs_selecionados_persistentes)

    def _extrair_texto_completo_pdfs(self, pdfs_nomes: list) -> dict:
        resultados, cache = {}, self._vs._carregar_todos_pdfs_cache()
        if not cache:
            return resultados
        for pdf in pdfs_nomes:
            if self._cancel_event.is_set():
                break
            nome_norm = _normalizar_nome(pdf)
            for key, data in cache.items():
                if key == nome_norm or nome_norm in key or key in nome_norm:
                    resultados[data["nome_original"]] = "\n".join(data["chunks"])
                    self._log_ui(f"[RAG] Extraído {len(data['chunks'])} chunks de {data['nome_original']}")
                    break
            else:
                self._log_ui(f"[AVISO] PDF não encontrado no índice: {pdf}")
        return resultados

    # ------------------------------------------------------------------ listas da aba flashcards
    def _popular_lista_flash(self):
        self._todos_pdfs = self._obter_pdfs()
        if hasattr(self, 'entry_search_flash'):
            self.entry_search_flash.delete(0, "end")
        self._filtrar_pdfs_flash()

    def _filtrar_pdfs_flash(self, event=None):
        termo = self.entry_search_flash.get().strip().lower() if hasattr(self, 'entry_search_flash') else ""
        pdfs = [p for p in self._todos_pdfs if termo in p.lower()] if termo else list(self._todos_pdfs)

        for w in self.pdf_list_flash.winfo_children():
            w.destroy()

        if not pdfs:
            ctk.CTkLabel(self.pdf_list_flash, text="Nenhum PDF encontrado.",
                         text_color=RED_DIM, font=_mono(10)).pack(padx=10, pady=10)
            return

        self._pdf_selecionado_var = ctk.StringVar(value="")
        for pdf in pdfs:
            rb = ctk.CTkRadioButton(
                self.pdf_list_flash, text=pdf, variable=self._pdf_selecionado_var,
                value=pdf,
                fg_color=BLUE_BRIGHT, hover_color=BLUE_MID,
                border_color=BORDER_DIM,
                text_color=TXT_PRIMARY, font=_mono(10),
            )
            rb.pack(anchor="w", padx=10, pady=2)
            if hasattr(self, 'pdf_selecionado_flash') and self.pdf_selecionado_flash == pdf:
                rb.select()
            def on_select(pdf=pdf):
                self.pdf_selecionado_flash = pdf
            rb.configure(command=on_select)

        if not self._pdf_selecionado_var.get() and pdfs:
            self._pdf_selecionado_var.set(pdfs[0])
            self.pdf_selecionado_flash = pdfs[0]

    def _filtrar_pdfs_flash_debounce(self, event=None):
        if hasattr(self, '_debounce_id_flash') and self._debounce_id_flash:
            self.after_cancel(self._debounce_id_flash)
        self._debounce_id_flash = self.after(250, self._filtrar_pdfs_flash)

    def _atualizar_lista_flash(self):
        self._popular_lista_flash()
        self._log("[SYS] Lista de PDFs (Flashcards) atualizada.")

    # ------------------------------------------------------------------ resumo
    def _iniciar_thread_resumo(self):
        topico      = self.entry_topico_resumo.get().strip()
        selecionados = self._obter_pdfs_selecionados()
        if not topico and not selecionados:
            self._log("[ERRO] Forneça tópico ou selecione PDFs.")
            return
        self._set_busy(self.btn_gerar_resumo, "  Gerando...", busy=True)
        self._cancel_event.clear()
        threading.Thread(target=self._gerar_resumo_thread, args=(topico, selecionados), daemon=True).start()

    def _gerar_resumo_thread(self, topico, pdfs):
        try:
            if self._cancel_event.is_set():
                self._log_ui("[SYS] Geração de resumo cancelada.")
                return
            if topico:
                self._log_ui(f"[RAG] Buscando contexto para o tópico: '{topico}'")
                ctx_aula = self._vs.buscar_contexto_por_topico(topico, k=50)
                ctx_sup, _ = self._vs.buscar_contexto_suporte(topico, self._log_ui)
                if not ctx_aula:
                    self._log_ui("[ERRO] Nenhum conteúdo para o tópico.")
                    return
                self._log_ui(f"[RAG] Contexto obtido: {len(ctx_aula)} caracteres.")
                prompt = montar_prompt_resumo(topico, ctx_aula, ctx_sup)
                self._log_ui("[AI] Solicitando resumo estruturado (Bloom) ao Gemini...")
                resp   = self._gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                if self._cancel_event.is_set():
                    self._log_ui("[SYS] Cancelamento detectado após resposta da API.")
                    return
                self._resumo_texto = getattr(resp, "text", "").strip()
                self._log_ui(f"[OK] Resumo gerado com {len(self._resumo_texto)} caracteres.")
            else:
                self._log_ui(f"[RAG] Extraindo conteúdo de {len(pdfs)} PDF(s)...")
                conteudos = self._extrair_texto_completo_pdfs(pdfs)
                if not conteudos:
                    self._log_ui("[ERRO] Nenhum conteúdo dos PDFs.")
                    return
                if self._cancel_event.is_set():
                    return
                if not self._misturar_conteudos:
                    partes = []
                    for nome, texto in conteudos.items():
                        if self._cancel_event.is_set():
                            break
                        self._log_ui(f"[AI] Gerando resumo para: {nome}")
                        ctx_sup, _ = self._vs.buscar_contexto_suporte(nome, self._log_ui)
                        prompt = montar_prompt_resumo(nome, texto[:20000], ctx_sup)
                        resp = self._gemini.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=prompt)
                        if self._cancel_event.is_set():
                            break
                        partes.append(f"--- RESUMO: {nome} ---\n{getattr(resp,'text','')}")
                    self._resumo_texto = "\n\n".join(partes)
                    self._log_ui(f"[OK] {len(partes)} resumo(s) separados gerados.")
                else:
                    total = "\n\n".join(conteudos.values())
                    ctx_sup, _ = self._vs.buscar_contexto_suporte("geral", self._log_ui)
                    self._log_ui(f"[RAG] Conteúdo total consolidado: {len(total)} caracteres.")
                    prompt = montar_prompt_resumo("documentos consolidados", total[:20000], ctx_sup)
                    resp  = self._gemini.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt)
                    if self._cancel_event.is_set():
                        return
                    self._resumo_texto = getattr(resp, "text", "")
                    self._log_ui(f"[OK] Resumo consolidado gerado.")
            self.after(0, self._atualizar_area_texto)
        except Exception as e:
            self._log_ui(f"[ERRO] {e}")
        finally:
            self._set_busy_ui(self.btn_gerar_resumo, "  Gerando...", False)
            self._cancel_event.clear()

    # ------------------------------------------------------------------ questões
    def _iniciar_thread_questoes(self):
        topico = self.entry_topico_resumo.get().strip()
        try:
            qtd = _validar_qtd(self.spin_questoes.get())
            self._log_ui(f"[INFO] Quantidade de questões solicitada: {qtd}")
        except Exception:
            self._log("[ERRO] Quantidade inválida.")
            return
        selecionados = self._obter_pdfs_selecionados()
        if not topico and not selecionados:
            self._log("[ERRO] Forneça tópico ou selecione PDFs.")
            return
        self._set_busy(self.btn_gerar_questoes, "  Gerando...", busy=True)
        self._cancel_event.clear()
        threading.Thread(target=self._gerar_questoes_thread, args=(topico, selecionados, qtd), daemon=True).start()

    def _gerar_questoes_thread(self, topico, pdfs, qtd):
        try:
            if self._cancel_event.is_set():
                self._log_ui("[SYS] Geração de questões cancelada.")
                return
            if topico:
                self._log_ui(f"[RAG] Buscando contexto para o tópico: '{topico}'")
                conteudo = self._vs.buscar_contexto_por_topico(topico, k=40)
                if not conteudo:
                    self._log_ui("[ERRO] Nenhum conteúdo.")
                    return
                ctx_sup, _ = self._vs.buscar_contexto_suporte(topico, self._log_ui)
                prompt = montar_prompt_questoes(qtd, conteudo, ctx_sup)
                self._log_ui(f"[AI] Solicitando {qtd} questões ao Gemini...")
                resp   = self._gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                if self._cancel_event.is_set():
                    self._log_ui("[SYS] Cancelamento detectado após resposta da API.")
                    return
                self._questoes_texto = getattr(resp, "text", "")
                self._log_ui(f"[OK] Questões geradas ({len(self._questoes_texto)} caracteres).")
            else:
                conteudos = self._extrair_texto_completo_pdfs(pdfs)
                if not conteudos:
                    return
                if self._cancel_event.is_set():
                    return
                if not self._misturar_conteudos:
                    partes = []
                    for nome, texto in conteudos.items():
                        if self._cancel_event.is_set():
                            break
                        self._log_ui(f"[RAG] Gerando questões para: {nome}")
                        sup, _ = self._vs.buscar_contexto_suporte(nome, self._log_ui)
                        prompt = montar_prompt_questoes(qtd, texto, sup)
                        resp   = self._gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                        if self._cancel_event.is_set():
                            break
                        partes.append(f"--- QUESTÕES: {nome} ---\n{getattr(resp,'text','')}")
                    self._questoes_texto = "\n\n".join(partes)
                    self._log_ui(f"[OK] {len(partes)} conjunto(s) de questões gerados.")
                else:
                    total = "\n\n".join(conteudos.values())
                    sup_total = "\n".join(
                        self._vs.buscar_contexto_suporte(p, self._log_ui)[0] for p in pdfs)
                    self._log_ui(f"[RAG] Conteúdo total consolidado: {len(total)} caracteres.")
                    prompt = montar_prompt_questoes(qtd, total, sup_total)
                    resp = self._gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                    if self._cancel_event.is_set():
                        return
                    self._questoes_texto = getattr(resp, "text", "")
                    self._log_ui(f"[OK] Questões consolidadas geradas.")
            self.after(0, self._atualizar_area_texto)
        except Exception as e:
            self._log_ui(f"[ERRO] {e}")
        finally:
            self._set_busy_ui(self.btn_gerar_questoes, "  Gerando...", False)
            self._cancel_event.clear()

    def _atualizar_area_texto(self):
        self.text_resumo_questoes.delete("1.0", "end")
        if self._resumo_texto:
            self.text_resumo_questoes.insert("end", "═══ RESUMO ═══\n\n" + self._resumo_texto + "\n\n")
        if self._questoes_texto:
            self.text_resumo_questoes.insert("end", "═══ QUESTÕES ═══\n\n" + self._questoes_texto)
        if not self._resumo_texto and not self._questoes_texto:
            self.text_resumo_questoes.insert("end", "Nenhum conteúdo gerado ainda.\nUse os botões acima.")

    def _limpar_conteudo(self):
        self._resumo_texto = ""
        self._questoes_texto = ""
        self.text_resumo_questoes.delete("1.0", "end")
        self._log("[SYS] Conteúdo limpo.")

    # ------------------------------------------------------------------ exportar PDF
    def _exportar_pdf(self):
        if not self._resumo_texto and not self._questoes_texto:
            self._log("[ERRO] Nada para exportar.")
            return
        from tkinter import filedialog
        arquivo = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not arquivo:
            return

        def limpar_caracteres_especiais(t: str) -> str:
            subs = {
                '\u2013': '-',
                '\u2014': '-',
                '\u2018': "'",
                '\u2019': "'",
                '\u201c': '"',
                '\u201d': '"',
                '\u2022': '*',
                '\u20ac': 'EUR',
            }
            for k, v in subs.items():
                t = t.replace(k, v)
            return t

        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            try:
                pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
                pdf.set_font('DejaVu', '', 11)
                use_unicode = True
            except:
                pdf.set_font('Arial', '', 11)
                use_unicode = False

            pdf.set_font('', 'B', 14)
            pdf.cell(200, 8, txt="Resumo e Questoes Geradas", ln=True, align='C')
            pdf.ln(6)

            if self._resumo_texto:
                pdf.set_font('', 'B', 12)
                pdf.cell(200, 7, txt="RESUMO", ln=True)
                pdf.set_font('', '', 11)
                texto = limpar_caracteres_especiais(self._resumo_texto)
                if not use_unicode:
                    texto = texto.encode('latin-1', 'replace').decode('latin-1')
                for line in texto.splitlines():
                    if line.strip():
                        pdf.multi_cell(0, 6, line)
                        pdf.ln(2)
                    else:
                        pdf.ln(4)
                pdf.ln(4)

            if self._questoes_texto:
                pdf.set_font('', 'B', 12)
                pdf.cell(200, 7, txt="QUESTÕES", ln=True)
                pdf.set_font('', '', 11)
                texto = limpar_caracteres_especiais(self._questoes_texto)
                if not use_unicode:
                    texto = texto.encode('latin-1', 'replace').decode('latin-1')
                for line in texto.splitlines():
                    if line.strip():
                        pdf.multi_cell(0, 6, line)
                        pdf.ln(2)
                    else:
                        pdf.ln(4)

            pdf.output(arquivo)
            self._log(f"[OK] PDF exportado: {arquivo}")
        except Exception as e:
            self._log(f"[ERRO] PDF: {e}")


# ---------------------------------------------------------------------------
# 7. ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = AppAnki()
    app.mainloop()