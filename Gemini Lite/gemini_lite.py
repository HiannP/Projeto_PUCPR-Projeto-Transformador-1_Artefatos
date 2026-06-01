"""
gemini_lite.py -- avaliar_modelo_gemini_lite.py
================================
Avalia o modelo Gemini Lite na tarefa de classificacao Fake/True
usando o MESMO split 70/15/15 dos treinamentos anteriores.

PRE-REQUISITOS:
    1. Conta no Google AI Studio
    2. API key configurada no arquivo .env como GEMINI_API_KEY

USO BASICO:
    python gemini_lite.py --csv corpus_fake_news_unificado.csv
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# Carrega as variaveis de ambiente do arquivo .env
load_dotenv()

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

PROMPT_SISTEMA = (
    "Voce e um classificador de noticias. Para cada noticia, decida se ela e "
    "FALSA ou VERDADEIRA. Considere indicios como sensacionalismo, falta de "
    "fontes, contradicoes factuais e linguagem manipulativa. "
    "Responda APENAS com JSON no formato: {\"label\": 0} para falsa "
    "ou {\"label\": 1} para verdadeira. Sem nenhum texto adicional."
)

PROMPT_USUARIO = (
    "Titulo: {titulo}\n\n"
    "Texto: {texto}\n\n"
    "Classifique como 0 (falsa) ou 1 (verdadeira) e responda apenas com o JSON."
)

MAX_CHARS_TEXTO = 12000

# ---------------------------------------------------------------------------
# Funcoes Principais
# ---------------------------------------------------------------------------

def carregar_test_split(caminho_csv: str, amostra_base: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Reproduz EXATAMENTE o mesmo split original."""
    df_completo = pd.read_csv(caminho_csv, encoding="utf-8-sig")

    df_completo = df_completo.dropna(subset=["text", "label"])
    df_completo["text"] = df_completo["text"].astype(str)
    df_completo["label"] = df_completo["label"].astype(int)

    df_completo = df_completo.dropna(subset=["title"]).copy()
    df_completo["title"] = df_completo["title"].astype(str)

    df_amostra = df_completo.sample(amostra_base, random_state=seed)

    _, df_temp = train_test_split(
        df_amostra, test_size=0.30, random_state=seed, stratify=df_amostra["label"],
    )

    _, df_teste = train_test_split(
        df_temp, test_size=0.50, random_state=seed, stratify=df_temp["label"],
    )

    return df_teste.reset_index(drop=True)

def parsear_resposta(bruto: str) -> int | None:
    if not bruto:
        return None
    bruto = bruto.strip()
    bruto = re.sub(r"^```(?:json)?\s*|\s*```$", "", bruto, flags=re.MULTILINE).strip()

    try:
        obj = json.loads(bruto)
        label = int(obj["label"])
        return label if label in (0, 1) else None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    match = re.search(r'"label"\s*:\s*([01])', bruto)
    if match:
        return int(match.group(1))
    for ch in bruto:
        if ch in ("0", "1"):
            return int(ch)
    return None

def classificar_uma(modelo_nome: str, titulo: str, texto: str, tentativas: int = 3) -> int | None:
    titulo = (titulo or "")[:500]
    texto = (texto or "")[:MAX_CHARS_TEXTO]
    prompt = PROMPT_USUARIO.format(titulo=titulo, texto=texto)

    model = genai.GenerativeModel(
        model_name=modelo_nome,
        system_instruction=PROMPT_SISTEMA,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=20
        )
    )

    for tentativa in range(tentativas):
        try:
            resp = model.generate_content(prompt)
            return parsear_resposta(resp.text)
        except ResourceExhausted as e:
            espera = 2 ** tentativa
            if tentativa < tentativas - 1:
                time.sleep(espera)
                continue
            print(f"[erro rate limit] {e}", file=sys.stderr)
            return None
        except GoogleAPIError as e:
            print(f"[erro api google] {e}", file=sys.stderr)
            return None
        except Exception as e:
            if "finish_reason: SAFETY" in str(e) or "SafetyException" in str(type(e).__name__):
                print(f"[aviso] Texto bloqueado pelos filtros de segurança.", file=sys.stderr)
                return None 
            print(f"[erro] {e}", file=sys.stderr)
            return None
    return None

def rodar_predicoes(df: pd.DataFrame, modelo: str, workers: int) -> pd.DataFrame:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "ERRO: defina a variavel de ambiente GEMINI_API_KEY no arquivo .env.\n"
            "Pegue uma chave em: https://aistudio.google.com/app/apikey"
        )
    
    genai.configure(api_key=api_key)

    predicoes = [None] * len(df)
    inicio = time.time()
    feitos = 0
    total = len(df)

    def tarefa(i: int):
        return i, classificar_uma(modelo, df.at[i, "title"], df.at[i, "text"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futuros = [pool.submit(tarefa, i) for i in range(total)]
        for fut in as_completed(futuros):
            i, pred = fut.result()
            predicoes[i] = pred
            feitos += 1
            if feitos % 25 == 0 or feitos == total:
                elapsed = time.time() - inicio
                taxa = feitos / elapsed if elapsed else 0
                eta = (total - feitos) / taxa if taxa else 0
                print(f"  [{feitos}/{total}] {taxa:.1f} req/s -- ETA {eta:.0f}s")

    df = df.copy()
    df["pred"] = predicoes
    return df

def calcular_e_salvar_metricas(df: pd.DataFrame, pasta_saida: Path, nome_modelo: str) -> None:
    val = df.dropna(subset=["pred"]).copy()

    if len(val) == 0:
        print("\n[ERRO] Nenhuma predicao valida! Todas as chamadas falharam.")
        return

    val["pred"] = val["pred"].astype(int)
    descartados = len(df) - len(val)
    y_true = val["label"].values
    y_pred = val["pred"].values

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    relatorio = classification_report(y_true, y_pred, target_names=["FALSA (0)", "VERDADEIRA (1)"], digits=4)

    linhas = [
        "=" * 60,
        f"METRICAS -- {nome_modelo} (Google) no test split",
        "=" * 60,
        f"Total avaliado:        {len(val)}",
        f"Falhas/descartados:    {descartados}",
        "",
        f"Accuracy:              {acc:.4f}",
        f"Precision (true=1):    {prec:.4f}",
        f"Recall    (true=1):    {rec:.4f}",
        f"F1        (true=1):    {f1:.4f}",
        "",
        f"Precision (macro):     {prec_macro:.4f}",
        f"Recall    (macro):     {rec_macro:.4f}",
        f"F1        (macro):     {f1_macro:.4f}",
        "",
        "Matriz de confusao (linhas = real, colunas = previsto):",
        "                  Pred FALSA   Pred VERDADEIRA",
        f"  Real FALSA      {cm[0,0]:>10d}   {cm[0,1]:>15d}",
        f"  Real VERDADEIRA {cm[1,0]:>10d}   {cm[1,1]:>15d}",
        "",
        "Relatorio detalhado:",
        relatorio,
    ]
    texto_metricas = "\n".join(linhas)
    print("\n" + texto_metricas)

    (pasta_saida / "metricas.txt").write_text(texto_metricas, encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["FALSA", "VERDADEIRA"])
        ax.set_yticks([0, 1], ["FALSA", "VERDADEIRA"])
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusao -- {nome_modelo}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(pasta_saida / "matriz_confusao.png", dpi=120)
        plt.close(fig)
    except ImportError:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="corpus_fake_news_unificado.csv")
    p.add_argument("--amostra-base", type=int, default=5000)
    p.add_argument("--modelo", default="gemini-2.0-flash-lite", help="Modelo Google Lite (default: gemini-2.0-flash-lite)")
    p.add_argument("--workers", type=int, default=3, help="Workers (Mantenha 3 para evitar rate limit na conta gratuita)")
    p.add_argument("--saida", default="resultados_gemini_lite_test")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Carregando {args.csv} e reproduzindo split...")
    df = carregar_test_split(args.csv, args.amostra_base, seed=args.seed)
    print(f"      OK -- {len(df)} linhas no test set")

    print(f"\n[2/3] Rodando predicoes com {args.modelo} ({args.workers} workers)...")
    df_pred = rodar_predicoes(df, args.modelo, args.workers)

    saida_csv = pasta / "predicoes.csv"
    df_pred[["title", "text", "label", "pred"]].to_csv(saida_csv, index=False, encoding="utf-8-sig")
    print(f"      Predicoes salvas em {saida_csv}")

    print(f"\n[3/3] Calculando metricas...")
    calcular_e_salvar_metricas(df_pred, pasta, args.modelo)

    print(f"\nPronto! Resultados em: {pasta.resolve()}")

if __name__ == "__main__":
    main()