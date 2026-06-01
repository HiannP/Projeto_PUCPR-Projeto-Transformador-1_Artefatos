"""
gpt.py -- avaliar_modelo_gpt.py
================================
Avalia o GPT (OpenAI API) na tarefa de classificacao Fake/True
usando o MESMO split 70/15/15 do treinamento BERT, pra permitir
comparacao metodologicamente justa entre os modelos.

SPLIT (igual ao BERT):
    1. Amostra 5000 do corpus com random_state=42
    2. 70% treino (3500) -- ignorado, GPT nao precisa de treino
    3. 15% validacao (750) -- ignorado
    4. 15% teste (750) -- ESTE e usado pra avaliar

Resultado: GPT e BERT sao avaliados nas MESMAS 750 noticias.

PRE-REQUISITOS:
    1. Conta na OpenAI com credito: https://platform.openai.com/
    2. API key gerada em: https://platform.openai.com/api-keys

USO BASICO (PowerShell):
    $env:OPENAI_API_KEY="sk-..."
    python gpt.py --saida resultados_gpt_test

OPCOES:
    --csv corpus_fake_news_unificado.csv
    --modelo gpt-4o-mini
    --workers 5                            # OpenAI costuma tolerar mais paralelismo que tier 1 Anthropic
    --saida resultados_gpt_test/
    --amostra-base 5000

SAIDAS:
    predicoes.csv, metricas.txt, matriz_confusao.png
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
from openai import OpenAI, APIError, RateLimitError, APIStatusError
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
# Funcoes
# ---------------------------------------------------------------------------

def carregar_test_split(caminho_csv: str, amostra_base: int = 5000,
                        seed: int = 42) -> pd.DataFrame:
    """
    Reproduz EXATAMENTE o mesmo split do treinamento BERT:
    - amostra de 5000 com random_state=42
    - split 70/30 (train/temp) com random_state=42
    - split 50/50 do temp (val/test) com random_state=42

    Retorna apenas o test set (~750 linhas, 15% do total).
    """
    df_completo = pd.read_csv(caminho_csv, encoding="utf-8-sig")

    df_completo = df_completo.dropna(subset=["text", "label"])
    df_completo["text"] = df_completo["text"].astype(str)
    df_completo["label"] = df_completo["label"].astype(int)

    # Tambem precisa de title pro nosso prompt
    df_completo = df_completo.dropna(subset=["title"]).copy()
    df_completo["title"] = df_completo["title"].astype(str)

    # Amostra de 5000 (mesma seed do BERT)
    df_amostra = df_completo.sample(amostra_base, random_state=seed)

    # Split 70/30 (mesma seed)
    _, df_temp = train_test_split(
        df_amostra,
        test_size=0.30,
        random_state=seed,
        stratify=df_amostra["label"],
    )

    # Split 50/50 do temp (mesma seed)
    _, df_teste = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=seed,
        stratify=df_temp["label"],
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


def classificar_uma(client: OpenAI, modelo: str, titulo: str, texto: str,
                    tentativas: int = 3) -> int | None:
    titulo = (titulo or "")[:500]
    texto = (texto or "")[:MAX_CHARS_TEXTO]
    prompt = PROMPT_USUARIO.format(titulo=titulo, texto=texto)

    for tentativa in range(tentativas):
        try:
            resp = client.chat.completions.create(
                model=modelo,
                max_tokens=20,
                temperature=0,
                messages=[
                    {"role": "system", "content": PROMPT_SISTEMA},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            texto_resp = resp.choices[0].message.content if resp.choices else ""
            return parsear_resposta(texto_resp)
        except (RateLimitError, APIStatusError) as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "overloaded" in msg or "503" in msg or "529" in msg:
                espera = 2 ** tentativa
                if tentativa < tentativas - 1:
                    time.sleep(espera)
                    continue
            print(f"[erro api] {e}", file=sys.stderr)
            return None
        except APIError as e:
            print(f"[erro api] {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"[erro] {e}", file=sys.stderr)
            return None
    return None


def rodar_predicoes(df: pd.DataFrame, modelo: str, workers: int) -> pd.DataFrame:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "ERRO: defina a variavel de ambiente OPENAI_API_KEY.\n"
            "PowerShell: $env:OPENAI_API_KEY=\"sk-...\"\n"
            "Pegue uma chave em: https://platform.openai.com/api-keys"
        )

    client = OpenAI()
    predicoes = [None] * len(df)
    inicio = time.time()
    feitos = 0
    total = len(df)

    def tarefa(i: int):
        return i, classificar_uma(
            client, modelo, df.at[i, "title"], df.at[i, "text"]
        )

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


def calcular_e_salvar_metricas(df: pd.DataFrame, pasta_saida: Path) -> None:
    val = df.dropna(subset=["pred"]).copy()

    if len(val) == 0:
        print("\n[ERRO] Nenhuma predicao valida! Todas as chamadas falharam.")
        print("Verifique:")
        print("  - Se OPENAI_API_KEY esta setada corretamente")
        print("  - Se ha credito na conta")
        print("  - Se ha mensagens [erro] acima")
        return

    val["pred"] = val["pred"].astype(int)

    descartados = len(df) - len(val)
    y_true = val["label"].values
    y_pred = val["pred"].values

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    relatorio = classification_report(
        y_true, y_pred, target_names=["FALSA (0)", "VERDADEIRA (1)"], digits=4
    )

    linhas = [
        "=" * 60,
        "METRICAS -- GPT (OpenAI) no test split (mesmo do BERT)",
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
        ax.set_title("Matriz de confusao -- GPT (test split)")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14,
                )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(pasta_saida / "matriz_confusao.png", dpi=120)
        plt.close(fig)
        print(f"\n[ok] grafico salvo em {pasta_saida / 'matriz_confusao.png'}")
    except ImportError:
        print("[aviso] matplotlib nao instalado, pulei o grafico.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", default="corpus_fake_news_unificado.csv")
    p.add_argument("--amostra-base", type=int, default=5000,
                   help="Amostra inicial antes do split (igual ao BERT, default 5000)")
    p.add_argument("--modelo", default="gpt-4o-mini",
                   help="Modelo OpenAI (default: gpt-4o-mini)")
    p.add_argument("--workers", type=int, default=5,
                   help="Workers paralelos (default: 5)")
    p.add_argument("--saida", default="resultados_gpt_test")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Carregando {args.csv} e reproduzindo split do BERT...")
    print(f"      Amostra base: {args.amostra_base} -> 70/15/15 -> usando test (15%)")
    df = carregar_test_split(args.csv, args.amostra_base, seed=args.seed)
    print(f"      OK -- {len(df)} linhas no test set")
    print(f"      Distribuicao: {df['label'].value_counts().to_dict()}")

    print(f"\n[2/3] Rodando predicoes com {args.modelo} ({args.workers} workers)...")
    df_pred = rodar_predicoes(df, args.modelo, args.workers)

    saida_csv = pasta / "predicoes.csv"
    df_pred[["title", "text", "label", "pred"]].to_csv(
        saida_csv, index=False, encoding="utf-8-sig"
    )
    print(f"      Predicoes salvas em {saida_csv}")

    print(f"\n[3/3] Calculando metricas...")
    calcular_e_salvar_metricas(df_pred, pasta)

    print(f"\nPronto! Resultados em: {pasta.resolve()}")


if __name__ == "__main__":
    main()
