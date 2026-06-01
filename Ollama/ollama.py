"""
ollama.py -- avaliar_modelo_ollama.py
================================
Avalia modelos locais via Ollama na tarefa de classificacao Fake/True
usando o MESMO split 70/15/15 dos treinamentos anteriores.

PRE-REQUISITOS:
    1. Ollama instalado e a correr localmente (http://localhost:11434)
    2. pip install pandas scikit-learn requests matplotlib
    3. Ter feito pull do modelo previamente (ex: `ollama pull llama3`)

USO BASICO:
    python ollama.py
    (Uma janela será aberta para selecionares o ficheiro CSV)
"""

import argparse
import json
import os
import re
import sys
import time
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

# Mantemos os prompts exatamente iguais para controlo cientifico
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

def selecionar_ficheiro_csv() -> str:
    """Abre uma janela para o utilizador selecionar o ficheiro CSV."""
    root = tk.Tk()
    root.withdraw()
    caminho = filedialog.askopenfilename(
        title="Selecione o ficheiro CSV do Corpus",
        filetypes=[("Ficheiros CSV", "*.csv"), ("Todos os ficheiros", "*.*")]
    )
    return caminho

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
    """
    Tenta extrair a label (0 ou 1) da resposta do modelo.
    """
    if not bruto:
        return None
        
    bruto = bruto.strip()
    bruto = re.sub(r"^```(?:json)?\s*|\s*```$", "", bruto, flags=re.MULTILINE).strip()

    # 1. Tenta fazer o parse direto do JSON
    try:
        obj = json.loads(bruto)
        label = int(obj["label"])
        return label if label in (0, 1) else None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    # 2. Fallback: Procura pelo padrao "label": 0 ou "label": 1 no meio do texto
    match = re.search(r'"label"\s*:\s*([01])', bruto)
    if match:
        return int(match.group(1))
        
    # 3. Fallback final: Retorna o primeiro 0 ou 1 que encontrar na resposta
    for ch in bruto:
        if ch in ("0", "1"):
            return int(ch)
            
    return None

def classificar_uma(url_base: str, modelo_nome: str, titulo: str, texto: str, tentativas: int = 3) -> int | None:
    """Envia um unico pedido para a API local do Ollama."""
    titulo = (titulo or "")[:500]
    texto = (texto or "")[:MAX_CHARS_TEXTO]
    prompt_usuario_local = PROMPT_USUARIO.format(titulo=titulo, texto=texto)

    endpoint = f"{url_base.rstrip('/')}/api/generate"
    
    payload = {
        "model": modelo_nome,
        "system": PROMPT_SISTEMA,
        "prompt": prompt_usuario_local,
        "stream": False,
        "format": "json", # Forca o modelo a devolver JSON valido
        "options": {
            "temperature": 0.0,
            "num_predict": 20
        }
    }

    for tentativa in range(tentativas):
        try:
            resposta = requests.post(endpoint, json=payload, timeout=120)
            resposta.raise_for_status()
            
            dados_resposta = resposta.json()
            texto_gerado = dados_resposta.get("response", "")
            
            return parsear_resposta(texto_gerado)
            
        except requests.exceptions.RequestException as e:
            espera = 2 ** tentativa
            if tentativa < tentativas - 1:
                time.sleep(espera)
                continue
            print(f"[erro conexao ollama] {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"[erro] {e}", file=sys.stderr)
            return None
    return None

def rodar_predicoes(df: pd.DataFrame, url_base: str, modelo: str, workers: int) -> pd.DataFrame:
    """Gere as threads para processar o dataset (cuidado com sobrecarga local)."""
    # Testa a conexao antes de comecar
    try:
        requests.get(url_base, timeout=5)
    except requests.exceptions.RequestException:
        sys.exit(f"ERRO: Nao foi possivel conectar ao Ollama em {url_base}.\n"
                 f"Verifica se o servico esta a correr localmente.")

    predicoes = [None] * len(df)
    inicio = time.time()
    feitos = 0
    total = len(df)

    def tarefa(i: int):
        return i, classificar_uma(url_base, modelo, df.at[i, "title"], df.at[i, "text"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futuros = [pool.submit(tarefa, i) for i in range(total)]
        for fut in as_completed(futuros):
            i, pred = fut.result()
            predicoes[i] = pred
            feitos += 1
            if feitos % 10 == 0 or feitos == total: # Atualiza mais frequentemente (local e mais lento)
                elapsed = time.time() - inicio
                taxa = feitos / elapsed if elapsed else 0
                eta = (total - feitos) / taxa if taxa else 0
                print(f"  [{feitos}/{total}] {taxa:.2f} req/s -- ETA {eta:.0f}s")

    df = df.copy()
    df["pred"] = predicoes
    return df

def calcular_e_salvar_metricas(df: pd.DataFrame, pasta_saida: Path, nome_modelo: str) -> None:
    """Calcula exatidao, precisao, recall, F1 e guarda a matriz de confusao."""
    val = df.dropna(subset=["pred"]).copy()

    if len(val) == 0:
        print("\n[ERRO] Nenhuma predicao valida! Todos os pedidos falharam.")
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
        f"METRICAS -- {nome_modelo} (Ollama) no test split",
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
        print("[Aviso] matplotlib nao instalado. Matriz de confusao PNG nao foi gerada.")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amostra-base", type=int, default=5000)    
    p.add_argument("--modelo", default="llama3", help="Modelo local instalado no Ollama (ex: llama3, mistral)")
    p.add_argument("--url", default="http://localhost:11434", help="URL do servidor Ollama")
    p.add_argument("--workers", type=int, default=1, help="Workers em simultaneo (predefinido=1 para nao encravar a maquina local)")
    p.add_argument("--saida", default="resultados_ollama_test")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    caminho_csv = selecionar_ficheiro_csv()
    if not caminho_csv:
        sys.exit("Nenhum ficheiro CSV selecionado. A encerrar.")

    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] A carregar {caminho_csv} e a reproduzir o split...")
    df = carregar_test_split(caminho_csv, args.amostra_base, seed=args.seed)
    print(f"      OK -- {len(df)} linhas no test set")

    print(f"\n[2/3] A correr predicoes com o modelo {args.modelo} via Ollama ({args.workers} workers)...")
    df_pred = rodar_predicoes(df, args.url, args.modelo, args.workers)

    saida_csv = pasta / "predicoes.csv"
    df_pred[["title", "text", "label", "pred"]].to_csv(saida_csv, index=False, encoding="utf-8-sig")
    print(f"      Predicoes guardadas em {saida_csv}")

    print(f"\n[3/3] A calcular metricas...")
    calcular_e_salvar_metricas(df_pred, pasta, args.modelo)

    print(f"\nPronto! Resultados em: {pasta.resolve()}")

if __name__ == "__main__":
    main()