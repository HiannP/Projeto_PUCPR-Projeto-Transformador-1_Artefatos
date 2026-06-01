"""
bert.py -- treinar_e_avaliar_bert.py
================================
Faz o Fine-Tuning e Avalia um modelo da familia BERT (ex: DistilBERT)
na tarefa de classificacao Fake/True usando o MESMO split 70/15/15.

PRE-REQUISITOS:
    pip install pandas scikit-learn matplotlib seaborn torch transformers datasets accelerate

USO BASICO:
    python bert.py
    (Uma janela será aberta para você selecionar o CSV)
"""

import argparse
import os
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

def selecionar_arquivo_csv() -> str:
    """Abre uma janela para o usuario selecionar o arquivo CSV."""
    root = tk.Tk()
    root.withdraw()
    caminho = filedialog.askopenfilename(
        title="Selecione o arquivo CSV do Corpus",
        filetypes=[("Arquivos CSV", "*.csv"), ("Todos os arquivos", "*.*")]
    )
    return caminho

def carregar_splits_treinamento(caminho_csv: str, amostra_base: int = 5000, seed: int = 42):
    """
    Diferente das APIs (que so usam o Teste), o BERT precisa aprender.
    Esta funcao reproduz o MESMO split 70/15/15, mas retorna as 3 partes.
    """
    df_completo = pd.read_csv(caminho_csv, encoding="utf-8-sig")

    df_completo = df_completo.dropna(subset=["text", "label"])
    df_completo["text"] = df_completo["text"].astype(str)
    df_completo["label"] = df_completo["label"].astype(int)

    df_completo = df_completo.dropna(subset=["title"]).copy()
    df_completo["title"] = df_completo["title"].astype(str)

    df_amostra = df_completo.sample(amostra_base, random_state=seed)

    # 70% Treino, 30% Restante
    df_treino, df_temp = train_test_split(
        df_amostra, test_size=0.30, random_state=seed, stratify=df_amostra["label"],
    )

    # Dos 30% restantes: 15% Validacao, 15% Teste
    df_val, df_teste = train_test_split(
        df_temp, test_size=0.50, random_state=seed, stratify=df_temp["label"],
    )

    # Resetando os indices para o Hugging Face Dataset nao se perder
    return df_treino.reset_index(drop=True), df_val.reset_index(drop=True), df_teste.reset_index(drop=True)

def calcular_e_salvar_metricas(df_teste: pd.DataFrame, pasta_saida: Path, nome_modelo: str) -> None:
    """Calcula acuracia, precision, recall, F1 e salva a matriz de confusao."""
    y_true = df_teste["label"].values
    y_pred = df_teste["pred"].values

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    relatorio = classification_report(y_true, y_pred, target_names=["FALSA (0)", "VERDADEIRA (1)"], digits=4)

    linhas = [
        "=" * 60,
        f"METRICAS -- {nome_modelo} no test split",
        "=" * 60,
        f"Total avaliado no teste: {len(df_teste)}",
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

    # Gerando grafico da Matriz de Confusao com Seaborn (igual ao seu Colab)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FALSA (0)', 'VERDADEIRA (1)'],
                yticklabels=['FALSA (0)', 'VERDADEIRA (1)'], ax=ax)
    
    ax.set_xlabel("Rótulo Previsto")
    ax.set_ylabel("Rótulo Real")
    ax.set_title(f"Matriz de Confusao -- {nome_modelo}")
    fig.tight_layout()
    fig.savefig(pasta_saida / "matriz_confusao.png", dpi=120)
    plt.close(fig)

def compute_metrics_treino(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amostra-base", type=int, default=5000)
    p.add_argument("--modelo", default="distilbert-base-multilingual-cased", help="Modelo Hugging Face")
    p.add_argument("--epocas", type=int, default=2, help="Numero de epocas de treinamento")
    p.add_argument("--batch", type=int, default=16, help="Tamanho do lote (reduza se faltar memoria VRAM)")
    p.add_argument("--max-len", type=int, default=256, help="Tamanho maximo de tokens do texto")
    p.add_argument("--saida", default="resultados_bert_test")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("="*60)
    print("VERIFICACAO DE HARDWARE PARA O BERT")
    print("="*60)
    tem_gpu = torch.cuda.is_available()
    print(f"GPU disponível: {tem_gpu}")
    if tem_gpu:
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    print("="*60 + "\n")

    caminho_csv = selecionar_arquivo_csv()
    if not caminho_csv:
        sys.exit("Nenhum arquivo CSV selecionado. Encerrando.")

    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Carregando {caminho_csv} e gerando splits (Treino/Val/Teste)...")
    df_treino, df_val, df_teste = carregar_splits_treinamento(caminho_csv, args.amostra_base, seed=args.seed)
    print(f"      OK -- Treino: {len(df_treino)} | Val: {len(df_val)} | Teste: {len(df_teste)}")

    dataset_hf = DatasetDict({
        'train': Dataset.from_pandas(df_treino),
        'validation': Dataset.from_pandas(df_val),
        'test': Dataset.from_pandas(df_teste)
    })

    print(f"\n[2/4] Baixando Tokenizador e Tokenizando os textos (max_length={args.max_len})...")
    tokenizer = AutoTokenizer.from_pretrained(args.modelo)

    def tokenize_function(examples):
        # Concatenamos Titulo e Texto para o BERT ter o contexto completo (similar aos Prompts das APIs)
        textos_combinados = [str(t) + " - " + str(x) for t, x in zip(examples["title"], examples["text"])]
        return tokenizer(textos_combinados, padding="max_length", truncation=True, max_length=args.max_len)

    tokenized_datasets = dataset_hf.map(tokenize_function, batched=True)

    print(f"\n[3/4] Baixando Arquitetura do Modelo e Iniciando Treinamento ({args.epocas} epocas)...")
    model = AutoModelForSequenceClassification.from_pretrained(args.modelo, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(pasta / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epocas,
        weight_decay=0.01,
        fp16=tem_gpu, 
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics_treino,
    )

    trainer.train()

    print(f"\n[4/4] Treino finalizado! Avaliando o modelo cego no conjunto de Teste...")
    resultados_teste = trainer.predict(tokenized_datasets["test"])
    
    logits = resultados_teste.predictions
    y_previsto = np.argmax(logits, axis=-1)

    df_teste_resultado = df_teste.copy()
    df_teste_resultado["pred"] = y_previsto

    saida_csv = pasta / "predicoes_bert.csv"
    df_teste_resultado[["title", "text", "label", "pred"]].to_csv(saida_csv, index=False, encoding="utf-8-sig")
    print(f"      Predicoes salvas em {saida_csv}")

    calcular_e_salvar_metricas(df_teste_resultado, pasta, args.modelo)

    print(f"\nPronto! Resultados em: {pasta.resolve()}")

if __name__ == "__main__":
    main()