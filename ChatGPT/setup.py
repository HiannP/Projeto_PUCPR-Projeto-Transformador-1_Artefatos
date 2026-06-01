#!/usr/bin/env python3
"""
Script para verificar e instalar dependências
"""
import subprocess
import sys

PACKAGES = [
    "python-dotenv",
    "openai",
    "pandas",
    "scikit-learn",
    "matplotlib",
]

def install_packages():
    """Instala todos os pacotes necessarios"""
    print("Instalando dependências necessárias...")
    cmd = [sys.executable, "-m", "pip", "install"] + PACKAGES
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

if __name__ == "__main__":
    if install_packages():
        print("\n✅ Dependências instaladas com sucesso!")
        print("\nAgora você pode rodar:")
        print("  python gpt.py --csv corpus_fake_news_unificado.csv --saida resultados_gpt_test")
    else:
        print("\n❌ Erro ao instalar dependências!")
        sys.exit(1)
