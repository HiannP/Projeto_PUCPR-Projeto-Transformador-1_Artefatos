#!/usr/bin/env python3
"""
Script para verificar e instalar dependências do teste local com o Ollama
"""
import subprocess
import sys

PACKAGES = [
    "requests",
    "pandas",
    "scikit-learn",
    "matplotlib",
]

def install_packages():
    """Instala todos os pacotes necessários"""
    print("A instalar as dependências necessárias...")
    cmd = [sys.executable, "-m", "pip", "install"] + PACKAGES
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

if __name__ == "__main__":
    if install_packages():
        print("\n✅ Dependências instaladas com sucesso!")
        print("\nLembre-se de verificar se o aplicativo do Ollama está aberto e rodando no seu computador.")
        print("\nAgora pode executar:")
        print("  python ollama.py")
    else:
        print("\n❌ Erro ao instalar dependências!")
        sys.exit(1)