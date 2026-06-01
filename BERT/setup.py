#!/usr/bin/env python3
"""
Script para verificar e instalar dependências do teste com BERT local
"""
import subprocess
import sys

PACKAGES = [
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "torch",
    "transformers",
    "datasets",
    "accelerate"
]

def install_packages():
    print("A instalar as dependências do BERT (Isto pode demorar devido ao PyTorch)...")
    cmd = [sys.executable, "-m", "pip", "install"] + PACKAGES
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

if __name__ == "__main__":
    if install_packages():
        print("\n✅ Dependências instaladas com sucesso!")
        print("\nAgora pode executar:")
        print("  python bert.py")
    else:
        print("\n❌ Erro ao instalar dependências!")
        sys.exit(1)