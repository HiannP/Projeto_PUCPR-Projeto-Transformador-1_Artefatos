PROJETO DE DETECÇÃO DE FAKE NEWS

Este repositório contém todos os artefatos, códigos-fonte, dados e
resultados da avaliação de múltiplas Inteligências Artificiais na tarefa de classificação de notícias como Falsas ou Verdadeiras.

O objetivo principal desta pesquisa foi comparar a eficácia de abordagens Zero-Shot
(usando LLMs como Gemini, Claude e GPT) contra abordagens de Fine-Tuning tradicional
(usando DistilBERT Multilíngue).

Abaixo detalha-se a estrutura de diretórios e a função de cada arquivo entregue.

ESTRUTURA GERAL DOS DADOS E CORPUS

./corpus_fake_news_unificado.csv :
A base de dados principal contendo as notícias brutas. Todos os scripts utilizam
este arquivo como entrada (--csv). O dataset possui as colunas essenciais 'title',
'text' e 'label' (0 para Falsa, 1 para Verdadeira).

./readme.txt :
Este arquivo de documentação descrevendo a entrega.

./templates_env.txt :
Um arquivo de referência demonstrando a estrutura necessária das chaves de API
para quem desejar reproduzir os testes com o Gemini, Claude e GPT.

SCRIPTS E RESULTADOS DAS APIs NA CLOUD (ZERO-SHOT)

Estes scripts utilizam as APIs oficiais para classificar as notícias. Todos
implementam a mesma semente de divisão (seed=42) para extrair uma fração de teste
justa (15% da amostra) e avaliar a capacidade Zero-Shot dos modelos.

./Gemini/gemini_lite.py :
Script de teste exclusivo para o modelo Google Gemini 2.0 Flash Lite.

Execução: python gemini_lite.py --csv ../corpus_fake_news_unificado.csv

O parâmetro --workers define a concorrência da API (padrão 3).

./Gemini/resultados_gemini_lite_test/ :
Diretório gerado após a execução contendo os artefatos de saída:

predicoes.csv : Planilha detalhada com a coluna 'pred' (classificação da IA).

metricas.txt : Relatório formal contendo Acurácia, Precisão, Recall e F1.

matriz_confusao.png : Representação visual dos Acertos vs Erros.

./Claude/claude.py :
Script de teste exclusivo para o modelo Anthropic Claude 3 Haiku.
Contém funções robustas de processamento (parseamento) de JSON para lidar com respostas ruidosas.

Execução: python claude.py (Possui interface gráfica para selecionar o CSV).

./Claude/resultados_claude_haiku_test/ :
Diretório de saída com as métricas, log e matriz de confusão do Claude.

./GPT/gpt.py :
Script de teste para os modelos da OpenAI (padrão: gpt-4o-mini).

Execução: python gpt.py --csv ../corpus_fake_news_unificado.csv

O parâmetro --workers define a concorrência da API (padrão 5, pois a OpenAI
costuma tolerar maior paralelismo).

./GPT/resultados_gpt_test/ :
Diretório de saída contendo as métricas, predições e matriz de confusão do GPT.

SCRIPTS E RESULTADOS DOS MODELOS LOCAIS

Estes scripts dependem do hardware da máquina host para rodar, não exigindo chaves
de API, mas sim requisitos de processamento (CPU/GPU).

./Ollama/ollama_eval.py :
Script desenhado para conectar-se ao servidor local do Ollama (http://localhost:11434).

Execução: python ollama_eval.py --modelo llama3

O parâmetro --modelo permite trocar o LLM subjacente facilmente.

O parâmetro --workers 1 é utilizado para não sobrecarregar a memória (VRAM).

./Ollama/resultados_ollama_test/ :
Diretório contendo os resultados e a matriz de confusão da inferência local.

./BERT/bert.py :
Script responsável pela etapa de Fine-Tuning. Diferente dos LLMs, este script
carrega o modelo 'distilbert-base-multilingual-cased' da Hugging Face, e
realiza o treino ativo usando 70% dos dados, 15% para validação e 15% para teste cego.

Execução: python bert.py --epocas 2 --batch 16

Identifica automaticamente a presença de GPU (CUDA) para ativar precisão mista (fp16).

./BERT/resultados_bert_test/ :
O diretório de saída final do BERT. Após o treino completo, este diretório armazena
o metricas.txt com o resultado do teste cego, permitindo comparação direta com
as métricas extraídas dos diretórios do Gemini, Claude, GPT e Ollama.

./BERT/resultados_bert_test/checkpoints/ :
Diretório que armazena os pesos intermediários gerados durante as épocas de
treinamento pelo Hugging Face Trainer. (Pode estar vazio na entrega se os
pesos forem muito pesados para upload na Cloud).