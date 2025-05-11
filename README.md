# Classificação de Texto com CNN - Análise de Sentimento no Twitter

## 📅 Projeto de Aprendizado de Máquina - NLP com Deep Learning

### Ô Equipe:

* Felipe Botero
* Claudio Sampaio
* Henrique Cavalcante
* Tiago Souza

---

## 📖 Tema

Classificação de sentimentos em mensagens do Twitter utilizando Redes Neurais Convolucionais (CNNs) com TensorFlow/Keras.

---

## §1. Importação das Bibliotecas Necessárias

Utilizamos bibliotecas para:

* Manipulação de dados: `pandas`, `numpy`
* Visualização: `matplotlib`, `seaborn`
* Modelagem: `tensorflow.keras`, `sklearn`

```python
# Author: Yousef Mohamed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Dense, ReLU, Embedding, BatchNormalization, Concatenate, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
```

---

## §2. Leitura e Processamento dos Dados

### 2.1 Leitura dos dados

* O dataset foi dividido em treino, validação e teste.
* Cada linha possui uma frase e o sentimento associado.

### 2.2 Visualização da distribuição dos rótulos

* Geração de gráficos de pizza para visualizar o desbalanceamento.

### 2.3 Balanceamento dos dados

* Rótulos menos frequentes ("love" e "surprise") foram removidos.
* Os demais foram amostrados para garantir proporcionalidade.

### 2.4 Codificação e Tokenização

* `LabelEncoder` transforma os rótulos de texto em inteiros.
* `Tokenizer` converte texto para sequências numéricas.
* `pad_sequences` garante tamanho fixo (maxlen=50).

---

## §3. Construção do Modelo de Deep Learning

### 3.1 Arquitetura da Rede

* Duas branches paralelas (CNNs) com:

  * `Embedding` layer
  * `Conv1D + BatchNormalization + ReLU`
  * `Dropout` + `GlobalMaxPooling1D`
* Concatenadas e passadas por camadas densas para classificação.

### 3.2 Compilação do Modelo

* Otimizador: `Adamax`
* Métricas: `accuracy`, `precision`, `recall`
* Perda: `categorical_crossentropy`

### 3.3 Treinamento

* Epochs: 25
* Batch size: 256
* Validação em tempo real durante o treinamento

---

## §4. Avaliação e Visualização dos Resultados

### 4.1 Avaliação Quantitativa

* Avaliação nos conjuntos de treino e teste.
* Impressão de `accuracy`, `precision` e `recall`.

### 4.2 Visualização

* Gráficos de curva de aprendizado por epoch para:

  * `Loss`
  * `Accuracy`
  * `Precision`
  * `Recall`

### 4.3 Matriz de Confusão e Relatório

* `confusion_matrix` + `classification_report`
* Rótulos: `anger`, `fear`, `joy`, `sadness`

---

## §5. Salvamento do Modelo

* Tokenizer salvo com `pickle`.
* Modelo salvo no formato `.keras` (recomendado pelo Keras).

```python
model.save('nlp.keras')
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)
```

---

## §6. Função de Previsão

* Função `predict(text, model_path, token_path)`:

  * Carrega modelo e tokenizer.
  * Gera sequência para novo texto.
  * Retorna e plota probabilidade para cada emoção.

Exemplo:

```python
txt = 'I am very happy to finish this project'
predict(txt, 'nlp.keras', 'tokenizer.pkl')
```

---

## ✅ Resultados

* Accuracy no teste: **93.8%**
* Modelo robusto com boa capacidade de generalização
* CNN demonstrou excelente desempenho em classificação de sentimentos curtos no estilo Twitter

---

## 🎓 Referência Base

Dataset: [Emotions Dataset for NLP](https://www.kaggle.com/datasets/danofer/emotion-classification)

Autor base do notebook: Yousef Mohamed (Kaggle)

Projeto adaptado e expandido para fins acadêmicos.
