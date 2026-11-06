# Guia prático de modelos de Machine Learning

Coleção em português para consultar e adaptar em exercícios. **19 notebooks independentes**, com explicações em Markdown, código comentado e resultados executados. Os exemplos de regressão, classificação e agrupamento são sintéticos; o PCA usa uma base pública incluída no scikit-learn. Nenhum resultado aqui é evidência de desempenho em seus dados reais.

## Como começar
1. Abra esta pasta no VS Code e instale as extensões Python e Jupyter, se necessário.
2. Selecione como kernel o Python de `machine-learning/.venv/bin/python` (ambiente existente).
3. Abra `aprendizado_supervisionado/01_regressao_linear/exemplo.ipynb` ou `aprendizado_supervisionado/08_regressao_logistica/exemplo.ipynb`.
4. Execute as células na ordem, leia os resultados e só depois altere os dados.
5. Estude os comparativos ao fim de cada etapa supervisionada para selecionar entre famílias de modelos.

Os notebooks também funcionam em Jupyter. Não precisam baixar datasets. Para um ambiente novo, crie uma venv e instale as dependências de `requirements-guia.txt`. O arquivo registra versões usadas na validação, sem alterar as dependências do seu projeto.

## Ordem de estudo

### Aprendizado supervisionado

Primeiro regressão (prever números), depois classificação (prever categorias). Os comparativos fecham cada etapa.

1. [Regressao Linear](aprendizado_supervisionado/01_regressao_linear/exemplo.ipynb)
2. [Ridge](aprendizado_supervisionado/02_ridge/exemplo.ipynb)
3. [Lasso](aprendizado_supervisionado/03_lasso/exemplo.ipynb)
4. [Arvore Regressao](aprendizado_supervisionado/04_arvore_regressao/exemplo.ipynb)
5. [Floresta Regressao](aprendizado_supervisionado/05_floresta_regressao/exemplo.ipynb)
6. [Boosting Regressao](aprendizado_supervisionado/06_boosting_regressao/exemplo.ipynb)
7. [Comparacao Regressao](aprendizado_supervisionado/07_comparacao_regressao/exemplo.ipynb)
8. [Regressao Logistica](aprendizado_supervisionado/08_regressao_logistica/exemplo.ipynb)
9. [Arvore Classificacao](aprendizado_supervisionado/09_arvore_classificacao/exemplo.ipynb)
10. [Knn](aprendizado_supervisionado/10_knn/exemplo.ipynb)
11. [Naive Bayes](aprendizado_supervisionado/11_naive_bayes/exemplo.ipynb)
12. [Floresta Classificacao](aprendizado_supervisionado/12_floresta_classificacao/exemplo.ipynb)
13. [Svm](aprendizado_supervisionado/13_svm/exemplo.ipynb)
14. [Boosting Classificacao](aprendizado_supervisionado/14_boosting_classificacao/exemplo.ipynb)
15. [Comparacao Classificacao](aprendizado_supervisionado/15_comparacao_classificacao/exemplo.ipynb)

### Aprendizado não supervisionado

Explore grupos sem resposta conhecida. PCA fica no fim porque seu exemplo também usa classificação para avaliar a redução de colunas.

1. [Kmeans](aprendizado_nao_supervisionado/01_kmeans/exemplo.ipynb)
2. [Hierarquico](aprendizado_nao_supervisionado/02_hierarquico/exemplo.ipynb)
3. [Dbscan](aprendizado_nao_supervisionado/03_dbscan/exemplo.ipynb)
4. [Pca](aprendizado_nao_supervisionado/04_pca/exemplo.ipynb)

## Roteiro que se repete nos exemplos supervisionados
Problema → métrica → dados → reserva do teste → análise do treino → preparação no pipeline → baseline → validação cruzada e busca → escolha → teste final → previsão e interpretação.

Cada pasta individual compara seu algoritmo com uma baseline. Os comparativos colocam as famílias na mesma busca de decisão, sem selecionar pelo teste. O vencedor é definido pelo programa, não escolhido antecipadamente no texto.

## Como adaptar ao CSV do exercício
Substitua a geração de dados por algo como:
```python
from pathlib import Path
# Ajuste o caminho em relação à pasta do notebook.
dados = pd.read_csv(Path("..") / ".." / "meus_dados.csv")
y = dados["nome_do_alvo"]
X = dados.drop(columns=["nome_do_alvo"])
# Exclua IDs e atributos indisponíveis no instante da previsão.
# Depois faça a separação e a análise SOMENTE do treino, seguindo o notebook.
```
Atualize `numericas` e `categoricas`, a tarefa, a métrica e as grades. Investigue alvo ausente, duplicatas, unidades e categorias. Nos exemplos binários, 1 é o evento de interesse; para multiclasse é necessário adaptar AP e o acesso à coluna de probabilidade, além de escolher métricas como F1 macro.

## Escolher a métrica
| Situação | Métrica inicial | Atenção |
|---|---|---|
| Regressão, custo aproximadamente proporcional ao erro | MAE | Mesma unidade do alvo |
| Regressão, erros grandes são especialmente caros | RMSE | Mais sensível a erros extremos |
| Classificação desbalanceada, qualidade de ranking | Average Precision | Depende da prevalência; não define o limiar |
| Erro ao deixar passar um positivo é caro | Recall junto com precisão/custo | Ajuste limiar somente na validação |
| Classes com importância semelhante | Balanced accuracy / F1 macro | Confira métricas por classe |
| Agrupamento | Silhouette + cobertura + estabilidade | Geometria não comprova utilidade |

## Cuidados para reutilizar
- Teste não serve para escolher parâmetros, colunas ou algoritmo. Se usado repetidamente para decidir, deixa de ser uma avaliação independente.
- A média vencedora da busca tem otimismo de seleção; o desvio dos folds não é intervalo de confiança. Para uma avaliação mais robusta de todo o procedimento, estude validação aninhada.
- Dados temporais pedem validação temporal; várias linhas por pessoa/empresa pedem divisão por grupos. A divisão aleatória dos exemplos pressupõe observações independentes.
- Imputação, escala, PCA, seleção de atributos e reamostragem aprendem somente no treino de cada fold.
- SMOTE não é aplicado nestes exemplos. Se usar, coloque em `imblearn.pipeline.Pipeline`; nunca reamostre antes de separar treino/teste.
- Maior score não encerra a escolha: considere custo, estabilidade, interpretabilidade e utilidade. Não ajuste essas regras olhando o teste.
- Seeds fixas ajudam a reproduzir, mas não demonstram robustez a outras amostras.
- Esta coleção cobre algoritmos tabulares fundamentais, não todos os modelos existentes. Redes neurais, séries temporais, NLP e bibliotecas externas podem formar módulos posteriores.

## Referências oficiais
- [Vazamento e boas práticas](https://scikit-learn.org/stable/common_pitfalls.html)
- [Validação cruzada](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Agrupamento e suas métricas](https://scikit-learn.org/stable/modules/clustering.html)

Veja `VALIDACAO.md` para o registro real de execução.
