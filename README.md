# TP1 — Programação genética para regressão simbólica

Implementação em Python dos componentes de GP e da rotina experimental. Usa árvores de expressões numéricas e minimiza RMSE.

## Instalação

Requer Python 3.11 ou superior. A partir da raiz do projeto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Execução inicial

```bash
python -m gp run --dataset synth1 --population-size 20 --generations 5 --max-depth 4 --seed 0
```

O comando usa `configs/baseline.toml` e permite sobrescrever os parâmetros
mostrados. A configuração de referência usa população 100, 100 gerações,
profundidade máxima 7, torneio de tamanho 2, elitismo de um indivíduo,
`pc=0.9`, `pm=0.05` e reprodução direta com probabilidade 0.05.
Também pode ser usado o comando instalado `gp`, equivalente a `python -m gp`.

```bash
python -m gp run --dataset concrete --selection epsilon_lexicase --seed 1
python -m gp --help
python -m gp run --help
```

## Dados

Os CSVs originais ficam em `bases-TP1/datasets/<base>/`. O carregador usa os
arquivos `<base>-train.csv` e `<base>-test.csv`, sem cabeçalho, com a última
coluna como alvo e todas as anteriores como entradas. Não aplica normalização.

| Base | Entradas | Treino | Teste |
|---|---:|---:|---:|
| synth1 | 2 | 60 | 600 |
| synth2 | 2 | 300 | 1000 |
| concrete | 8 | 824 | 206 |

`concrete1`, nome utilizado no enunciado, é aceito como alias de `concrete`.
Os caminhos relativos dos comandos são resolvidos a partir do diretório atual;
execute na raiz ou informe `--config`, `--data-dir` e os diretórios de saída.

## Decisões da implementação

- **Profundidade:** a raiz tem profundidade 0; um terminal isolado tem
  profundidade 0. O limite 7 é contado em arestas e vale na inicialização,
  no crossover e na mutação. Quantidade de nós é uma métrica separada.
- **Inicialização:** ramped half-and-half distribui a população entre
  profundidades-alvo de 2 até a máxima, dividindo cada grupo entre grow e full.
  Quando a divisão não é exata, grupos e métodos diferem em no máximo um
  indivíduo. Grow pode terminar antes da profundidade-alvo; full a alcança.
- **Terminais:** variáveis de entrada e constantes efêmeras uniformes no
  intervalo configurado, inicialmente [-5, 5]. Grow amostra uniformemente
  entre funções, variáveis e um terminal gerador de constante.
- **Funções:** adição, subtração, multiplicação, divisão protegida, seno e
  cosseno na configuração inicial. Negação também está disponível. Todas
  operam sobre números reais; não é necessário GP tipado.
- **Divisão protegida:** retorna 1 quando `abs(denominador) <= 1e-12`.
  Overflow e outros resultados não finitos recebem tratamento na fitness.
- **Fitness:** RMSE de treino, calculado com escala para evitar overflow ao
  elevar resíduos ao quadrado. A penalidade/teto numérico é `1e100`.
  Qualquer saída não finita invalida o indivíduo inteiro e penaliza todos
  os seus casos. Erros absolutos por caso também são limitados a esse teto.
- **Seleção:** torneio com reposição e desempate aleatório; roleta com pesos
  `1/(1+RMSE)`; ε-lexicase com erros absolutos, ordem aleatória dos casos
  e MAD recalculado entre os candidatos de cada caso. A condição é
  `erro <= melhor erro + MAD`, preservando os melhores mesmo quando MAD=0.
- **Variação:** cada evento escolhe exclusivamente crossover (`pc`), mutação
  (`pm`) ou reprodução (`1-pc-pm`). É obrigatório `pc+pm <= 1`. Crossover
  troca subárvores e gera dois filhos; se resta só uma vaga, aproveita o
  primeiro. As probabilidades são de eventos, não de proporção de filhos.
  Mutação substitui uma subárvore por outra gerada com grow.
- **Limites:** um filho de crossover que excede a profundidade retorna ao pai
  correspondente, registrando a rejeição. Mutação usa somente a profundidade
  restante disponível. Pais não são modificados, pois as árvores são imutáveis.
- **Elitismo:** preserva os melhores `elite_size` indivíduos. O campeão
  histórico de treino também é salvo fora da população, inclusive quando
  o elitismo está desativado.
- **Treino/teste:** somente treino participa da evolução e da escolha de
  parâmetros. O campeão histórico é avaliado no teste depois da evolução.
  Os indicadores de teste permitem a análise final de generalização.

## Experimentos

`configs/experiments.toml` descreve etapas separadas. Cada configuração usa
as mesmas 30 sementes (0 a 29). Não é necessário executar o plano inteiro
para verificar o programa. Inspecione uma etapa antes de executá-la:

```bash
python -m gp experiment --stage population --datasets synth1 --dry-run
python -m gp experiment --stage population --datasets synth1 --seeds 0 1
```

O segundo comando reduz deliberadamente a quantidade de repetições para um
teste preliminar. Sem `--seeds`, valem as 30 do plano.

A sequência prevista é `population`, `generations`, `selection`, `crossover`,
`mutation`, `operators`, `elitism` e `transfer`. Nas etapas com um parâmetro,
todos os demais ficam fixos. A etapa `operators` é a comparação adicional dos
pares (0.9, 0.05) e (0.6, 0.3), e não isola o efeito de cada parâmetro.
Na etapa `mutation`, `pc=0.6` fica fixo para permitir comparar os dois valores
de `pm` sem violar a exclusividade.

Após analisar o treino, atualize ou copie o baseline com os parâmetros
escolhidos antes da etapa seguinte. Para escolhas diferentes em synth1 e
concrete, use `--datasets` para restringir a base e `--baseline` para informar
seu arquivo TOML. O baseline fornecido nunca é atualizado automaticamente.
Por exemplo, se salvar a escolha em `configs/synth1_chosen.toml`:

```bash
python -m gp experiment --stage transfer --baseline configs/synth1_chosen.toml
```

`transfer` executa apenas synth2, reutilizando a escolha feita em synth1.
Utilize diretórios de saída separados quando repetir a mesma etapa/semente.

## Resultados e análises

Cada execução cria um diretório único em `results/runs/` contendo:

- `config.json`: todos os parâmetros efetivos e a semente;
- `best_tree.json`: árvore reconstruível com `Node.from_dict`;
- `history.csv`: estatísticas da população inicial (geração 0) até a geração G;
- `operators.csv`: operador, pais, fitness média dos pais, fitness do filho,
  melhora/piora/empate e rejeições por profundidade;
- `summary.json`: desempenho de treino/teste, expressão, tamanho, profundidade,
  tempo total, dimensões dos dados e versões do ambiente.

As estatísticas por geração incluem melhor, pior e média do RMSE, duplicados,
tamanho/profundidade médios e máximos, inválidos e efeitos dos operadores.
Duplicados são `população - assinaturas estruturais distintas`; não se tenta
provar equivalência algébrica entre árvores diferentes.
Crossover é comparado ao RMSE médio dos dois pais; mutação e reprodução, ao
RMSE do respectivo pai. Os contadores de resultado se referem aos filhos que
entraram na nova população, incluindo retornos ao pai por limite. Rejeições
são tentativas de filhos excessivos por evento, registradas uma única vez.

```bash
python -m gp analyze
```

Isso gera `results/summaries/summary.csv` e figuras em `results/figures/`,
agrupando por base, etapa e configuração. A tabela apresenta média e
desvio-padrão amostral (ddof=1). Com uma repetição, o desvio fica vazio.
Sementes repetidas no mesmo grupo são rejeitadas, evitando contagem artificial
de repetições. As figuras mostram convergência, crescimento das árvores e
RMSE final de treino/teste. Tempos e identificadores de diretório variam entre
execuções; os resultados evolutivos são reproduzíveis no mesmo ambiente com
a mesma semente.

## Testes e organização

```bash
python -m pytest -q
```

`src/gp/` separa representação/inicialização/operadores, avaliação/seleção,
evolução/métricas, execução de experimentos e análise. `tests/` cobre esses
componentes, os dados fornecidos, persistência e a interface de terminal.
Os testes usam problemas pequenos e saídas temporárias; não executam o estudo
experimental completo.

