# Home Guardian: Modelo Matemático

> Modelo acadêmico de **apoio à decisão**. Não substitui detectores certificados de fumaça, gás ou vazamento, que continuam funcionando de forma independente.

## 1. Enquadramento do problema

O Home Guardian aloca **recursos limitados de resposta** a ocorrências domésticas simultâneas. Os recursos otimizados são:

| Recurso | Onde aparece no modelo |
|---|---|
| **Custo** | custo de cada intervenção (cortar gás, acionar válvula, ligar para contato) |
| **Tempo** | prazo até o dano e tempo de execução de cada ação |
| **Atenção** | capacidade limitada do morador de tratar ocorrências ao mesmo tempo |

O **risco** não é o recurso otimizado: é o que a função objetivo busca reduzir com os recursos disponíveis.

A interface do morador (mensagens simples, botões *Sim* / *Preciso de ajuda*) faz parte do modelo: a resposta dele atualiza a probabilidade de a ocorrência ser um perigo real.

O sistema é **preventivo e reativo**: a parte preventiva estima, a partir das leituras, a probabilidade $p_i^0$ e o prazo $d_i$ (Seções 4 e 5); a parte reativa decide como responder (Seções 8 e 9). A saída da primeira alimenta a segunda.

---

## 2. Conjuntos e índices

- $I = \{1,\dots,n\}$: ocorrências ativas (ex.: fogão, vazamento, sobreaquecimento).
- $J$: ações possíveis (notificar, alarme, acionar contato, cortar gás, cortar energia, fechar válvula d'água).
- $J_i \subseteq J$: ações aplicáveis à ocorrência $i$.
- $C \subset (I\times J)\times(I\times J)$: pares de ações em conflito.

---

## 3. Parâmetros

| Símbolo | Significado | Origem |
|---|---|---|
| $S_i$ | gravidade potencial (1 a 10) | `parametros.json` |
| $p_i$ | probabilidade de perigo real (após resposta do morador) | Seção 4 |
| $d_i$ | prazo até o dano (min) | Seção 5 |
| $R_i$ | risco da ocorrência | Seção 6 |
| $e_{ij}\in[0,1]$ | eficácia da ação $j$ sobre $i$ | matriz $E$ |
| $c_{ij}$ | custo da ação | `parametros.json` |
| $\tau_{ij}$ | tempo até a ação surtir efeito (min) | `parametros.json` |
| $a_{j}$ | carga de atenção exigida do morador pela ação $j$ | `parametros.json` |
| $q_{j}$ | incômodo da ação $j$ se for alarme falso | `parametros.json` |
| $\pi_j\in\{0,1\}$ | ação autorizada/disponível na casa | configuração |
| $B$ | orçamento de intervenção | `parametros.json` |
| $K$ | capacidade de atenção simultânea | `parametros.json` |
| $\lambda_c,\lambda_f$ | pesos de custo e de falso alarme | `parametros.json` |

---

## 4. Probabilidade de perigo real

### 4.1 Prior: regressão logística sobre as leituras

A probabilidade inicial $p_i^0$ vem de uma regressão logística sobre duas características da leitura, normalizadas pela escala do sensor ($x_{crit}-x_{normal}$):

$$
p_i^0 = \sigma\bigl(w_1\,\dot{x}_i + w_2\,\ddot{x}_i + b\bigr), \qquad \sigma(z)=\frac{1}{1+e^{-z}}
$$

onde $\dot{x}_i$ é a taxa de variação (Seção 5) e $\ddot{x}_i$ a aceleração. Os pesos são ajustados por máxima verossimilhança (L-BFGS-B, com regularização L2). O modelo distingue anomalias **perigosas** (crescimento exponencial acelerado) de **benignas** (aquecem e estabilizam, como uma panela em uso normal).

Com dados simulados, o ajuste obteve AUC ≈ 0,80 e acurácia ≈ 0,74. É útil, mas não perfeito: por isso $p_i^0$ de perigos reais fica em torno de 0,4 a 0,5, e a resposta do morador resolve a incerteza restante.

### 4.2 Atualização bayesiana pela resposta do morador

Seja $m_i \in \{\text{sim},\ \text{ajuda},\ \text{sem resposta}\}$ a resposta à pergunta exibida na tela (*"O fogão está ligado há mais tempo que o habitual. Você está utilizando-o?"*).

Com o **Teorema de Bayes**, o prior $p_i^0$ é atualizado:

$$
p_i = P(\text{perigo}\mid m_i) = \frac{P(m_i\mid\text{perigo})\,p_i^0}{P(m_i\mid\text{perigo})\,p_i^0 + P(m_i\mid\text{seguro})\,(1-p_i^0)}
$$

Verossimilhanças (hipóteses de simulação, a calibrar):

| Resposta $m_i$ | $P(m_i\mid\text{perigo})$ | $P(m_i\mid\text{seguro})$ | Efeito |
|---|---|---|---|
| Sim, estou usando | 0,20 | 0,80 | $p_i$ cai, alerta tende a ser suprimido |
| Preciso de ajuda | 0,90 | 0,05 | $p_i$ sobe, escalada imediata |
| Sem resposta no prazo | 0,60 | 0,10 | $p_i$ sobe, escalada |

O alerta é suprimido quando $p_i < 0{,}3$. Se o morador foi consultado e **não respondeu a nenhuma** consulta, ele é tratado como indisponível e a capacidade de atenção efetiva passa a $K = 0$; caso contrário vale o $K$ configurado.

---

## 5. Cálculo (taxa de variação e prazo)

Para ocorrências com variável contínua (temperatura $x$, nível de água), a **taxa** $\dot{x}$ é a derivada estimada pelo coeficiente angular de uma regressão linear sobre as últimas leituras (janela de 12 pontos), e a **aceleração** $\ddot{x}$ pelo dobro do coeficiente quadrático de uma regressão de grau 2.

O prazo até o valor crítico $x_{crit}$, pela extrapolação linear, é:

$$
d_i^{lin} = \frac{x_{crit} - x_i(t)}{\dot{x}_i}
$$

Em crescimento acelerado a extrapolação linear é **otimista**: nas simulações, o prazo real foi de 45% a 66% do estimado. Aplica-se então um fator de segurança $\phi_s$ por sensor:

$$
d_i = \phi_s\, d_i^{lin}, \qquad \phi_{fogão}=0{,}55,\ \ \phi_{vazamento}=0{,}35,\ \ \phi_{sobreaquec.}=0{,}55
$$

calibrado em simulação (percentil 10 da razão prazo real / estimado) para cobrir cerca de 90% dos casos. **Vale só dentro do modelo simulado**; com dados reais teria de ser refeito.

A forma simbólica (derivada, tempo até o crítico e integral do risco) é obtida com **SymPy** e conferida contra versões numéricas (diferença finita e integração do SciPy).

O **risco acumulado** de uma ocorrência não tratada entre $0$ e $t$ é a integral da taxa de risco:

$$
\text{Risco}_i(t) = \int_0^{t} h_i(s)\,ds
$$

onde $h_i(s)$ cresce à medida que $s$ se aproxima de $d_i$ (ex.: $h_i(s) = S_i\,p_i\,e^{k_i s}$).

---

## 6. Risco por ocorrência

$$
R_i = S_i \cdot p_i \cdot u_i, \qquad u_i = \min\!\Bigl(1,\ \frac{d_{ref}}{\max(d_i,\ \varepsilon)}\Bigr)
$$

$u_i\in(0,1]$ é a **urgência**. Com o prazo de referência $d_{ref}=10$ min, a urgência vale 1 (máxima) quando $d_i \le d_{ref}$ e diminui proporcionalmente a $d_{ref}/d_i$ quando o prazo é maior. Ou seja, ocorrências com bastante tempo pesam menos, e o teto de 1 mantém $R_i$ na mesma escala de custo e gravidade.

### 6.1 Envelhecimento da gravidade

Uma ocorrência deixada de fora do plano de ação não pode ficar esquecida. A cada rodada do otimizador em que ela fica sem ação principal, sua gravidade efetiva cresce:

$$
g_i^{ef} = \min\bigl(10,\ S_i\,(1 + \alpha\,k_i)\bigr), \qquad \alpha = 0{,}25
$$

onde $k_i$ é o número de rodadas sem ação. O otimizador passa a usar $g_i^{ef}$ no lugar de $S_i$. Assim a ocorrência acaba vencendo a competição por orçamento **sem regra especial** dentro do PLI. O contador volta a zero quando a ocorrência retorna ao estado NORMAL. Consequência: o `risco_inicial` registrado em rodadas seguintes já inclui a gravidade envelhecida.

---

## 7. Variáveis de decisão

$$
x_{ij} =
\begin{cases}
1 & \text{se a ação } j \text{ for executada na ocorrência } i\\
0 & \text{caso contrário}
\end{cases}
\qquad x_{ij}\in\{0,1\}
$$

---

## 8. Função objetivo

$$
\min Z \;=\; \underbrace{\sum_{i\in I} R_i\Bigl(1-\sum_{j\in J_i} e_{ij}\,x_{ij}\Bigr)}_{\text{risco residual}}
\;+\; \lambda_c\underbrace{\sum_{i,j} c_{ij}\,x_{ij}}_{\text{custo}}
\;+\; \lambda_f\underbrace{\sum_{i,j} (1-p_i)\,q_j\,x_{ij}}_{\text{penalidade por falso alarme}}
$$

O terceiro termo faz o sistema evitar ações invasivas (cortar gás, ligar para contato) quando a probabilidade de perigo real é baixa.

Em forma matricial, com $R$ o vetor de riscos e $E$ a matriz de eficácia:

$$
\text{risco residual} = R^\top\bigl(\mathbf{1} - E\,x\bigr)
$$

---

## 9. Restrições

**(R1) No máximo uma ação principal por ocorrência** (mantém o modelo linear, sem o produto $\prod(1-e_{ij}x_{ij})$):

$$
\sum_{j\in J_i} x_{ij} \le 1 \qquad \forall i\in I
$$

**(R2) Orçamento de intervenção:**

$$
\sum_{i,j} c_{ij}\,x_{ij} \le B
$$

**(R3) Prazo de resposta** (ação só é válida se fizer efeito antes do dano):

$$
\tau_{ij}\,x_{ij} \le d_i \qquad \forall i,j
$$

Na implementação, ações com $\tau_{ij}>d_i$ são removidas de $J_i$ antes de resolver.

**(R4) Capacidade de atenção do morador:**

$$
\sum_{i,j} a_{j}\,x_{ij} \le K
$$

**(R5) Conflitos entre ações:**

$$
x_{ij} + x_{kl} \le 1 \qquad \forall\,\bigl((i,j),(k,l)\bigr)\in C
$$

Exemplo: acionar o mesmo contato de confiança para duas ocorrências ao mesmo tempo.

**(R6) Ações autorizadas:**

$$
x_{ij} \le \pi_j \qquad \forall i,j
$$

**(R7) Integralidade:**

$$
x_{ij}\in\{0,1\}
$$

**Classe do problema:** programação linear inteira binária, uma **mochila multidimensional** (R2 e R4) com atribuição e conflitos. Resolvido com `PuLP` (CBC).

---

## 10. Máquina de estados (Matemática Discreta)

Cada ocorrência percorre um grafo dirigido $G=(V,A)$ de 8 estados: NORMAL, ALERTA, CONSULTA, SUPRIMIDO, ESCALADA, CRÍTICO, EM_RESPOSTA e RESOLVIDO. As transições:

| Estado atual | Evento | Próximo estado |
|---|---|---|
| NORMAL | leitura acima do limiar | ALERTA |
| ALERTA | pergunta enviada ao morador | CONSULTA |
| ALERTA | leitura normal | NORMAL |
| CONSULTA | "Sim" e $p_i$ abaixo do limiar | SUPRIMIDO |
| CONSULTA | "Sim", mas $p_i$ ainda alta (reconsulta) | ALERTA |
| CONSULTA | "Preciso de ajuda" | CRÍTICO |
| CONSULTA | timeout sem resposta | ESCALADA |
| CONSULTA | limite de consultas atingido | ESCALADA |
| CONSULTA | leitura normal | NORMAL |
| SUPRIMIDO | leitura piora ou novo timeout | ALERTA |
| SUPRIMIDO | leitura normal | NORMAL |
| ESCALADA | contato acionado sem resolução, **ou prazo abaixo do limite sem ação** | CRÍTICO |
| ESCALADA | otimizador seleciona ação | EM_RESPOSTA |
| ESCALADA | leitura normal | NORMAL |
| CRÍTICO | otimizador seleciona ação | EM_RESPOSTA |
| CRÍTICO | leitura normal | RESOLVIDO |
| EM_RESPOSTA | leitura volta ao normal | RESOLVIDO |
| EM_RESPOSTA | leitura piora (a ação não bastou) | CRÍTICO |
| RESOLVIDO | rearme | NORMAL |

O número máximo de consultas por ocorrência é 2.

**Quando o otimizador roda.** O PLI da Seção 9 é executado sobre as ocorrências em **ESCALADA e CRÍTICO** (não em ALERTA), e só quando não há mais consultas pendentes ao morador, de modo que as ocorrências escaladas dividam o mesmo orçamento.

**Ocorrência preterida.** Quem fica em ESCALADA sem ação principal recebe a **ação passiva** `notificar`, de custo zero e sem consumir a atenção do morador, tem a gravidade envelhecida (Seção 6.1) e, se o prazo restante cair abaixo de 3 min sem ação, passa a CRÍTICO, que aciona o contato de confiança.

**Análise do grafo.** Representa-se $G$ por uma matriz de adjacência $A\in\{0,1\}^{8\times8}$. A alcançabilidade é calculada por busca em largura e verifica-se que o grafo é **fortemente conexo**: todo estado alcança todos os outros, ou seja, o sistema sempre pode voltar ao repouso.

---

## 11. Álgebra linear: organização dos dados

- Matriz $E\in[0,1]^{n\times m}$ (ocorrências × ações): eficácia.
- Matrizes $C_{cost}$ e $A_{att}$ (mesmo formato): custos e carga de atenção.
- Máscara de viabilidade $F_{ij}=\mathbb{1}[\tau_{ij}\le d_i]\cdot\pi_j$: zera ações inviáveis.
- Matriz de adjacência do grafo de estados (Seção 10).

---

## 12. Cenários de simulação

| Cenário | Ocorrências simultâneas | Objetivo do teste |
|---|---|---|
| C1 | 1 (fogão) | comportamento base e efeito da resposta do morador |
| C2 | 2 (fogão + vazamento) | competição por orçamento e atenção |
| C3 | 3 (fogão + vazamento + sobreaquecimento) | conflitos e prazos curtos |
| C4 | 3, com orçamento reduzido | priorização forçada |
| C5 | *(preencher: descrição atual em `data/cenarios/c5.json`)* | *(preencher)* |
| C6 | *(preencher: descrição em `data/cenarios/c6.json`)* | *(preencher)* |

Os parâmetros dos cenários ficam em `data/cenarios/`. A aba **Cenários** da tela técnica mostra a descrição de cada um.

---

## 13. Estratégias de comparação (baseline)

| Estratégia | Regra |
|---|---|
| **Ingênua** | trata por ordem de chegada até esgotar orçamento/atenção |
| **Gulosa** | ordena por **ganho líquido** (redução de risco menos custo ponderado e penalidade de falso alarme) e escolhe até esgotar recursos |
| **PLI (proposta)** | solução ótima do modelo da Seção 9 |

**Métricas:** valor de $Z$, risco residual total, custo total, atenção usada, ganho percentual do PLI sobre cada estratégia e tempo de execução.

---

## 14. Análise de sensibilidade e Monte Carlo

- **Sensibilidade:** variar $B$, $K$, prazos $d_i$ e pesos $\lambda_c,\lambda_f$; plotar risco residual × orçamento (fronteira de Pareto). Um mapa de calor em $(B,K)$ mostra em que faixas o PLI se separa das heurísticas.
- **Monte Carlo:** sortear $S_i$, $p_i$ (distribuição Beta) e $d_i$ (distribuições em `parametros.json`), além de $B$ e $K$, em milhares de execuções; comparar a distribuição de $Z$ entre as três estratégias. Como o PLI é ótimo, ele nunca perde para as heurísticas (indicador "PLI pior" deve ser 0%).
- **Calibração de $p_i^0$:** regressão logística sobre séries simuladas rotuladas como perigo/seguro (Seção 4.1). As características e o rótulo ficam no SQLite, e o modelo pode ser **retreinado a partir do banco**.

---

## 15. Cobertura dos requisitos da disciplina

| Requisito | Onde |
|---|---|
| Modelagem matemática | Seções 3 a 9 |
| Implementação em Python | `modelo/otimizador.py` (PuLP), `modelo/calculo.py` (SymPy), `sensores/deteccao.py` (NumPy/SciPy) |
| Otimização e simulação | Seções 12 a 14 |
| Interpretação | `relatorios/graficos.py`, `app/tela_tecnica.py` |
| Cálculo | Seção 5 (derivada, prazo e integral do risco) |
| Álgebra linear | Seção 11 e matriz de adjacência da Seção 10 |
| Matemática discreta | Seção 10 (grafo de estados, alcançabilidade, conexidade) |
| Regressão / estatística | Seção 4 (logística e Bayes) |
| Operations Research | Seções 8 e 9 |
| Banco de dados | `persistencia/db.py` (SQLite) |

---

## 16. Limitações

- **Todos os dados são sintéticos.** As métricas da regressão medem a capacidade de recuperar o que foi simulado, não o desempenho em casas reais.
- Os parâmetros dos sensores (limiares, taxas, patamares), de eficácia, custo e as verossimilhanças são **hipóteses de simulação**, não medições.
- Os fatores de segurança do prazo foram calibrados em simulação e teriam de ser refeitos com dados reais.
- O modelo prioriza e recomenda; **não garante** segurança física.
- R1 simplifica o efeito combinado de várias ações sobre a mesma ocorrência; uma extensão futura pode modelá-lo com linearização.
- O orçamento $B$ não se acumula entre rodadas: uma ocorrência que escala depois das outras é decidida com orçamento novo, sem descontar o que já foi gasto. Uma extensão futura pode carregar o orçamento restante.