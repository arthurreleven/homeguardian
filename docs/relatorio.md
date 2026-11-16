# Home Guardian: Relatório do Projeto

**Disciplina:** Modelagem Matemática (CDC, 3º semestre), Smart Home 2050
**Tema:** otimização de recursos de resposta a ocorrências domésticas com Pesquisa Operacional

> Projeto acadêmico de apoio à decisão. Não substitui detectores certificados de fumaça, gás ou vazamento. Todos os dados são sintéticos.

---

## 1. Resumo

O Home Guardian é um sistema de segurança e prevenção residencial que monitora o fogão, a lavanderia e a área elétrica. Quando várias ocorrências surgem ao mesmo tempo, não basta enviar três notificações: é preciso decidir **quais receber intervenção, com qual ação e dentro de quais limites** de custo, tempo e atenção do morador.

Modelamos essa decisão como uma **programação linear inteira binária** (mochila multidimensional com conflitos), resolvida com PuLP. A entrada do modelo vem de uma etapa preventiva: regressão e derivada estimam a taxa de variação das leituras e o prazo até o valor crítico, uma regressão logística estima a probabilidade de perigo real e o Teorema de Bayes a atualiza com a resposta do morador.

Em 900 instâncias de Monte Carlo, o PLI reduziu o valor da função objetivo em **11,0% em média** (até 79,4%) em relação ao tratamento por ordem de chegada, e **nunca foi pior** que as duas heurísticas. Contra uma heurística gulosa bem projetada, o ganho médio foi de apenas **0,5%**: a gulosa já é quase ótima na maioria dos casos. O valor do PLI está na garantia de otimalidade e na robustez quando orçamento e atenção são escassos.

---

## 2. Objetivo e escopo

**Objetivo.** Aumentar a segurança e a autonomia do morador por meio de monitoramento e tomada de decisão otimizada, tanto preventiva (antecipar riscos) quanto reativa (gerenciar ocorrências).

**Dentro do escopo:** três situações de risco (fogão ligado por tempo excessivo, vazamento de água, sobreaquecimento de equipamento), detecção, priorização, resposta, histórico em SQLite e interface com duas telas.

**Fora do escopo:** hardware real (ESP32), API, voz, integração com outros módulos da Smart Home.

---

## 3. Modelagem (resumo)

A formulação completa está em [`modelo_matematico.md`](modelo_matematico.md). Em resumo:

- **Risco** de cada ocorrência: $R_i = S_i\,p_i\,u_i$, com gravidade $S_i$, probabilidade de perigo real $p_i$ e urgência $u_i=\min(1,d_{ref}/d_i)$.
- **Variáveis:** $x_{ij}\in\{0,1\}$, a ação $j$ é executada na ocorrência $i$.
- **Objetivo:**

$$
\min Z = \sum_i R_i\Bigl(1-\sum_j e_{ij}x_{ij}\Bigr) + \lambda_c\sum_{i,j} c_{ij}x_{ij} + \lambda_f\sum_{i,j}(1-p_i)\,q_j\,x_{ij}
$$

- **Restrições:** uma ação principal por ocorrência (R1), orçamento (R2), prazo (R3), atenção do morador (R4), conflitos (R5), ações autorizadas (R6) e integralidade (R7).
- **Máquina de estados** de 8 estados conduz cada ocorrência de NORMAL até RESOLVIDO. O otimizador roda sobre as ocorrências em ESCALADA e CRÍTICO.

O terceiro termo de $Z$ é o que impede o sistema de "intervir em tudo, sempre": ações invasivas só compensam quando $p_i$ é alta o bastante.

---

## 4. Implementação

| Camada | Arquivos | Tecnologia |
|---|---|---|
| Sensores simulados e detecção | `sensores/simulador.py`, `deteccao.py` | NumPy, SciPy |
| Cálculo simbólico | `modelo/calculo.py` | SymPy |
| Risco e otimizador (PLI) | `modelo/risco.py`, `otimizador.py` | PuLP (CBC) |
| Heurísticas de comparação | `modelo/heuristica.py` | Python |
| Interação com o morador | `modelo/interacao.py` | Bayes |
| Máquina de estados | `modelo/estados.py` | Python, NumPy |
| Simulação | `simulacao/cenarios.py`, `monte_carlo.py`, `sensibilidade.py` | NumPy |
| Persistência | `persistencia/db.py` | SQLite |
| Relatórios e interface | `relatorios/graficos.py`, `app/` | Plotly, Streamlit |

O motor de decisão não depende de sensor, banco nem interface, o que permite testá-lo isoladamente e trocar a interface sem tocar na matemática. O projeto tem testes automatizados para o otimizador, o risco, a interação, os estados, os sensores, o cálculo e o banco.

---

## 5. Metodologia dos experimentos

- **Cenários C1 a C6:** instâncias fixas, com as três estratégias (ingênua, gulosa, PLI).
- **Monte Carlo:** 900 instâncias com número de ocorrências, gravidade, probabilidade (Beta), prazo, orçamento $B$ e atenção $K$ sorteados.
- **Sensibilidade:** curva de orçamento (fronteira de Pareto), pesos $\lambda_c$ e $\lambda_f$, fator de prazo e mapa de calor em $(B, K)$.
- **Comparação.** *Ingênua:* trata por ordem de chegada até esgotar recursos. *Gulosa:* ordena por ganho líquido. *PLI:* ótimo do modelo.

---

## 6. Resultados

### 6.1 Sensores e regressão

A regressão logística, treinada com séries simuladas rotuladas, obteve **AUC 0,80** e **acurácia 0,74**. É útil, mas não perfeita: as probabilidades iniciais de perigos reais ficam em torno de 0,4 a 0,5, e a resposta do morador é o que resolve a incerteza restante. O modelo pode ser retreinado com o histórico guardado no SQLite.

Geometricamente, os pontos em que a regressão dá $P(\text{perigo})=\ell$ formam uma **reta** no plano (taxa, aceleração) normalizado, e a distância com sinal de uma leitura a essa reta mede quão longe da dúvida ela está (detalhes na Seção 4.1 do modelo). A figura `08_fronteira_decisao_regressao` mostra as leituras simuladas (alarmes falsos e perigos reais) no plano (taxa, aceleração) e a reta $P=0{,}5$. A reta tem **inclinação negativa**: uma taxa de variação maior compensa uma aceleração menor, e vice-versa, para que a leitura seja classificada como perigo real. Os pontos acima e à direita da reta, do lado positivo da distância com sinal, têm $P>0{,}5$.

A extrapolação linear do prazo se mostrou otimista: com crescimento acelerado, o prazo real foi de 45% a 66% do estimado. Um fator de segurança por sensor (0,55 fogão, 0,35 vazamento, 0,55 sobreaquecimento), calibrado para cobrir cerca de 90% dos casos, corrige isso. **Vale só dentro do modelo simulado.**

### 6.2 Cenários C1 a C6

Valor de $Z$ por estratégia (menor é melhor):

| Cenário | Ingênua | Gulosa | PLI | Ganho do PLI sobre a ingênua | Risco residual (ing. / gul. / PLI) |
|---|---:|---:|---:|---:|---|
| C1 | 20,0 | 20,0 | 20,0 | 0% | 2,6 / 2,6 / 2,6 |
| C2 | 34,2 | 34,2 | 34,2 | 0% | 3,8 / 3,8 / 3,8 |
| C3 | 67,6 | 56,2 | 56,2 | 16,9% | 32,6 / 24,2 / 24,2 |
| C4 | 85,0 | 78,0 | 77,6 | 8,7% | 67,4 / 60,2 / 62,9 |
| C5 | 104,7 | 67,6 | 67,6 | 35,4% | 76,9 / 39,5 / 39,5 |
| C6 | 69,0 | 52,0 | 52,0 | 24,7% | 31,4 / 24,0 / 24,0 |

Leitura:

- **C1 e C2 (recursos sobrando):** as três estratégias empatam. Sem escassez, não há o que otimizar.
- **C3, C5 e C6 (escassez):** a ordem de chegada passa a custar caro, de 17% a 35% a mais em $Z$. Ela gasta o orçamento nas primeiras ocorrências que chegam, mesmo quando outra ocorrência traria mais redução de risco. PLI e gulosa chegam à mesma solução.
- **C4 (as mesmas ocorrências do C3, com orçamento reduzido de 10 para $B=5$):** é o único cenário em que o PLI se separa da gulosa (77,6 contra 78,0). O PLI escolhe uma ação de custo menor (custo 4 contra 5) e aceita um **risco residual maior** (62,9 contra 60,2). Isso não é erro: o PLI minimiza $Z$, e não o risco residual sozinho. Com $\lambda_c = 3$, uma unidade de custo economizada vale 3 em $Z$, mais que os 2,7 de risco que ele deixa de reduzir. Quem quiser priorizar o risco residual deve reduzir $\lambda_c$ (ver sensibilidade).
- **C5 (morador não respondeu a nenhuma consulta):** a atenção efetiva cai a $K=0$, então só ações automáticas ($a_j=0$) são viáveis. A resposta "sem resposta" eleva as probabilidades por Bayes (fogão 0,70 → 0,93; vazamento 0,80 → 0,96; sobreaquecimento 0,60 → 0,90), e a ingênua desperdiça o orçamento: custa 35,4% a mais em $Z$.
- **C6 (morador confirma que está usando o fogão):** a resposta "sim" reduz a probabilidade do fogão de 0,70 para 0,37. Com $p$ baixa, o corte de gás (incômodo 8) passa a ser penalizado como possível falso alarme, e o PLI e a gulosa escolhem apenas **notificar o morador** no fogão, gastando o orçamento no vazamento e no equipamento elétrico. É o efeito da interação com o morador sobre a decisão.

Em C5 PLI e gulosa deixam o vazamento sem ação principal. Esse é o caso em que entra a regra de envelhecimento da gravidade com ação passiva de custo zero (Seção 6.5).

### 6.3 Monte Carlo (900 instâncias)

| Estratégia | Z médio | Risco residual médio | Custo médio |
|---|---:|---:|---:|
| Ingênua | 48,1 | 24,1 | 6,50 |
| Gulosa | 42,5 | 24,6 | 4,97 |
| PLI | 42,3 | 23,6 | 5,14 |

| Comparação | PLI melhor | Empate | PLI pior | Ganho médio de Z | Mediano | p95 | Máximo | Ganho médio quando vence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PLI × ingênua | 59,8% | 40,2% | 0,0% | 11,0% | 5,4% | 40,1% | 79,4% | 18,4% |
| PLI × gulosa | 11,3% | 88,7% | 0,0% | 0,5% | 0,0% | 4,8% | 14,7% | 4,8% |

Pontos principais:

1. **O PLI nunca perde (0,0%)**, como deve ser para um método exato sobre as mesmas instâncias.
2. **Contra a ingênua o ganho é grande e muito desigual:** a mediana é 5,4%, mas o p95 chega a 40,1%. Em 40,2% das instâncias há empate (recursos sobrando); nas outras, o ganho médio é de 18,4%.
3. **Contra a gulosa o ganho médio é só 0,5%,** com empate em 88,7% das instâncias. A gulosa por ganho líquido é uma heurística forte neste problema. O PLI vence em 11,3% dos casos, e nesses o ganho médio é de 4,8% (máximo 14,7%).
4. **O PLI tem o menor risco residual médio** (23,6), mesmo sendo a estratégia que gasta um pouco mais que a gulosa (custo médio 5,14 contra 4,97). A ingênua gasta mais (6,50) e reduz menos risco.

### 6.4 Sensibilidade

As figuras estão em `docs/figuras/`:

- `02_fronteira_pareto`: risco residual × custo conforme o orçamento.
- `03_sensibilidade_prazos`: efeito de encurtar ou alongar os prazos.
- `05_mapa_ganho_vs_gulosa`, `06_mapa_vitorias_vs_gulosa`, `07_mapa_ganho_vs_ingenua`: em que faixas de orçamento $B$ e atenção $K$ o PLI se separa das heurísticas.

Resultados no cenário C3:

- **Curva de orçamento.** O $Z$ do PLI cai de 106,4 ($B=0$) para 53,4 ($B\ge13$), em degraus. Há um patamar em $B=9$ a 12 ($Z=56{,}2$, risco residual 24,2, custo 9) e outro a partir de $B=13$ ($Z=53{,}4$, resíduo 8,6, custo 13, que corresponde a tratar as três ocorrências com ações automáticas, 5+4+4). Passar de 12 para 13 reduz o risco residual em 15,6, mas $Z$ cai só 2,8, porque o custo extra (4 unidades × $\lambda_c=3$) quase anula o ganho: **orçamento adicional deixa de compensar**.
- **O PLI se separa da gulosa em orçamentos intermediários.** Em $B=7$ e $B=8$, o PLI obtém $Z=62{,}8$ contra 70,4 da gulosa (10,8% menor). Em outros orçamentos os dois coincidem ou ficam muito próximos.
- **Comportamento das heurísticas.** A curva do PLI nunca sobe quando o orçamento aumenta. A ingênua piora de $B=7$ para $B=8$ (70,4 → 73,8): com mais dinheiro, a ordem de chegada gasta mal. A gulosa também piora levemente de $B=4$ para $B=5$ (77,6 → 78,0).
- **Pesos $\lambda_c$ e $\lambda_f$.** A decisão é estável numa faixa ampla: a mesma solução (custo 9, resíduo 24,2) vale de $\lambda_c=0$ a 6 e de $\lambda_f=0$ a 6. Só com $\lambda_c=10$ o sistema quase deixa de intervir (custo 1, resíduo 86,5), e com $\lambda_f=12$ o fogão passa a receber só uma notificação (resíduo 62,9). A política muda de forma abrupta quando o custo ou o incômodo ultrapassa o benefício, e por isso a escolha dos pesos é uma decisão de política de segurança.
- **Prazos.** Com prazos pela metade ($\times0{,}5$), $Z$ sobe para 69,35 (resíduo 34,4), porque ações lentas deixam de ser viáveis (R3). Com prazos dobrados, o PLI chega a 44,3 contra 46,4 da gulosa e 52,9 da ingênua. Nesse ponto o resíduo do PLI (18,9) é maior que com $\times1{,}5$ (16,5), de novo a troca entre custo e risco.
- **Mapa $B\times K$ (100 instâncias por célula).** O ganho médio do PLI sobre a gulosa é pequeno em toda a grade (no máximo 1,9%), mas as vitórias chegam a 35% das instâncias. Elas se concentram em **atenção intermediária ($K=2$ e $3$)** e em orçamentos $B=2$ e $B=6$ a 8. Com $K=0$ (só ações automáticas) quase não há diferença, com $B=0$ não há o que decidir, e com $K=4$ a atenção sobra e o ganho cai. Sobre a ingênua o ganho fica entre 9% e 15% para $B\ge4$ (máximo 15,1% em $K=2$, $B=8$) e diminui com orçamento muito folgado ($B=14$: 5% a 11%).

> **Verificação.** Nos dois pontos em que a tabela de sensibilidade mostrava o PLI 0,1 acima da gulosa ($B=3$ e prazos $\times0{,}5$), as duas estratégias escolheram exatamente as mesmas ações, com $Z=84{,}55$ e $Z=69{,}35$. Esses valores ficam no limite entre duas casas decimais, e a diferença aparente era só arredondamento na impressão. Em nenhum ponto da sensibilidade o PLI foi pior que a gulosa.

### 6.5 Ocorrência preterida

Na demonstração com três ocorrências simultâneas e morador sem resposta, o vazamento ficava em ESCALADA sem nenhuma ação, porque o orçamento era consumido pelas outras duas. A regra adotada tem três partes:

1. **Ação passiva:** quem fica de fora recebe `notificar`, de custo zero e sem consumir a atenção do morador.
2. **Envelhecimento:** a gravidade efetiva cresce a cada rodada sem ação, $g^{ef}=\min(10,\,S(1+0{,}25\,k))$, e a ocorrência acaba vencendo a competição por orçamento sem regra especial no PLI.
3. **Prazo crítico:** se o prazo restante cair abaixo de 3 min sem ação, a ocorrência passa a CRÍTICO e aciona o contato de confiança.

Um teste automatizado reproduz exatamente esse cenário.

### 6.6 Interface

- **Tela do morador:** letras grandes, linguagem simples, a pergunta e dois botões ("Sim" e "Preciso de ajuda"), tempo estimado, contato de confiança e o histórico do que aconteceu.
- **Tela técnica:** ocorrências ao vivo, grafo de estados, decisões do otimizador, cenários, sensibilidade, Monte Carlo, mapa $B \times K$, regressão, banco de dados e o modelo matemático.

---

## 7. Discussão

**Onde o PLI importa.** O ganho do PLI aparece quando os recursos são escassos e há interação entre as escolhas (orçamento, atenção e conflitos entre ações). Com recursos sobrando, todas as estratégias empatam. Contra uma heurística gulosa bem escolhida, o PLI melhora pouco em média, mas **garante** a solução ótima, aceita restrições novas sem reescrever a lógica e permite análises de sensibilidade (alterar $B$, $K$ ou $\lambda$ e reotimizar).

**Honestidade sobre a heurística.** O resultado de que a gulosa é quase ótima é um achado, e não uma falha: para este tamanho de problema (no máximo 3 ocorrências, 3 a 4 ações por ocorrência) o espaço de soluções é pequeno. A vantagem do PLI cresceria com mais ocorrências, mais ações e mais restrições acopladas.

**Risco residual × $Z$.** O PLI minimiza $Z$, não o risco residual isolado. Como o C4 mostra, ele pode aceitar um risco residual ligeiramente maior para economizar custo. A escolha de $\lambda_c$ e $\lambda_f$ é, portanto, uma decisão de política de segurança, e não apenas um parâmetro técnico.

**Interação com o morador.** A resposta do morador não é um enfeite: o Bayes reduz ou eleva $p_i$ e muda a decisão (por exemplo, evita cortar o gás de uma panela em uso). A regressão sozinha é incerta (AUC 0,80), e a pergunta resolve parte dessa incerteza.

---

## 8. Limitações

- **Dados sintéticos.** As métricas medem a capacidade de recuperar o que foi simulado, não o desempenho em casas reais.
- **Parâmetros hipotéticos.** Limiares dos sensores, eficácias, custos, verossimilhanças e distribuições do Monte Carlo são hipóteses de simulação.
- **Fatores de segurança do prazo** calibrados em simulação; com dados reais teriam de ser refeitos.
- **Orçamento por rodada.** O orçamento $B$ não se acumula entre rodadas do otimizador: uma ocorrência que escala depois das outras é decidida com orçamento novo.
- **Uma ação principal por ocorrência (R1)** simplifica o efeito combinado de várias ações.
- **Porte do problema.** Com 3 ocorrências a comparação com heurísticas tem pouco espaço para diferença.
- O modelo prioriza e recomenda; **não garante** segurança física.

---

## 9. Conclusão e trabalhos futuros

O projeto entregou um motor de decisão completo, do sensor à ação, com modelagem matemática explícita, implementação em Python, simulações em múltiplos cenários e uma interface que torna as decisões interpretáveis. O principal resultado é uma comparação honesta: o PLI nunca é pior, supera de forma expressiva a ordem de chegada (11,0% em média, até 79,4%), e entrega apenas um ganho marginal sobre uma gulosa bem projetada (0,5% em média), com vantagem concentrada em cenários de escassez.

Trabalhos futuros:

- carregar o orçamento restante entre rodadas;
- aumentar o número de ocorrências e de ações para testar a escala;
- validar parâmetros e fatores de segurança com dados reais (sensores ESP32);
- modelar o efeito combinado de várias ações (relaxar R1);
- integração com os demais módulos da Smart Home e interação por voz.

---

## 10. Cobertura dos requisitos da disciplina

| Requisito | Onde está |
|---|---|
| Modelagem matemática | `modelo_matematico.md` (equações, restrições, variáveis) |
| Implementação em Python | NumPy, SciPy, SymPy e PuLP em `src/home_guardian/` |
| Otimização e simulação | PLI, cenários C1 a C6, Monte Carlo, sensibilidade |
| Interpretação | gráficos em `docs/figuras/`, `app/tela_tecnica.py`, este relatório |
| Operations Research | programação linear inteira binária, mochila multidimensional |
| Geometria analítica e Álgebra linear | fronteira de decisão da regressão como reta e distância com sinal (`modelo/geometria.py`); matriz de eficácia, máscara de viabilidade, matriz de adjacência |
| Matemática discreta | máquina de estados, alcançabilidade, conexidade forte |
| Fundamentos de Cálculo | derivada (taxa e prazo) e integral (risco acumulado) |
| MySQL / SQLite | `persistencia/db.py` (SQLite) |
| Matplotlib / Plotly | `relatorios/graficos.py` (Plotly) |