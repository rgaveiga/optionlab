# Plano de correção estatística da PoP e das expectativas

## Objetivo e escopo

Corrigir as inconsistências em `optionlab/support.py` e sua integração com `optionlab/engine.py`, preservando as fórmulas que já estão corretas. Este documento orienta um agente de código; sua criação não autoriza alterações adicionais nesta etapa.

Antes de implementar, leia as instruções aplicáveis do repositório e confira a versão atual dos arquivos. Preserve alterações do usuário. Não atualize valores esperados dos testes apenas para fazê-los passar: justifique cada mudança por uma referência analítica ou numérica independente.

## Diagnóstico que deve orientar a implementação

- A distribuição lognormal sob a medida neutra ao risco está correta: `m = log(S0) + (r - q - sigma**2 / 2) * T` e `v = sigma * sqrt(T)`.
- `_integrate_linear_profit_bs` integra corretamente um perfil linear por trechos contra essa distribuição, inclusive o primeiro momento truncado.
- As expectativas são condicionais. Dividir o resultado ponderado pela probabilidade do evento é correto e deve ser preservado.
- `_get_profit_range` deixa um intervalo sem classificação entre pontos consecutivos em que o perfil cruza o alvo. Por isso, as probabilidades complementares podem somar menos de um.
- A PoP prolonga a classificação das extremidades, enquanto as expectativas prolongam suas inclinações. Essas regras diferentes podem produzir resultados incompatíveis; ambas podem representar incorretamente a estratégia fora da malha.
- Prazo zero é permitido por `BlackScholesModelInputs`, mas provoca divisão por zero.
- Os testes existentes de soma das probabilidades têm tolerância larga demais para detectar o problema na malha fina.

## 1. Unificar a representação dos eventos e a integração

Crie uma representação interna de segmentos do perfil, com limites inferior e superior, inclinação e intercepto. Use a mesma representação para PoP, expectativas e intervalos públicos de resultado.

Para cada segmento `g(s) = a*s + b` e alvo `c`:

1. Se `a != 0`, calcule o cruzamento `x = (c - b) / a`.
2. Se `x` estiver estritamente dentro do segmento, divida-o nesse ponto.
3. Classifique cada segmento resultante como `g(s) >= c` ou `g(s) < c`.
4. Use o mesmo limite de cruzamento nos segmentos adjacentes. Não deixe o espaço entre dois preços da malha sem classificação.
5. Integre as probabilidades e os resultados ponderados de cada grupo.
6. Divida cada resultado ponderado pela probabilidade do mesmo grupo para obter a expectativa condicional.

Use o momento truncado já implementado:

```text
P(L < S_T < U) = Phi(z_U) - Phi(z_L)
z_x = (log(x) - m) / v

E[S_T * 1_{L < S_T < U}]
    = exp(m + v²/2) * [Phi(z_U - v) - Phi(z_L - v)]

E[(a*S_T + b) * 1_{L < S_T < U}]
    = a * E[S_T * 1_{L < S_T < U}] + b * P(L < S_T < U)
```

Limites zero e infinito devem ser tratados explicitamente. Una segmentos adjacentes da mesma classificação ao construir os intervalos públicos.

Não corrija a soma simplesmente normalizando as probabilidades atuais, nem usando `1 - PoP` sobre uma PoP calculada com limites errados: isso esconderia a perda de massa sem corrigir o evento.

Elimine o deslocamento indiscriminado `profit - target + 1e-10` como definição do evento. A convenção é `>= target`; tolerâncias eventualmente necessárias para raízes numéricas devem ser locais, documentadas e coerentes com a precisão do resultado. Um trecho constante exatamente no alvo pertence integralmente ao evento de sucesso, pois pode carregar probabilidade positiva.

## 2. Representar corretamente o perfil fora da malha

### Estratégias avaliadas no vencimento

Solução preferida: construir o perfil linear por trechos a partir das pernas da estratégia, em vez de inferir seu comportamento global a partir da malha de visualização.

- Reúna todos os strikes das opções relevantes, além de zero e infinito como limites externos.
- Construa os coeficientes de cada trecho a partir dos payoffs de calls, puts, ações e posições fechadas.
- Preserve quantidades, sinais de compra/venda, comissões e o tratamento existente de posições anteriores. Reaproveite a lógica existente para evitar duas definições divergentes do P/L.
- Inclua todos os strikes mesmo quando estiverem fora de `min_stock` e `max_stock` ou entre pontos da malha.
- Calcule cruzamentos, probabilidades e expectativas sobre esse perfil completo.

Com isso, `min_stock`, `max_stock` e `price_step` continuam controlando o domínio de visualização, mas não alteram as métricas globais no vencimento. Preserve os significados de `minimum_return_in_the_domain` e `maximum_return_in_the_domain`: eles continuam restritos ao domínio solicitado.

### Estratégias avaliadas antes do vencimento

O perfil de uma opção ainda viva é curvo. Não prometa integração exata usando segmentos lineares, nem prolongue indefinidamente uma secante arbitrária da malha.

Solução preferida: disponibilizar um avaliador do P/L da estratégia em qualquer preço positivo, com a mesma lógica das pernas usada pelo motor. Integrar sob a normal padronizada:

```text
s(z) = exp(m + v*z)
P_A = integral 1_{g(s(z)) >= c} * phi(z) dz
M_A = integral g(s(z)) * 1_{g(s(z)) >= c} * phi(z) dz
```

Localize os cruzamentos e divida a integração nos mesmos limites para probabilidades e momentos. Use quadratura adaptativa para os momentos e, uma vez conhecidos os limites dos eventos, diferenças de CDF para suas probabilidades.

A busca de raízes deve contemplar múltiplos cruzamentos, regiões estreitas e tangências; uma busca fixa baseada apenas em trocas de sinal não garante encontrar todos os eventos. Uma tangência isolada não altera a probabilidade para `v > 0`, mas um trecho constante no alvo altera. Use subdivisão adaptativa, avaliação de extremos quando disponível e verificações de convergência. Se o algoritmo não puder estabelecer convergência, informe a limitação em vez de retornar silenciosamente um valor supostamente preciso.

É aceitável implementar inicialmente uma malha interna adaptativa, independente da malha pública, desde que haja controle explícito de erro para probabilidades e expectativas e tratamento das caudas. Controle apenas da probabilidade omitida não basta para payoffs ilimitados: controle também o primeiro momento truncado e um limite para o crescimento absoluto do P/L. A fórmula de momento truncado já disponível pode ajudar a limitar esse erro. Antes de dividir pelo peso de um evento raro, considere que o erro do numerador será amplificado por essa divisão.

### Compatibilidade dos helpers

`get_pop(s, profit, ...)` sozinho não possui informação suficiente para reconstruir strikes ou curvaturas ausentes da malha. Não tente resolver essa falta de informação apenas com heurísticas.

Introduza um caminho interno que receba segmentos completos ou um avaliador da estratégia, usado pelo motor. Se mantiver o caminho baseado somente em arrays, documente explicitamente seu modelo de interpolação/extrapolação e garanta que PoP e expectativas usem o mesmo perfil. Nesse caminho, não prometa exatidão para uma estratégia desconhecida fora do domínio fornecido.

## 3. Tratar distribuições degeneradas diretamente

Antes de qualquer divisão por `v`, trate:

- `T == 0`: `S_T = S0`.
- `sigma == 0` com `T > 0`, caso suportado: `S_T = S0 * exp((r - q)*T)`.

Avalie o P/L nesse preço determinístico e compare diretamente com o alvo. A probabilidade do evento verdadeiro é 1 e a do outro é 0. A expectativa condicional do evento verdadeiro é o próprio P/L.

Para o evento de probabilidade zero, a esperança condicional não está definida. Preserve inicialmente o retorno `0.0` por compatibilidade, mas documente que é uma convenção de ausência de evento, não uma expectativa matemática.

Há uma inconsistência de validação a resolver: `Inputs.volatility` aceita zero, mas `TheoreticalModelInputs.volatility` exige valor positivo. Recomenda-se suportar zero no caminho Black–Scholes de forma consistente. Verifique o impacto sobre os demais modelos antes de alterar a classe-base e sobre precificação/greeks antes de prometer suporte em `run_strategy` com todos os cálculos habilitados. Não substitua o caso determinístico por uma volatilidade artificial de `1e-10`.

## 4. Estabilidade numérica e convenções

- Para intervalos na cauda direita, calcule a diferença por simetria, `Phi(-z_L) - Phi(-z_U)`, evitando subtrair dois valores próximos de um. Aplique o mesmo cuidado ao primeiro momento truncado. Para eventos extremos, considere operações em logaritmos; teste os limites de precisão declarados.
- Não use clipping amplo ou renormalização para esconder erros de integração. Eventuais ajustes de arredondamento devem ser limitados à precisão de máquina.
- Mantenha precisão integral durante os cálculos e arredonde apenas os valores monetários de saída, conforme o contrato atual.
- Valide arrays vazios, incompatíveis ou com preços repetidos/não crescentes no caminho que depende de interpolação.
- Documente a convenção para uma malha de um único ponto; ela não define um perfil global sem hipótese adicional.

## 5. Documentar o significado econômico sem mudar silenciosamente a API

- As métricas Black–Scholes são calculadas sob a medida neutra ao risco. Não apresentá-las como estimativas da frequência real de lucro sem um modelo físico adicional.
- Preserve o alvo padrão `0.01`. Ele define sucesso como P/L de pelo menos um centavo, e não estritamente maior que zero. O evento complementar pode incluir equilíbrio e pequenos ganhos.
- Preserve os nomes condicionais públicos. Não substitua expectativas condicionais por contribuições incondicionais `E[g*1_A]`.
- O P/L atual subtrai custos/prêmios nominais, sem capitalizá-los até a data-alvo. Isso é uma convenção de resultado nominal, não um erro na integração. Não introduza desconto ou financiamento silenciosamente.
- Para pernas de ações, dividendos recebidos não se confundem com o dividend yield usado na distribuição do preço ex-dividendos. Documente se os fluxos de dividendos fazem parte ou não do perfil atual.
- Uma futura opção de resultado com financiamento/dividendos ou drift físico é uma ampliação de escopo, não pré-requisito para corrigir os erros aqui descritos.

## 6. Testes de regressão e referências independentes

Use como caso-base `S0=100`, `r=0.05`, `q=0`, `sigma=0.20`, `T=1` e alvo `c=0.01`. Valores abaixo foram conferidos na análise por quadratura independente, com o perfil nominal atual.

| Perfil | PoP correta | Lucro condicional | Resultado condicional abaixo do alvo |
| --- | --- | --- | --- |
| Uma ação: `g(s)=s-100` | 0.5594204551783997 | 19.638887641345605 | -13.29904189784633 |
| Call comprada: `g(s)=max(s-100,0)-5` | 0.4623849888709348 | 18.23721577479595 | -4.550130321375576 |

As saídas monetárias atuais devem permanecer arredondadas a centavos. Recalcule referências no teste por fórmulas independentes quando possível; não reutilize os helpers sob teste como oráculo.

### Regressões obrigatórias

1. **Partição da probabilidade:** verificar `P(A) + P(Ac) = 1` com tolerância absoluta próxima de `1e-12` para os casos analíticos ordinários. Testar cruzamentos dentro da célula, exatamente nos nós e múltiplos cruzamentos. A tolerância atual `1e-3` é insuficiente.
2. **Malha grossa:** ação do caso-base com domínio 50 a 150 e passo 1. Atualmente a soma é aproximadamente `0.980308725153499`. Deve usar o cruzamento 100.01 e recuperar a PoP analítica.
3. **Cruzamento fora da malha pública:** ação com domínio 110 a 120. Atualmente a PoP é 1. No caminho do motor com perfil completo, deve ser a mesma PoP do caso-base.
4. **Strike fora da malha pública:** call do caso-base com domínio 50 a 90. Atualmente retorna PoP 0 e perda condicional -5. No motor, deve recuperar a referência da tabela.
5. **Strikes entre nós:** spreads, straddles e butterflies com strikes não alinhados ao passo. Verificar todos os segmentos e comparar com integração independente do payoff real.
6. **Identidade da esperança total:** antes do arredondamento, verificar `E[g] = P(A)*E[g|A] + P(Ac)*E[g|Ac]`, com a convenção apropriada para eventos impossíveis. Para a ação, a referência é `100*(exp(0.05)-1)`. Para a call, é `exp(r*T)*C_BS - 5`, calculando `C_BS` de forma independente. Na API arredondada, admitir o erro monetário correspondente, em vez de exigir igualdade de máquina.
7. **Prazo zero:** com `S0=100`, `g(s)=s-100` e alvo zero, PoP deve ser 1. Com alvo 0.01, deve ser 0. Não pode haver `NaN` ou avisos de divisão por zero.
8. **Volatilidade zero:** preço determinístico `S0*exp((r-q)*T)`, inclusive igualdade exata com o alvo. Testar a interface pública se esse suporte for implementado nela.
9. **Eventos especiais:** perfil constante acima, abaixo e exatamente no alvo; eventos de probabilidade zero; regiões lucrativas desconectadas e um zero isolado do perfil. Não descartar massa de trechos constantes.
10. **Antes do vencimento:** comparar contra quadratura independente do P/L precificado em cada preço, com dividend yield não nulo e pernas com diferentes vencimentos. Variar a malha pública e verificar estabilidade dentro da precisão declarada.
11. **Caudas:** eventos na cauda direita, inclusive intervalos cujas CDFs usuais arredondam ambas para 1. Verificar que probabilidades representáveis não sejam artificialmente zeradas e que momentos/expectativas tenham sinais e magnitudes coerentes.
12. **Compatibilidade:** preservar o caminho `ArrayInputs`, os controles `calculations`, comissões, quantidades e posições anteriores. Não aplicar a correção de intervalos analíticos à média empírica, que já classifica cada amostra por `>= target`.

Monte referências por fórmulas fechadas ou quadratura dividida nos strikes e cruzamentos; Monte Carlo pode ser uma verificação adicional, mas não o único oráculo devido ao ruído amostral.

## 7. Sequência sugerida e critérios de conclusão

1. Acrescente regressões que reproduzam as falhas atuais e referências independentes para os cálculos já corretos.
2. Centralize a partição em segmentos e reutilize-a para probabilidades e expectativas.
3. Implemente o perfil completo no vencimento e conecte-o ao motor.
4. Implemente o tratamento determinístico e a estabilidade das caudas.
5. Implemente a avaliação anterior ao vencimento com controle de erro e comportamento explícito quando não houver convergência.
6. Atualize documentação e valores de regressão afetados, justificando as diferenças.
7. Execute os testes relevantes e a suíte existente de acordo com as instruções do projeto.

O trabalho estará concluído quando os eventos formarem uma partição consistente, probabilidades e expectativas usarem o mesmo perfil, as métricas globais do motor não dependerem do domínio de visualização, os casos determinísticos forem tratados sem divisão por zero e a precisão da aproximação anterior ao vencimento estiver demonstrada. Se alguma etapa permanecer pendente, declare-a explicitamente; não apresente uma correção parcial como solução completa.

Na entrega, informe arquivos alterados, testes executados, mudanças nos resultados e limitações numéricas remanescentes. Não faça commit, push ou publicação sem autorização correspondente.

## Referência conceitual

Martin Haugh, Columbia University, *The Black-Scholes Model*: https://www.columbia.edu/~mh2078/FoundationsFE/BlackScholes.pdf — dinâmica lognormal, medida neutra ao risco e precificação por expectativa descontada. As falhas específicas e os exemplos numéricos acima provêm da inspeção e execução do código local.
