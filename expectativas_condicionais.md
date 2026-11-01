# Síntese crítica: expectativas condicionais de preços sob GBM neutro ao risco

## 1. Escopo e hipóteses

Comparação dos arquivos `conditional_expectation_chatgpt.md` e `conditional_expectation_claude.md`. Ambos consideram movimento browniano geométrico (GBM) sob a medida neutra ao risco Q, taxa livre de risco e volatilidade constantes. O primeiro incorpora desde o início um *dividend yield* contínuo `q`; o segundo começa com `q=0` e instrui a substituir `r` por `r-q` quando houver dividendos contínuos. Portanto, **não há contradição** entre as expressões.

O modelo consolidado é

$$\frac{dS_t}{S_t}=(r-q)\,dt+\sigma\,dW_t^{\mathbb Q},$$

com $S_0>0$, $K>0$, $T>0$, $\sigma>0$, e $r,q$ expressos como taxas contínuas anuais. $T$ é medido em anos. Dividendos discretos em dinheiro **não** estão modelados.

## 2. Convergências

1. A solução exata é $S_T=S_0\exp[(r-q-\sigma^2/2)T+\sigma\sqrt T Z]$, $Z\sim N(0,1)$.
2. Ambos identificam as probabilidades $\mathbb Q(S_T>K)=\Phi(d_2)$ e $\mathbb Q(S_T<K)=\Phi(-d_2)$.
3. Ambos calculam os momentos parciais $\mathbb E[S_T\mathbf 1_{S_T>K}]=S_0e^{(r-q)T}\Phi(d_1)$ e $\mathbb E[S_T\mathbf 1_{S_T<K}]=S_0e^{(r-q)T}\Phi(-d_1)$.
4. Ambos chegam às mesmas expectativas condicionais e verificam a lei da expectativa total.

## 3. Diferenças e avaliação crítica

| Aspecto | ChatGPT | Claude | Síntese crítica |
|---|---|---|---|
| Dividendos | Inclui $q$ explicitamente em todas as etapas | Desenvolve com $q=0$ e indica substituição $r\to r-q$ | Preferir $q$ explícito na API e nas fórmulas. |
| Derivação | Integra a densidade normal de $X=\ln S_T$ | Integra a densidade de $Z$ usando lema de completar quadrados | Ambas corretas; a segunda evidencia a translação da normal, a primeira evidencia a lognormalidade. |
| Interpretação | Ressalta expectativa neutra ao risco versus previsão real e preço descontado | Acrescenta interpretação de mudança de medida (numerário da ação) | São perspectivas complementares. Com dividendos contínuos, a interpretação de numerário exige o tratamento do ativo total-return/reinvestimento, não simplesmente tomar o preço ex-dividendo como numerário sem ajuste. |
| Testes | Lei da expectativa total | Lei da expectativa total, ordenação, limites e exemplo | Combinar os testes. |
| Opções | Não desenvolve Black–Scholes | Relaciona momentos truncados a calls e puts | Válido, mas atualizar os preços das opções com dividendos. |
| Valores numéricos | Sem exemplo | Exemplo $S_0=K=100$, $r=5\%$, $\sigma=20\%$, $T=1$, $q=0$ | Útil como teste aproximado. |

**Observação:** a afirmação de Claude sobre o numerário é adequada como intuição de mudança de medida no caso sem dividendos; na extensão com dividendos, deve-se considerar a estratégia de reinvestimento dos dividendos. A conexão com Black–Scholes também requer o fator $e^{-qT}$ quando $q\ne0$.

## 4. Derivação consolidada

Seja $m=\ln S_0+(r-q-\sigma^2/2)T$ e $v=\sigma\sqrt T$. Então $X=\ln S_T\sim N(m,v^2)$ e

$$\mathbb E^{\mathbb Q}[S_T]=e^{m+v^2/2}=S_0e^{(r-q)T}=:F.$$

Defina

$$d_2=\frac{\ln(S_0/K)+(r-q-\sigma^2/2)T}{\sigma\sqrt T},\qquad d_1=d_2+\sigma\sqrt T.$$

Por padronização, $P_>=\Phi(d_2)$ e $P_<=\Phi(-d_2)$. Completando o quadrado no integrando $e^x$ multiplicado pela densidade normal de $X$:

$$\mathbb E[S_T\mathbf 1_{S_T>K}]=F\Phi(d_1),\qquad \mathbb E[S_T\mathbf 1_{S_T<K}]=F\Phi(-d_1).$$

Dividindo pelas probabilidades dos eventos, obtemos

$$\boxed{\mathbb E^{\mathbb Q}[S_T\mid S_T>K]=F\frac{\Phi(d_1)}{\Phi(d_2)}}$$

$$\boxed{\mathbb E^{\mathbb Q}[S_T\mid S_T<K]=F\frac{\Phi(-d_1)}{\Phi(-d_2)}}$$

Como a distribuição é contínua, $P(S_T=K)=0$. A lei da expectativa total é

$$P_>E_>+P_<E_<=F.$$

A relação correta com Black–Scholes, para dividendos contínuos, é

$$C=S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2),$$

$$P=Ke^{-rT}\Phi(-d_2)-S_0e^{-qT}\Phi(-d_1).$$

As expectativas condicionais são **preços terminais esperados**, não valores presentes nem previsões sob a medida física. Para projeção sob a medida física, seria necessário especificar a deriva física e assumir que o GBM é adequado.

## 5. Implementação Python

A função retorna ambos os valores esperados e as probabilidades condicionantes. Usa `log_ndtr` para evitar divisões instáveis entre probabilidades muito pequenas. Não é necessário simular trajetórias nem escolher passo temporal para calcular esses resultados.

```python
import math
from scipy.special import log_ndtr


def conditional_stock_expectations(S0, K, T, r, sigma, q=0.0):
    """Expectativas condicionais de S_T sob GBM neutro ao risco.

    Parâmetros
    ----------
    S0 : float
        Preço atual ex-dividendo (> 0).
    K : float
        Threshold do preço terminal (> 0).
    T : float
        Horizonte em anos (> 0).
    r : float
        Taxa livre de risco anual continuamente composta.
    sigma : float
        Volatilidade anual (> 0).
    q : float, optional
        Dividend yield contínuo anual (default 0).

    Retorna
    -------
    dict
        expected_above, expected_below, probability_above,
        probability_below, unconditional_mean, d1, d2.

    Notas
    -----
    Para eventos de probabilidade numericamente desprezível,
    a expectativa é calculada com razão de CDFs em log.
    Valores extremos podem exceder o intervalo de floats.
    """
    vals = (S0, K, T, r, sigma, q)
    if not all(math.isfinite(float(x)) for x in vals):
        raise ValueError("Todos os parâmetros devem ser finitos.")
    S0, K, T, r, sigma, q = map(float, vals)
    if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S0, K, T e sigma devem ser positivos.")

    v = sigma * math.sqrt(T)
    d2 = (math.log(S0 / K) + (r - q - 0.5 * sigma**2) * T) / v
    d1 = d2 + v
    log_forward = math.log(S0) + (r - q) * T

    # Razões de probabilidades avaliadas no domínio logarítmico.
    log_above = log_forward + log_ndtr(d1) - log_ndtr(d2)
    log_below = log_forward + log_ndtr(-d1) - log_ndtr(-d2)

    return {
        "expected_above": math.exp(log_above),
        "expected_below": math.exp(log_below),
        "probability_above": math.exp(log_ndtr(d2)),
        "probability_below": math.exp(log_ndtr(-d2)),
        "unconditional_mean": math.exp(log_forward),
        "d1": d1,
        "d2": d2,
    }
```

### Exemplo e verificações

```python
result = conditional_stock_expectations(
    S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0
)
for key, value in result.items():
    print(f"{key}: {value:.8f}")

assert result["expected_above"] > 100.0
assert result["expected_below"] < 100.0
assert math.isclose(
    result["probability_above"] * result["expected_above"]
    + result["probability_below"] * result["expected_below"],
    result["unconditional_mean"], rel_tol=1e-12
)

# Teste com dividendos contínuos:
with_dividends = conditional_stock_expectations(100, 100, 1, 0.05, 0.20, q=0.02)
assert math.isclose(
    with_dividends["unconditional_mean"], 100 * math.exp(0.03),
    rel_tol=1e-12
)
```

No caso sem dividendos acima, as expectativas são aproximadamente **119,6** para $S_T>100$ e **86,7** para $S_T<100$, em consonância com o exemplo de Claude.

### Limitações práticas

- As expressões pressupõem GBM lognormal, volatilidade e taxas constantes, dividendos proporcionais contínuos e medida neutra ao risco.
- Para `T=0` ou `sigma=0`, a distribuição é degenerada e os eventos condicionantes podem ter probabilidade zero; a função rejeita esses casos.
- Em thresholds extremamente distantes, as probabilidades retornadas podem arredondar para zero, embora as expectativas possam continuar numericamente calculáveis; não interpretar uma expectativa de evento praticamente impossível como uma previsão operacional robusta.
- Os cálculos são analíticos e não dependem de discretização de Monte Carlo.

## 6. Conclusão

Os documentos são consistentes nas expressões principais. A melhor síntese preserva o dividend yield explícito do ChatGPT e a exposição adicional de Claude sobre o lema gaussiano, testes e precificação de opções, com as correções necessárias para a extensão com dividendos. A função apresentada implementa as duas expectativas condicionais com estabilidade numérica superior à divisão direta de CDFs.
