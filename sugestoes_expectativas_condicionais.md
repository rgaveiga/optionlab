# Sugestões: expectativas condicionais Black-Scholes

Análise de 01/11/2026. Apenas este documento deve ser salvo; nenhum snippet foi aplicado. Fonte: expectativas_condicionais.md.

## Conclusão e adequação mínima

O código atual JÁ usa os momentos analíticos propostos no documento. Portanto não há alteração funcional necessária nem novos valores de expected return a colocar nos asserts. A melhoria opcional mínima é centralizar o integrador duplicado de support.py em profit.integrate, sem criar funções novas.

O documento calcula preço condicionado da ação; OptionLab calcula lucro/prejuízo condicionado da estratégia. Não substituir um pelo outro. Para uma ação comprada a 100, subtrair 100 do preço condicionado; para opções e spreads, integrar cada trecho do payoff. O target padrão é 0.01, não zero. Preservar custos nominais, comissões e quantidades, sem desconto adicional.

## Demonstração e fluxo

Defina m=log(S0)+(r-q-sigma²/2)*T, v=sigma*sqrt(T), F=exp(m+v²/2). Para L<S_T<U, com zL=(log(L)-m)/v e zU=(log(U)-m)/v:

    P = Phi(zU)-Phi(zL)
    M = F * [Phi(zU-v)-Phi(zL-v)]
    E[(a*S_T+b)*1_intervalo] = a*M+b*P

profit.integrate calcula exatamente P e a*M+b*P. Para L=K e U=infinito, zL=-d2 e zL-v=-d1: M/P é F*Phi(d1)/Phi(d2), a fórmula do documento. O intervalo inferior dá a outra fórmula.

support.get_pop para BS e _compute_expected_returns_bs delegam a profile_pop(array_segments(...)). partition divide os segmentos no target; event_integrals soma probabilidades e momentos; profile_pop divide a soma dos momentos pela probabilidade do evento. Não tirar média simples de médias por segmento. dividend_yield já entra como q.

support._integrate_linear_profit_bs duplica essa fórmula, mas não é usado pelo fluxo de profile_pop. engine._strategy_pop usa strikes reais e segmentos independentes do domínio do gráfico para estratégias no vencimento. Antes do vencimento, adaptive_segments aproxima o perfil curvo: a integração de cada trecho é analítica, mas permanece erro de interpolação limitado a 0.0005 na média condicional e 1e-9 na probabilidade. As fórmulas do documento não eliminam essa aproximação para payoffs curvos.

Arrays fornecidos diretamente definem interpolação e extrapolação pelas secantes; strikes ausentes não podem ser reconstruídos. Tempo/volatilidade zero já são tratados deterministicamente em profile_pop. Preservar a convenção de zero para eventos vazios.

## Snippet opcional: reutilizar integrate

Adicionar integrate ao import de optionlab.profit em support.py. Manter assinatura e substituir apenas o corpo de _integrate_linear_profit_bs:

```python
    lower, upper, a, b = np.broadcast_arrays(
        lower_price, upper_price, slope, intercept
    )
    segments = np.column_stack((
        np.maximum(lower, 0.0).ravel(),
        np.maximum(upper, 0.0).ravel(),
        a.ravel(), b.ravel(),
    ))
    probability, moment = integrate(segments, log_mean, sigma)
    return probability.reshape(lower.shape), moment.reshape(lower.shape)
```

O sigma dessa assinatura é v (desvio do log-preço no horizonte), não volatilidade anual. maximum preserva o tratamento anterior de limites negativos como zero. Não importar support em profit: criaria ciclo. Remover cast e normal_mass do import de support somente se ficaram sem uso. Manter _lognormal_z evita ampliar o escopo. Não modificar engine.py.

Opcionalmente explicitar na docstring de profit.integrate:

```python
    """Integrate affine P/L with exact truncated lognormal moments.

    exp(m + v*v/2) is S0*exp((r-q)*T). Shifting z by -v gives
    the d1 moment; unshifted z gives the d2 probability.
    Conditional means are formed after summing event moments.
    """
```

normal_mass já evita cancelamento na cauda direita usando CDFs complementares. Não resolve todo underflow: probabilidades extremamente pequenas ainda podem virar zero. A versão log_ndtr do documento mantém certas razões representáveis nesses casos. Adaptá-la a intervalos e momentos com sinais exige diferenças estáveis de CDFs e agregação também estável; alterar somente a divisão final não recupera momentos perdidos. Tratar essa robustez extrema como melhoria separada, não como requisito para tornar analítico um cálculo que já é analítico.

## Asserts e validação para o implementador

Em tests/test_profit.py::test_complete_profile_analytic_reference, preservar os valores atuais:

| Caso | Probabilidade | Expected profit | Expected loss |
|---|---:|---:|---:|
| Ação comprada | 0.5594204551783997 | 19.638887641345562 | -13.299041897846353 |
| Call comprada K=100, prêmio=5 | 0.4623849888709348 | 18.237215774795903 | -4.550130321375573 |

Valores lidos dos testes, não recalculados nesta tarefa. Parâmetros: S0=100, r=0.05, sigma=0.2, T=1, q=0, target=0.01. Os 119.6 e 86.7 aproximados do documento são preços condicionais com threshold 100; não são substitutos desses retornos.

Para verificar diretamente a equivalência, acrescentar opcionalmente em tests/test_profit.py, reaproveitando imports e MODEL:

```python
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_stock_conditional_formula(q):
    model = MODEL.model_copy(update={"dividend_yield": q})
    target = 0.01
    threshold = 100.0 + target
    d2 = (np.log(100.0 / threshold) + 0.05 - q - 0.2**2 / 2) / 0.2
    d1 = d2 + 0.2
    forward = 100.0 * np.exp(0.05 - q)
    pa, pb = norm.cdf(d2), norm.cdf(-d2)
    ea = forward * norm.cdf(d1) / pa - 100.0
    eb = forward * norm.cdf(-d1) / pb - 100.0
    out = get_pop(
        np.array([50.0, 150.0]), np.array([-50.0, 50.0]), model, target
    )
    assert out.expected_return_above_target == pytest.approx(ea, abs=1e-10, rel=0)
    assert out.expected_return_below_target == pytest.approx(eb, abs=1e-10, rel=0)
    assert (
        pa * out.expected_return_above_target
        + pb * out.expected_return_below_target
    ) == pytest.approx(forward - 100.0, abs=1e-10, rel=0)
```

Preservar tests/profit_reference.py: usa quadratura independente, não os integradores de produção. tests/test_core.py já exclui retornos dos snapshots e usa essa referência, com tolerância 0.00051 para acomodar interpolação. tests/test_support.py verifica sinais e cálculos desabilitados; os asserts 1.5 e -2.0 são de ArrayInputs e não devem mudar. Preservar testes de cauda, eventos desconexos, degenerescência e platôs.

Após eventual implementação:

```text
python -m pytest tests/test_profit.py tests/test_support.py tests/test_core.py tests/test_engine.py
```

Não atualizar asserts automaticamente para aceitar saídas novas. Se houver falha, identificar a mudança e conferir uma referência independente. A premissa de que haverá novos expected returns não se confirma no código analisado. Nenhum teste foi executado nesta tarefa documental; os snippets são sugestões para revisão e implementação posterior.
