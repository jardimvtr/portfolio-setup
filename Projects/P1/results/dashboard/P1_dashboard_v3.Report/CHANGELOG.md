# P1_dashboard_v3 — changelog

Cópia incremental de `P1_dashboard` (Report + SemanticModel), criada para incorporar os
resultados do Script 07.3 (region vs country) sem alterar o projeto original nem os
ajustes manuais do autor. Ver `README_APLICAR.md` para os passos a seguir no Power BI
Desktop.

## 2026-09-22 — v3.0.0

**Modelo semântico** (`P1_dashboard_v3.SemanticModel`)
- Adicionadas 3 tabelas novas, importadas de `outputs/07.3_region_vs_country.xlsx`:
  `RVC_Variance` (14 linhas), `RVC_Region` (512 linhas), `RVC_Country` (28 linhas).
- Adicionados 2 relacionamentos: `RVC_Region[region_id] → Dim_Region[region_id]` e
  `RVC_Country[code] → Dim_Country[code]`, ambos muitos-para-um, direção única.
  `RVC_Variance` permanece sem relacionamento (conforme especificação).
- Adicionadas 11 medidas DAX em `Fact_Region`, pasta `Dashboard\Region vs country`:
  `RVC country share · profile`, `RVC country share · movement`, `RVC country share`,
  `RVC within share`, `RVC nearest region abroad`, `RVC share nearest region abroad`,
  `RVC differs from country majority`, `RVC countries with both directions`,
  `RVC foreign share k5`, `RVC baseline foreign share`, `RVC most similar region (text)`.
- Nenhuma tabela, relacionamento ou medida existente foi alterado.

**Relatório** (`P1_dashboard_v3.Report`)
- Nova página **"Region vs country"**, inserida logo após "Country context" na
  navegação (`pages.json`). Página automática (o visual `pageNavigator` já existente
  detecta a nova página sozinho, sem precisar de botões manuais).
  - Faixa de 4 KPIs: participação do país na variação do perfil (~70%), na variação do
    movimento (~12%), regiões com região mais parecida no exterior (179 · ~35%) e
    regiões que divergiriam do perfil majoritário do país (37).
  - Gráfico esquerdo: barras 100% empilhadas por variável (país explicado vs. dentro do
    país), ordenado por participação do país, com epsilon², p de permutação e grupo da
    variável no tooltip.
  - Gráfico direito: barras por país da fração de vizinhos estrangeiros entre os 5 mais
    parecidos, com linha de referência constante em 0,92 (esperado se país não
    importasse) e tooltip com a linha de base do país e o número de regiões.
  - Nenhuma página existente foi apagada, reordenada ou reformatada.
- Página de detalhe da região (`dt_region_profile`, cartão `dt_facts`): adicionados os
  campos **"Most similar region"** e **"Share of 5 most similar regions abroad"** (2
  novos blocos no cartão existente; `maxTiles` ajustado de 1 para 3 para acomodar os 5
  blocos sem estourar a área do cartão).
- Página de tooltip do mapa (`tt_region`) **não foi alterada**: o popup de 320×210px já
  está no limite do espaço disponível com os blocos existentes; adicionar mais campos
  ali arrisca cortar o conteúdo. Ver pendência no `README_APLICAR.md`.
- Mapa "most similar region: same country vs abroad" (item opcional B.3.8 da
  especificação) **não foi implementado** — pendência registrada, ver README.

**Validação**
- Todo o JSON novo/alterado é sintaticamente válido e foi validado contra os schemas
  oficiais do PBIR (`visualContainer` 2.9.0 — versão mais recente publicada no
  repositório `microsoft/json-schemas`; os arquivos declaram 2.12.0, igual às páginas
  já existentes no projeto, então a validação contra 2.9.0 ignorou apenas o campo
  `$schema` e checou toda a estrutura real): 0 erros em 11 arquivos novos/alterados.
  `page.json` e `pages.json` validados contra suas versões exatas (2.1.0 e 1.1.0): 0
  erros.
- Todo campo `Entity`/`Property` referenciado nos visuais novos foi conferido contra as
  colunas e medidas efetivamente definidas no modelo TMDL: 0 divergências.
- SHA-256 de todos os arquivos do projeto original (`P1_dashboard.pbip`,
  `P1_dashboard.Report/**`, `P1_dashboard.SemanticModel/**`) conferido antes e depois:
  idêntico.
