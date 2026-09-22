# P1_dashboard_v4

Dashboard construído do zero (novo `.Report` + `.SemanticModel`, não uma cópia de
`P1_dashboard`/`P1_dashboard_v3`), a partir de uma reavaliação de storytelling
(`STORYTELLING.md`). Objetivo: menos páginas, menos jargão, só o que tem propósito e
interação.

## Como abrir

1. Abra `P1_dashboard_v4.pbip` no Power BI Desktop.
2. **Atualizar** (Início → Atualizar) para carregar os dados — o modelo lê diretamente de
   `outputs/07.2_powerbi_model.xlsx` e `outputs/07.3_region_vs_country.xlsx`, os mesmos
   arquivos já validados (nenhum recálculo).
3. Duas páginas: **Overview** (mapa) e **Region vs country**, mais uma página de tooltip
   oculta.

## O que mudou frente a v1/v3

- **11 páginas → 2 + 1 tooltip.** Ver `STORYTELLING.md` para o raciocínio completo.
- **Modelo semântico: ~37 tabelas → 6** (`Dim_Country`, `Dim_Region`, `Fact_Region`,
  `RVC_Region`, `RVC_Country`, `RVC_Variance`), cada uma só com as colunas realmente usadas
  em algum visual ou medida.
- **`Dim_Country` agora é uma tabela nativa do Power Query** (28 países, hard-coded), não uma
  coluna calculada DAX — elimina uma dependência que causou um bug real em v3 (ver abaixo).
- **17 medidas DAX**, todas com propósito direto num visual (nenhuma medida "para o caso").
- **Mapa preservado** — mesmo TopoJSON, mesmo tema visual, copiados do dashboard original.
- **Correção de contraste**: cabeçalho passou a ter fundo azul-marinho real (`#14212E`) sob
  texto branco (era branco sobre branco antes). Um bug idêntico foi pego e corrigido no hover
  do menu de páginas durante a própria construção deste dashboard.
- **Bug do v3 não repetido**: o gráfico "onde as regiões se parecem com o exterior" lê direto
  de `RVC_Country` (uma linha por país, já agregada), em vez de agregar `RVC_Region` através
  de dois saltos de relacionamento — a causa raiz do bug de agrupamento visto em v3.

## Validação feita

- JSON de todo arquivo novo é sintaticamente válido.
- Todo `visual.json` e `page.json` validado contra os schemas oficiais do PBIR
  (`visualContainer` 2.9.0 — a versão mais recente publicada; os arquivos declaram 2.12.0,
  igual ao resto do projeto — e `page` 2.1.0, `pagesMetadata` 1.1.0, ambos na versão exata):
  0 erros.
- Todo campo `Entity`/`Property` referenciado nos visuais existe de fato no modelo TMDL: 0
  divergências.
- SHA-256 de todos os arquivos do `P1_dashboard` original conferido antes e depois de todo o
  trabalho: idêntico (nada no dashboard original foi tocado).

## Pendências / decisões deixadas para o autor

- **Botão "Does geography matter more than borders?"** na página 1 usa um link de navegação
  de página (`visualLink` com `PageNavigation`/`navigationSection`). A sintaxe passa na
  validação de schema, mas essa combinação específica não está documentada no schema público
  — se o clique não navegar, use o menu de páginas no topo (`Overview` / `Region vs
  country`), que é o mecanismo comprovado (mesmo usado em `P1_dashboard` original).
- **Sem tabela de detalhe por região** (era `dt_region_profile` em v1/v3): a decisão de
  "menos é mais" trocou a página de detalhe por um cartão inline que atualiza ao clicar no
  mapa, na própria página 1. Se precisar de mais profundidade por região (indicadores
  anuais, por exemplo), isso ficou de fora de propósito.
- **Tooltip do mapa simplificado**: mostra região, país, perfil, movimento e região mais
  parecida — sem o gráfico de fuzzy membership que existia em v1/v3 (a tabela `Dim_Cluster`/
  colunas `u_cluster_1`/`u_cluster_2` não fazem parte do modelo v4).
