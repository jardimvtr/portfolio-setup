# Como aplicar a cópia P1_dashboard_v3 no Power BI Desktop

## 1. Abrir
Feche o `P1_dashboard.pbip` original se estiver aberto (para não confundir o Desktop
sobre qual projeto está ativo) e abra **`results/dashboard/P1_dashboard_v3.pbip`**.

## 2. Atualizar
As 3 tabelas novas (`RVC_Variance`, `RVC_Region`, `RVC_Country`) já estão definidas no
modelo, mas os *dados* só são carregados quando você atualiza:
- Menu **Início → Atualizar** (ou clique direito em qualquer tabela no painel de
  dados → Atualizar).
- Isso lê `outputs/07.3_region_vs_country.xlsx` (mesma pasta absoluta já usada pelas
  outras tabelas do modelo).

Se o Desktop pedir para reintroduzir o caminho da fonte de dados (por causa do cache
removido nesta cópia), aponte para o mesmo arquivo:
`C:\Users\vitor\Documents\Portfolio\Projects\P1\outputs\07.3_region_vs_country.xlsx`.

## 3. Conferir a página nova
Abra a página **"Region vs country"** (logo após "Country context" na navegação).
Sem nenhum filtro aplicado, os KPIs devem mostrar:
- Country share of profile variation ≈ 70%
- Country share of movement variation ≈ 12%
- Most similar region is abroad ≈ 179 (35%)
- Would differ under country's majority profile: 37

Esses são os mesmos números do QA do Script 07.3 (ver
`results/07.3_results_report.md`).

## 4. Pendências (não incluídas nesta cópia)
- **Mapa "most similar region: same country vs abroad"** (item opcional B.3.8 da
  especificação): não foi criado. Se quiser, duplique o visual de mapa existente na
  página "Map" e colore por uma medida booleana baseada em
  `RVC_Region[nn1_is_foreign]`; recarregue o TopoJSON se necessário
  (`results/dashboard/map/P1_ADM1_regions.topojson`).
- **Tooltip do mapa** (`tt_region`): o popup de região (320×210px) já está no limite do
  espaço com os 3 blocos existentes. Os campos "Most similar region" e "Share of 5 most
  similar regions abroad" foram adicionados apenas no cartão de **detalhe da região**
  (`dt_region_profile`), não no tooltip do mapa. Adicionar ao tooltip exigiria redesenhar
  o popup — deixado como decisão do autor.
- **Consolidação do 07.3 nos scripts 07.1/07.2**: não foi feita (decisão do autor, ver
  `results/07.3_results_report.md`).

## 5. Publicar (opcional)
Esta cópia local ainda não está vinculada a nenhum workspace do Power BI Service (os
vínculos remotos do projeto original foram removidos desta cópia de propósito, para
evitar confusão sobre qual item do serviço ela representa). Publique como um item novo
quando estiver pronto.
