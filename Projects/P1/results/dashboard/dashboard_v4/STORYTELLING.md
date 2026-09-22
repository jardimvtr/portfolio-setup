# Storytelling reassessment — v1/v3 vs. v4

Reavaliação pedida: como alguém que nunca viu as especificidades metodológicas (M-Exp-FCMd,
fuzzy c-medoids, BoD, eta², epsilon², leave-one-out, permutação) interpretaria o dashboard
atual (v1/v3, 11 páginas) — e o que sobra quando se aplica "menos é mais".

## O que um visitante de primeira viagem via em v1/v3

- **Termos técnicos sem tradução**: "membership_margin", "u_cluster_1", "convergence_index",
  "epsilon² (viés-corrigido)", "p de permutação", "leave-one-out", "medoid", "transition zone"
  aparecem como rótulos de KPI/eixo sem explicação. Ninguém fora do método sabe se
  `convergence_index = -0.0192` é bom ou ruim.
- **11 páginas, muitas delas paralelas/redundantes**: Profiles, Map, Trajectories, Country
  context e Region vs country repetem o mesmo material de base (perfil, fuzzy membership,
  convergência) em layouts diferentes, sem uma página deixar claramente óbvio qual é a
  pergunta central do projeto.
- **Indicadores auxiliares sem interação**: tabelas de referência por país (todas as 28
  distâncias per região), rankings de contribuição por indicador, gráficos de linha de
  distância anual — informação real, mas nenhuma delas conecta com as outras nem responde
  a uma pergunta que o visitante tenha se feito.
- **O achado mais forte do projeto (região vs país) ficava enterrado na 5ª/6ª página**,
  depois de 4 páginas de vocabulário técnico — exatamente o oposto do "teste do recrutador de
  3 minutos" descrito no Guia-Mestre do miniportfólio.
- **Bug de contraste**: o cabeçalho branco com texto branco (`#FFFFFF` sobre `#FFFFFF`) nas
  páginas v1/v3 — historicamente presente em todas as páginas, provavelmente nunca notado
  porque o Power BI Desktop pode estar preenchendo com o tema por trás. Corrigido em v4 com
  fundo azul-marinho real (`#14212E`) sob texto branco.

## O que realmente importa (essência + propósito)

1. **Existem dois padrões de trajetória de desenvolvimento (C1, 100 regiões; C2, 412
   regiões)**, não uma classificação de "melhor/pior" — os valores medianos de cada dimensão
   são quase idênticos entre os dois grupos; o que os separa é o formato da trajetória
   2015-2019 (por isso o texto explicativo do v4 evita qualquer linguagem de ranking).
2. **Muitas regiões estão em movimento** — 35,5% (182 de 512) convergindo para o perfil
   alternativo.
3. **O ponto central de toda a extensão feita no Script 07.3**: o país explica ~70% de qual
   perfil uma região tem, mas só ~12% de como ela está se movendo — e para 35% das regiões, a
   região mais parecida está em outro país. Esse é o "propósito de termos calculado tudo
   isso": mostrar que comparar região a região revela algo que comparar só por país esconde.

## Decisão de design para o v4

- **2 páginas principais + 1 tooltip**, contra as 11 de v1/v3.
- **Mapa mantido como o recurso central** da página 1, com clique-para-detalhe inline (sem
  navegar para outra página) mostrando perfil, movimento e região mais parecida — já
  plantando a pergunta da página 2.
- **Nenhum indicador sem propósito**: cada número no v4 está numa KPI, num eixo ou num
  tooltip que alguém realmente vai ler; nada de tabela de 512×28 distâncias ou ranking de
  contribuição por indicador.
- **Linguagem**: "C1"/"C2" mantidos (são os identificadores reais, mudar o nome seria
  reescrever o resultado), mas cada aparição vem com uma frase que explica o que significam
  sem jargão.
- **Cores**: paleta reduzida a 4 (`#14212E` marinho, `#0F7C73` verde-azulado, `#F4F6F9`
  fundo, `#5C6B7A`/`#1B2632` texto) reaproveitada do dashboard original, mas toda combinação
  texto/fundo foi conferida manualmente (ver CHANGELOG) para nunca repetir claro-sobre-claro
  ou escuro-sobre-escuro — inclusive um bug que eu mesmo introduzi no hover do menu de
  páginas foi pego e corrigido nessa checagem.
