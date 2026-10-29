# CesarAlex

[English documentation](README.en.md) · Versão declarada no código: **1.0.0**

O CesarAlex é uma aplicação web para integrar e comparar registros de produção
científica presentes em currículos **Lattes**, perfis **ORCID** e na **OpenAlex**.
O DOI é a chave principal de comparação e o resultado mostra, para cada obra, em
quais fontes o identificador foi localizado.

A interface continua disponível em Streamlit e preserva os controles, relatórios
CSV, gráfico de fluxo e diagrama de cobertura já utilizados no projeto.

Instância pública: <https://cesaralex-beta.streamlit.app/>

## Por que o CesarAlex é necessário?

O Currículo Lattes é a principal fonte curricular da comunidade científica
brasileira, enquanto ORCID e OpenAlex oferecem identificadores e metadados com
alcance internacional. Consultar essas fontes separadamente torna difícil saber,
para cada pesquisador e DOI, quais produções estão presentes, ausentes ou apenas
em uma das bases. O CesarAlex automatiza essa integração e produz dados
auditáveis para estudos bibliométricos, diagnóstico de cobertura, curadoria de
identificadores e avaliação da completude de perfis acadêmicos.

O público principal inclui pesquisadores em Bibliometria e Cientometria,
bibliotecários, gestores de pesquisa e equipes que trabalham com dados da
Plataforma Lattes. A relação com ferramentas existentes e a motivação científica
são detalhadas no [artigo preparado para a JOSS](paper/paper.md).

## Funcionalidades

- leitura simultânea de um ou vários currículos Lattes em HTML;
- extração de nome, ORCID, DOI, título e ano do Lattes;
- recuperação de obras pela API pública do ORCID;
- recuperação paginada de obras da OpenAlex por ORCID;
- verificação opcional de currículos sem ORCID por DOI;
- inferência conservadora do ID OpenAlex por correspondência de nome;
- cache do perfil OpenAlex inferido para evitar consultas DOI a DOI desnecessárias;
- relatórios individuais e consolidado em CSV;
- identificação de currículos sem ORCID com produção localizada na OpenAlex;
- visualizações de fluxo e sobreposição entre as três fontes;
- acompanhamento, no menu lateral, dos saldos e do custo reportado pela OpenAlex.

## Saldos e consumo da OpenAlex

O menu lateral apresenta três barras:

1. currículos processados;
2. saldo diário restante, orçamento diário e valor usado no dia;
3. saldo pré-pago restante.

Também são exibidos o número de chamadas HTTP, as consultas em lista e
individuais e o custo reportado pela OpenAlex durante a execução. O botão
**Atualizar saldos** permite consultar os valores a qualquer momento. Uma nova
consulta também é feita automaticamente no início e ao final do processamento.

Quando são informadas várias chaves separadas por ponto e vírgula, a interface
agrega os saldos das chaves válidas. As chaves não são exibidas nem incluídas nos
relatórios.

## Correção do perfil virtual

Quando um DOI permite inferir com segurança um único ID de autor, o CesarAlex
carrega uma vez as obras desse perfil. Os próximos DOIs são procurados primeiro
nesse índice local. Um acerto pelo perfil virtual agora:

- marca `Presente na OpenAlex` como `True`;
- conserva o ID OpenAlex real em `IdsOpenAlex_Provaveis`;
- apresenta o nome real em `NomesOpenAlex_Provaveis` e `MatchPairs_OpenAlex`;
- registra `perfil_inferido` em `MetodoMatch`;
- não dispara uma nova consulta por DOI.

Se o DOI seguinte não existir nos perfis já conhecidos, a consulta direta à
OpenAlex é realizada normalmente. Um novo perfil forte pode então ser adicionado
ao índice do currículo atual.

## Instalação

O projeto requer Python 3.10 ou superior.

```bash
git clone https://github.com/wellbar2/CesarAlex.git
cd CesarAlex
python -m venv .venv
```

Ative o ambiente antes de instalar: `.venv\Scripts\activate` no Prompt de
Comando do Windows, `.venv\Scripts\Activate.ps1` no PowerShell, ou
`source .venv/bin/activate` no Linux/macOS.

```bash
python -m pip install -e .
```


## Execução

```bash
streamlit run app.py
```

Na barra lateral, informe uma ou mais chaves OpenAlex, separadas por ponto e
vírgula. Em seguida, envie os HTMLs do Lattes e selecione **Processar currículos**.

A opção **Verificar OpenAlex mesmo sem ORCID (por DOI)** ativa a inferência de
perfil e a busca complementar DOI a DOI.

Um roteiro completo, incluindo como exportar o HTML do Lattes, interpretar os
resultados e executar o exemplo sem rede, está em [docs/USAGE.md](docs/USAGE.md).
O arquivo [examples/curriculo_lattes_sintetico.html](examples/curriculo_lattes_sintetico.html)
não contém dados pessoais e pode ser usado para verificar a leitura do HTML.

## Documentação

- [Instalação detalhada](docs/INSTALLATION.md)
- [Guia de uso e exemplo reproduzível](docs/USAGE.md)
- [Arquitetura e decisões de projeto](docs/ARCHITECTURE.md)
- [API do núcleo Python](docs/API.md)
- [Validação e testes](docs/VALIDATION.md)
- [Privacidade e uso responsável](docs/PRIVACY.md)
- [Preparação e pendências da JOSS](docs/JOSS_READINESS.md)
- [Governança](GOVERNANCE.md) e [suporte](SUPPORT.md)

## Estrutura

```text
app.py                  entrada Streamlit
cesaralex/
  clients.py            clientes ORCID/OpenAlex e caches de requisição
  config.py             parâmetros internos centralizados
  lattes.py             leitura do HTML Lattes
  matching.py           heurística de correspondência de autoria
  models.py             modelos de domínio
  processor.py          fluxo principal, independente da interface
  profiles.py           cache de perfis virtuais inferidos
  reporting.py          CSVs, contadores e preparação de tabelas
  ui.py                 interface Streamlit
  usage.py              saldos, custos e agregação de múltiplas chaves
  visualizations.py     Sankey e diagrama de cobertura
tests/                  testes automatizados
docs/                   documentação técnica e de uso
examples/               exemplo sintético e demonstração sem rede
paper/                   artigo e referências para a JOSS
```

Essa separação permite validar o processamento sem iniciar o Streamlit nem fazer
requisições reais às APIs.

## Testes e qualidade

Instale as dependências de desenvolvimento e execute:

```bash
python -m pip install -r requirements-dev.txt
python -m ruff format --check cesaralex tests examples app.py
python -m ruff check cesaralex tests examples app.py
python -m pytest
python -m pytest --cov=cesaralex
```

Os testes incluem uma regressão específica para o acerto por perfil virtual: o
segundo DOI deve ser marcado como presente na OpenAlex, manter ID e nome corretos
e não realizar uma segunda consulta DOI.

## Saídas principais

Os CSVs mantêm separador `;`, codificação UTF-8 e aspas duplas em todos os
campos. As colunas principais são:

- `Arquivo`, `nomeLattes`, `ORCID`, `DOI`, `Título`, `Ano`;
- `Presente no Lattes`, `Presente no ORCID`, `Presente na OpenAlex`;
- `ISSN-L`;
- `MatchPairs_OpenAlex`, `IdsOpenAlex_Provaveis`,
  `NomesOpenAlex_Provaveis`, `ORCIDs_Provaveis`, `MetodoMatch`,
  `MatchScoreMax`.

## Limitações

- obras sem DOI não entram na comparação principal;
- a qualidade da integração depende dos identificadores informados nas fontes;
- a inferência de autoria é conservadora, mas continua sendo uma heurística;
- limites e disponibilidade das APIs externas podem afetar o processamento;
- a instância pública do Streamlit possui menos recursos que uma execução local.

Os arquivos Lattes podem conter dados pessoais. Consulte as orientações de
[privacidade e uso responsável](docs/PRIVACY.md) antes de processar ou compartilhar
currículos de terceiros.

## Autoria e citação

- Wellington Barbosa Rodrigues — desenvolvimento e responsabilidade técnica;
- Thamyres Vieira dos Santos — concepção e orientação de domínio;
- Rogério Mugnaini — concepção, orientação de domínio e direção do projeto.

Os metadados para citar o software estão em [CITATION.cff](CITATION.cff). As
contribuições são descritas em [AUTHORS.md](AUTHORS.md).

## Licença

GNU General Public License v3.0. Consulte o arquivo `LICENSE`.

## Scientific publications / Publicações científicas

- **Origin of the research approach:** Rodrigues, Santos, and Mugnaini (2026), *BiblioS*, [DOI: 10.5195/biblios.2026.1370](https://doi.org/10.5195/biblios.2026.1370). The study developed the scientific basis for the three-source DOI coverage analysis; do not assume it was run with the current release.
- **Confirmed application of CesarAlex:** Rodrigues, Tomassini, Mugnaini, and Mena-Chalco (2026), *Além do ORCID autodeclarado no Lattes: inferência de identificadores via DOI–OpenAlex e seus limites*, X EBBC, [DOI: 10.22477/x.ebbc.1017](https://doi.org/10.22477/x.ebbc.1017). The project lead confirms that this study used CesarAlex. The reported inference rates are specific to the studied population and should not be presented as general software accuracy.


## Evidências de validação

Em 29/10/2026, o espelho fornecido foi instalado em ambiente Linux limpo com
Python 3.12.14: os 28 testes passaram, incluindo a inicialização do Streamlit.
Ruff, verificação de dependências e construção das distribuições também passaram.
Esses testes usam respostas controladas; a validação com HTMLs originais e
consultas autenticadas ainda está pendente. Veja
[os resultados](docs/RESULTADOS_VALIDACAO_2026-10-29.md) e o
[roteiro com dados reais](docs/VALIDACAO_REAL_PASSO_A_PASSO.md).
