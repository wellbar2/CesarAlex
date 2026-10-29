# Histórico de mudanças

## 1.0.0 — primeira versão estável

- primeira versão pública estável preparada para distribuição e avaliação;
- documentação completa em português e inglês;
- artigo curto e bibliografia no formato exigido pela JOSS;
- metadados de citação em `CITATION.cff`;
- documentação da arquitetura, API, validação, governança e suporte;
- exemplo sintético reproduzível sem acesso às APIs externas;
- configurações de testes e compilação do artigo fornecidas separadamente para instalação e verificação no GitHub;
- pacote Python identificado como estável e instalável com `pip`.

## 0.3.0 — saldos e consumo da OpenAlex

- três barras no menu lateral: currículos, saldo diário e saldo pré-pago;
- consulta manual de saldos e atualização automática antes e depois do processamento;
- agregação dos saldos quando são informadas várias chaves OpenAlex;
- acompanhamento de chamadas HTTP, consultas em lista, consultas individuais e custo reportado;
- estimativa visual do consumo entre duas atualizações do endpoint de saldo;
- testes dos cálculos financeiros e das métricas do cliente.

## 0.2.0 — refatoração estrutural

- separação do núcleo de processamento e da interface Streamlit;
- cache explícito de perfis OpenAlex inferidos;
- correção da presença OpenAlex em DOIs resolvidos pelo perfil virtual;
- preservação do ID e do nome real do autor em resultados provenientes do cache;
- correção da presença OpenAlex em consultas DOI complementares de currículos com ORCID;
- leitura única de cada HTML Lattes;
- dependências diretas simplificadas;
- testes automatizados; a execução do fluxo de integração contínua deve ser confirmada no GitHub.

## Preparação documental — 29/10/2026

- inclusão do estudo do X EBBC como aplicação efetiva do CesarAlex;
- distinção entre essa aplicação e a pesquisa anterior publicada na BiblioS;
- registro da instalação limpa, 28 testes aprovados e demais verificações;
- roteiro para validação com currículos originais e APIs reais;
- indicação explícita das verificações ainda pendentes;
- remoção de uma data de lançamento não confirmada em `CITATION.cff`;
- nenhuma alteração nos algoritmos, parâmetros ou arquivos Python da aplicação.

Os números de versão acima descrevem o código e sua evolução. A existência e
as datas de tags/releases públicas precisam ser verificadas no GitHub; este
documento não constitui prova de que esses lançamentos foram publicados.
