# Histórico de mudanças

## 1.0.0 — primeira versão estável

- primeira versão pública estável preparada para distribuição e avaliação;
- documentação completa em português e inglês;
- artigo curto e bibliografia no formato exigido pela JOSS;
- metadados de citação em `CITATION.cff`;
- documentação da arquitetura, API, validação, governança e suporte;
- modelos públicos para relatos de erro, propostas de funcionalidade e *pull requests*;
- exemplo sintético reproduzível sem acesso às APIs externas;
- fluxo automatizado para compilar o rascunho do artigo da JOSS;
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
- testes automatizados e fluxo de integração contínua.
