# CesarAlex — Interseção entre Lattes, ORCID e OpenAlex via Identificadores Persistentes

O app **CesarAlex** é uma aplicação web para explorar e compreender a **interseção entre três fontes amplamente utilizadas na comunicação científica**:

- **Currículo Lattes** (a partir de arquivos HTML)
- **ORCID** (identificador persistente de pesquisador e lista de obras associadas)
- **OpenAlex** (base informacional aberta de metadados bibliográficos)

O foco do CesarAlex é **conectar e comparar registros usando Identificadores Únicos Persistentes (PIDs)** — principalmente **DOI** (para publicações) e **ORCID** (para autoria).  
Isso permite identificar o que aparece em mais de uma base, o que aparece apenas em uma delas e onde estão as diferenças de cobertura.

---

## Acesse o app (instância hospedada)

O CesarAlex está disponível em:

https://cesaralex-beta.streamlit.app/

---

## Para quem é o CesarAlex?

O CesarAlex é útil para:

- Pesquisadores que desejam **entender a cobertura e a sobreposição** de suas publicações entre bases
- Grupos de pesquisa, programas e laboratórios que precisam observar a interseção de registros de um conjunto de currículos
- Bibliotecários e analistas que trabalham com **curadoria e padronização de identificadores**
- Pessoas que produzem análises bibliométricas/cientométricas com dados abertos e querem **verificar consistência via PIDs**

---

## O que o app faz

A partir do upload de **um ou vários currículos Lattes em HTML**, o app CesarAlex:

1. **Detecta se existe ORCID no Lattes** (a partir do link `orcid.org` no HTML).
2. **Extrai DOIs do Lattes**, quando eles aparecem como link de DOI na produção.
3. Quando há ORCID:
   - Consulta o **ORCID** e obtém publicações associadas ao perfil (quando houver DOI disponível nas obras).
   - Consulta a **OpenAlex** usando o **ORCID** para obter publicações (incluindo **DOI** e **ISSN-L** quando disponível).
4. Quando **não há ORCID no Lattes**:
   - Opcionalmente (configurável na interface), consulta a **OpenAlex por DOI**, DOI a DOI, para verificar presença na base.
5. **Consolida os resultados por DOI**, indicando em quais bases cada DOI aparece.
6. Permite **download em CSV**:
   - Um CSV por pesquisador/arquivo
   - Um CSV consolidado (geral)
   - CSVs extras quando há casos “OpenAlex encontrado por DOI, mas ORCID ausente no Lattes”

---

## Identificadores persistentes utilizados

- **ORCID**: usado para recuperar obras no ORCID e também para consultar obras na OpenAlex associadas a um autor.
- **DOI**: usado como chave principal para comparar registros entre as bases (e também para consulta direta na OpenAlex quando não há ORCID).

> Importante: se uma obra não possui DOI (ou se o DOI não aparece/está divergente na fonte), ela pode não entrar na comparação.

---

## Visualizações disponíveis

O app apresenta duas visualizações principais:

### 1) “Identificação nas bases”
Um gráfico que mostra, a partir do conjunto de arquivos enviados:
- Quantos currículos **têm ORCID** no HTML
- Quantos currículos **não têm ORCID**
- Entre os currículos com ORCID, quantos retornam publicações na **OpenAlex** e quantos não retornam

### 2) “Cobertura entre as bases”
Um diagrama de Venn com **Lattes × ORCID × OpenAlex**, calculado a partir do relatório consolidado por DOI, exibindo:
- Contagens em cada região do diagrama
- Percentuais em cada região do diagrama

---

## Arquivos de saída (downloads)

Após o processamento, o app oferece:

### 1) CSV por pesquisador (por arquivo HTML)
Um botão “Baixar CSV de: <nome do arquivo>”.

Cada CSV individual contém, por DOI, as colunas:
- `Arquivo`
- `ORCID` (ou “Não disponível” quando não encontrado)
- `DOI`
- `Título`
- `Ano`
- `Presente no Lattes` (True/False)
- `Presente no ORCID` (True/False)
- `Presente na OpenAlex` (True/False)
- `ISSN-L` (quando disponível via OpenAlex)

### 2) CSV consolidado (todos os arquivos)
Um relatório geral reunindo todas as linhas (todos os currículos processados), com as mesmas colunas.

### 3) Casos “OpenAlex encontrado por DOI, mas sem ORCID no Lattes”
Quando a opção de consulta por DOI está ativa e o HTML não contém ORCID, o app pode gerar:
- Um CSV por arquivo com o resumo do caso (quantos DOIs do Lattes foram encontrados na OpenAlex e exemplos)
- Um CSV com as linhas DOI-a-DOI explicando o motivo

Esses arquivos ajudam a identificar situações em que **há sinais de presença na OpenAlex (via DOI), mas o ORCID não está registrado no Lattes**.

---

## Como usar

### Passo 1 — Exportar o Lattes em HTML
1. Acesse o Currículo Lattes desejado
2. Exporte/baixe o currículo em **HTML**
3. Salve o arquivo no seu computador

Repita para quantos currículos quiser comparar.

### Passo 2 — Enviar os arquivos no app
1. Acesse o app: https://cesaralex-beta.streamlit.app/
2. Em “Selecione os arquivos de currículo Lattes (HTML)”, envie um ou vários arquivos

### Passo 3 — Escolher opções
Na barra lateral, você pode ativar/desativar:

**“Verificar OpenAlex mesmo sem ORCID (por DOI)”**

- Ativada: se o currículo não tiver ORCID, o app ainda tenta verificar a OpenAlex consultando DOI por DOI.
- Desativada: sem ORCID, o app fica restrito ao que estiver disponível no Lattes (e não faz consulta DOI-a-DOI na OpenAlex).

### Passo 4 — Processar
Clique em **“Processar currículos”**.

### Passo 5 — Explorar e baixar resultados
Depois de concluir, você poderá:
- Baixar CSVs individuais por currículo
- Ver as visualizações (fluxo e Venn)
- Baixar o CSV consolidado e (se aplicável) os CSVs de casos sem ORCID no Lattes

---

## Limitações da versão hospedada e uso em maior escala

A instância hospedada do CesarAlex foi disponibilizada para uso público e demonstração, e por isso pode apresentar **limitações práticas** no processamento de grandes volumes de currículos (por exemplo, tempo total de execução e limites de recursos do ambiente de hospedagem).

Para análises em maior escala, **acima de 100 currículos**, recomendamos **entrar em contato com os autores** para solicitar a **versão destinada à execução em máquina local**, adequada para processamento de volumes maiores.


---

## Como interpretar os resultados

### “Presente no Lattes / ORCID / OpenAlex”
Esses campos indicam se o **mesmo DOI** apareceu naquela base.

Por exemplo:
- DOI presente em Lattes e OpenAlex, mas não em ORCID → pode indicar que a obra não está associada ao perfil ORCID, ou que o DOI não foi incluído/registrado naquela entrada do ORCID.
- DOI presente em ORCID e OpenAlex, mas não no Lattes → pode indicar ausência no currículo exportado, diferenças de cadastro, ou recortes do que está no HTML.
- DOI presente apenas em uma base → pode indicar diferenças de cobertura, atualização, metadados ou ausência do identificador em alguma fonte.

### Observação sobre Título e Ano
O app preenche `Título` e `Ano` com base em prioridades (dependendo do caso), podendo usar informações de ORCID, Lattes ou OpenAlex, e não havendo forte consistência nesse campo.  
O objetivo é facilitar leitura e comparação, mas o **vínculo principal de comparação é o DOI**.

---

## Limitações esperadas

- Obras sem DOI (ou com DOI ausente no HTML/ORCID) podem ficar fora da comparação.
- O Lattes em HTML pode não expor DOIs em todos os itens (depende de como a produção está registrada).
- Diferenças de registro entre bases podem ocorrer (mesmo para a “mesma” obra), especialmente quando o identificador está incompleto ou divergente.

---

## Créditos

- **Desenvolvimento e responsabilidade técnica:** Wellington Barbosa Rodrigues  
- **Concepção e orientação de domínio:** Rogério Mugnaini; Thamyres Vieira dos Santos

---

## Como citar / referenciar o projeto (opcional)

Se você utilizar o CesarAlex em um relatório, artigo ou produto acadêmico, você pode referenciar o repositório e a instância hospedada, destacando o uso para análise de interseção entre bases via identificadores persistentes (DOI/ORCID).
