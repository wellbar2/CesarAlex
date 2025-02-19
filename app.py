import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objects as go
#from matplotlib_venn import venn3
import matplotlib.pyplot as plt
import re
import math
import time  # Importado para a pausa

st.set_page_config(page_title="Plataforma CesarAlex", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("Integração de Currículo Lattes, ORCID e OpenAlex")
st.write("""
Este aplicativo recebe os arquivos de currículo Lattes (HTML) e, para cada um:
- Extrai o ORCID (se presente) diretamente do HTML;
- Consulta a API do ORCID para obter as publicações do pesquisador;
- Consulta a OpenAlex (usando o ORCID) para obter as publicações do pesquisador;
- Extrai os DOIs exibidos diretamente no Lattes, juntamente com informações de Título e Ano;
- Consolida os dados dos artigos (Título, Ano, DOI) considerando se o artigo está presente em cada fonte;
- Gera um arquivo .txt com as informações consolidadas;
- Exibe um relatório consolidado com os resultados;
- Apresenta um diagrama de Sankey e um diagrama de Venn-Euler para a cobertura dos DOIs.
""")

##############################
# FUNÇÕES AUXILIARES
##############################

def normalize_doi(doi):
    """
    Remove prefixos comuns e coloca o DOI em caixa baixa para permitir a comparação.
    """
    if not doi or doi == "Sem DOI":
        return None
    doi = doi.lower().strip()
    doi = doi.replace("http://dx.doi.org/", "")
    doi = doi.replace("https://dx.doi.org/", "")
    doi = doi.replace("https://doi.org/", "")
    doi = doi.replace("http://doi.org/", "")
    return doi

def extract_orcid_from_html(content):
    """
    Extrai o primeiro link que contenha 'orcid.org' do conteúdo HTML.
    Retorna o link ORCID ou None se não encontrado.
    """
    soup = BeautifulSoup(content, "html.parser")
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "orcid.org" in href:
            return href
    return None

def get_publications_from_orcid(orcid):
    """
    Consulta a API ORCID e retorna uma lista de publicações.
    Cada publicação é um dicionário com os campos: Título, Ano e DOI (normalizado).
    """
    if "https://orcid.org/" in orcid:
        orcid_id = orcid.replace("https://orcid.org/", "")
    elif "http://orcid.org/" in orcid:
        orcid_id = orcid.replace("http://orcid.org/", "")
    else:
        return []

    api_url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"
    headers = {"Accept": "application/json"}
    response = requests.get(api_url, headers=headers)
    publications = []

    if response.status_code == 200:
        data = response.json()
        if "group" in data:
            for group in data["group"]:
                for work_summary in group.get("work-summary", []):
                    title = work_summary.get("title", {}).get("title", {}).get("value", "Sem título")
                    publication_year = "Sem data"
                    if work_summary.get("publication-date") and work_summary["publication-date"].get("year"):
                        publication_year = work_summary["publication-date"]["year"].get("value", "Sem data")
                    doi_raw = "Sem DOI"
                    if "external-ids" in work_summary:
                        for ext_id in work_summary["external-ids"].get("external-id", []):
                            if ext_id.get("external-id-type", "").lower() == "doi":
                                doi_raw = ext_id.get("external-id-value", "Sem DOI")
                                break
                    doi = normalize_doi(doi_raw) or "Sem DOI"
                    publications.append({
                        "Título": title,
                        "Ano": publication_year,
                        "DOI": doi
                    })
    else:
        st.error(f"Erro ao consultar a API ORCID para {orcid}. Código: {response.status_code}")

    return publications

def get_publications_from_openalex(orcid):
    """
    Consulta a OpenAlex utilizando o ORCID (removendo o prefixo) e retorna uma lista de publicações.
    Cada publicação é um dicionário com os campos: Título, Ano, DOI (normalizado) e ISSN-L.
    """
    normalized_orcid = orcid.replace("https://orcid.org/", "").replace("http://orcid.org/", "").strip()
    url = f"https://api.openalex.org/works?filter=authorships.author.orcid:{normalized_orcid}&per_page=200"
    response = requests.get(url)
    publications = []
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            for work in data["results"]:
                doi_raw = work.get("doi", None)
                doi = normalize_doi(doi_raw) if doi_raw else "Sem DOI"
                title = work.get("display_name", "Sem título")
                publication_year = work.get("publication_year", "Sem data")
                # Extração do ISSN-L de primary_location -> source com verificação robusta
                primary_location = work.get("primary_location")
                if isinstance(primary_location, dict):
                    source = primary_location.get("source")
                    if isinstance(source, dict):
                        issn_l = source.get("issn_l", "Sem ISSN-L")
                    else:
                        issn_l = "Sem ISSN-L"
                else:
                    issn_l = "Sem ISSN-L"
                publications.append({
                    "Título": title,
                    "Ano": publication_year,
                    "DOI": doi,
                    "ISSN-L": issn_l
                })
    else:
        st.error(f"Erro ao consultar OpenAlex para {orcid}. Código: {response.status_code}")
    return publications

def check_doi_in_lattes(content, doi):
    """
    Verifica se o DOI normalizado está presente no conteúdo HTML do Lattes.
    Retorna True se encontrado, False caso contrário.
    """
    if doi and doi != "Sem DOI":
        return doi in content.lower()
    return False

def check_openalex(doi):
    """
    Consulta a API OpenAlex para verificar se o DOI (já normalizado) está presente.
    Retorna True se o DOI for encontrado, False caso contrário.
    """
    if doi and doi != "Sem DOI":
        url = f"https://api.openalex.org/works/http://dx.doi.org/{doi}"
        try:
            response = requests.get(url)
            return response.status_code == 200
        except Exception as e:
            return False
    return False

def extract_dois_from_lattes(content):
    """
    Extrai DOIs do HTML do currículo Lattes procurando links com a classe "icone-producao icone-doi".
    Retorna um conjunto com os DOIs normalizados.
    """
    soup = BeautifulSoup(content, "html.parser")
    dois = set()
    doi_links = soup.find_all("a", class_="icone-producao icone-doi")
    for link in doi_links:
        doi = link.get("href")
        norm = normalize_doi(doi)
        if norm:
            dois.add(norm)
    return dois

def extract_articles_info_from_lattes(content):
    """
    Extrai informações dos artigos (Título e Ano) do HTML do Lattes.
    Procura links com a classe "icone-producao icone-doi" e utiliza o texto do elemento pai.
    Retorna um dicionário: {doi_normalizado: {"Título": ..., "Ano": ...}, ...}
    """
    soup = BeautifulSoup(content, "html.parser")
    articles = {}
    doi_links = soup.find_all("a", class_="icone-producao icone-doi")
    for link in doi_links:
        doi_raw = link.get("href")
        norm = normalize_doi(doi_raw)
        if not norm:
            continue
        parent = link.find_parent()
        parent_text = parent.get_text(" ", strip=True) if parent else ""
        match = re.search(r'\b(19|20)\d{2}\b', parent_text)
        year = match.group(0) if match else "Não disponível"
        title = parent_text if parent_text else "Não disponível"
        articles[norm] = {"Título": title, "Ano": year}
    return articles

def generate_consolidated_txt_content(rows):
    """
    Gera o conteúdo do arquivo .txt a partir da lista de linhas consolidadas.
    Cada linha terá o formato:
    Índice|Título|Ano|DOI|Presente no Lattes|Presente no ORCID|Presente na OpenAlex|ISSN-L
    """
    lines = []
    header = "Índice|Título|Ano|DOI|Presente no Lattes|Presente no ORCID|Presente na OpenAlex|ISSN-L"
    lines.append(header)
    for idx, row in enumerate(rows, start=1):
        line = f"{idx}|{row['Título']}|{row['Ano']}|{row['DOI']}|{row['Presente no Lattes']}|{row['Presente no ORCID']}|{row['Presente na OpenAlex']}|{row['ISSN-L']}"
        lines.append(line)
    return "\n".join(lines)


def create_venn_figure(A, B, C):
    """
    Cria um diagrama de Venn interativo para três conjuntos usando Plotly.

    Parâmetros:
        A, B, C (set): Conjuntos de itens.

    Retorna:
        fig (go.Figure): Figura do Plotly com o diagrama de Venn.
    """
    # Cálculo das regiões
    a_only = len(A - B - C)
    b_only = len(B - A - C)
    c_only = len(C - A - B)
    ab = len((A & B) - C)
    ac = len((A & C) - B)
    bc = len((B & C) - A)
    abc = len(A & B & C)

    # Parâmetros para os círculos do diagrama
    r = 1.1  # raio dos círculos
    # Posicionamento baseado em um triângulo equilátero
    center_A = (0, 0)
    center_B = (1, 0)
    center_C = (0.5, math.sqrt(3)/2)  # aproximadamente (0.5, 0.866)

    # Função auxiliar para gerar a definição de um círculo (shape)
    def circle_shape(center, r, fillcolor, line_color):
        cx, cy = center
        return dict(
            type="circle",
            xref="x",
            yref="y",
            x0=cx - r,
            y0=cy - r,
            x1=cx + r,
            y1=cy + r,
            fillcolor=fillcolor,
            opacity=0.3,
            line_color=line_color,
        )

    # Definição dos shapes (círculos)
    shape_A = circle_shape(center_A, r, "rgba(255, 0, 0, 0.3)", "red")
    shape_B = circle_shape(center_B, r, "rgba(0, 255, 0, 0.3)", "green")
    shape_C = circle_shape(center_C, r, "rgba(0, 0, 255, 0.3)", "blue")

    # Cria a figura
    fig = go.Figure()

    # Adiciona os shapes dos círculos
    fig.update_layout(shapes=[shape_A, shape_B, shape_C])

    # Adiciona anotações para cada região
    annotations = [
        dict(x=-0.5, y=-0.4, text=str(a_only), showarrow=False, font=dict(size=14, color="red")),
        dict(x=1.5, y=-0.4, text=str(b_only), showarrow=False, font=dict(size=14, color="green")),
        dict(x=0.5, y=1.3, text=str(c_only), showarrow=False, font=dict(size=14, color="blue")),
        dict(x=0.5, y=-0.5, text=str(ab), showarrow=False, font=dict(size=14, color="black")),
        dict(x=0.0, y=0.5, text=str(ac), showarrow=False, font=dict(size=14, color="black")),
        dict(x=1.0, y=0.5, text=str(bc), showarrow=False, font=dict(size=14, color="black")),
        dict(x=0.5, y=0.3, text=str(abc), showarrow=False, font=dict(size=14, color="black")),
    ]
    fig.update_layout(annotations=annotations)

    # Adiciona traces "dummy" para gerar a legenda
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='rgba(255, 0, 0, 0.3)', line=dict(color='red', width=2)),
        name='ORCID: DOIs presentes no ORCID'
    ))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='rgba(0, 255, 0, 0.3)', line=dict(color='green', width=2)),
        name='Lattes: DOIs presentes no Lattes'
    ))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color='rgba(0, 0, 255, 0.3)', line=dict(color='blue', width=2)),
        name='OpenAlex: DOIs presentes na OpenAlex'
    ))

    # Configura os eixos para remover as linhas e marcas de fundo
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False, range=[-2, 3])
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False, range=[-2, 3])

    # Configuração do layout da figura
    fig.update_layout(
        title="Cobertura de Publicações: ORCID vs. Lattes vs. OpenAlex",
        width=600,
        height=600,
        plot_bgcolor="white",
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='Black',
            borderwidth=1
        )
    )

    return fig



##############################
# INTERFACE STREAMLIT
##############################

uploaded_files = st.file_uploader("Selecione os arquivos de currículo Lattes (HTML)", accept_multiple_files=True)

# Contadores para o Sankey:
files_with_orcid = 0
files_without_orcid = 0
files_with_openalex_count = 0
files_without_openalex_count = 0

# Conjuntos para o diagrama de Venn (DOIs normalizados)
all_dois_orcid = set()
all_dois_lattes = set()
all_dois_openalex = set()

all_report = []  # Lista para o relatório consolidado

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"---\n### Processando: **{uploaded_file.name}**")
        try:
            content = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Erro ao ler o arquivo {uploaded_file.name}: {e}")
            continue

        # Extrai DOIs diretamente do Lattes e informações (Título e Ano)
        lattes_dois = extract_dois_from_lattes(content)
        all_dois_lattes.update(lattes_dois)
        lattes_articles = extract_articles_info_from_lattes(content)

        # Extração do ORCID a partir do HTML
        orcid_link = extract_orcid_from_html(content)
        if not orcid_link:
            st.warning(f"ORCID não encontrado em {uploaded_file.name}.")
            files_without_orcid += 1
            # Mesmo sem ORCID, incluímos os DOIs extraídos do Lattes no relatório
            union_dois = lattes_dois
            consolidated_rows = []
            for doi in union_dois:
                title = lattes_articles[doi]["Título"] if doi in lattes_articles else "Não disponível"
                year = lattes_articles[doi]["Ano"] if doi in lattes_articles else "Não disponível"
                openalex_presente = check_openalex(doi)
                if openalex_presente:
                    all_dois_openalex.add(doi)
                consolidated_rows.append({
                    "Arquivo": uploaded_file.name,
                    "ORCID": "Não disponível",
                    "DOI": doi,
                    "Título": title,
                    "Ano": year,
                    "Presente no Lattes": True,
                    "Presente no ORCID": False,
                    "Presente na OpenAlex": openalex_presente,
                    "ISSN-L": "Não disponível"
                })
            txt_content = generate_consolidated_txt_content(consolidated_rows)
            st.download_button(
                label=f"Baixar arquivo consolidado (.txt) para {uploaded_file.name}",
                data=txt_content,
                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_consolidado.txt",
                mime="text/plain"
            )
            all_report.extend(consolidated_rows)
            continue
        else:
            st.success(f"ORCID encontrado: {orcid_link}")
            files_with_orcid += 1

        # Consulta à API ORCID para obter publicações
        orcid_publications = get_publications_from_orcid(orcid_link)
        orcid_dict = {}
        for pub in orcid_publications:
            doi = pub["DOI"]
            if doi and doi != "Sem DOI":
                orcid_dict[doi] = pub
        all_dois_orcid.update(set(orcid_dict.keys()))

        # Consulta à OpenAlex usando o ORCID e pausa de 1 segundo após a requisição
        openalex_publications = get_publications_from_openalex(orcid_link)
        time.sleep(1)  # Pausa de 1 segundo
        openalex_dict = {}
        for pub in openalex_publications:
            doi = pub["DOI"]
            if doi and doi != "Sem DOI":
                openalex_dict[doi] = pub
        all_dois_openalex.update(set(openalex_dict.keys()))

        # Atualiza os contadores de arquivos com ou sem publicações da OpenAlex (para arquivos com ORCID)
        if openalex_publications:
            files_with_openalex_count += 1
        else:
            files_without_openalex_count += 1

        # União dos DOIs oriundos do ORCID, dos extraídos do Lattes e dos obtidos da OpenAlex
        union_dois = set(orcid_dict.keys()) | lattes_dois | set(openalex_dict.keys())

        consolidated_rows = []
        for doi in union_dois:
            row = {}
            row["Arquivo"] = uploaded_file.name
            row["ORCID"] = orcid_link
            row["DOI"] = doi
            # Prioridade para definir Título e Ano: ORCID > Lattes > OpenAlex
            if doi in orcid_dict:
                row["Título"] = orcid_dict[doi].get("Título", "Sem título")
                row["Ano"] = orcid_dict[doi].get("Ano", "Sem data")
                row["Presente no ORCID"] = True
            elif doi in lattes_articles:
                row["Título"] = lattes_articles[doi].get("Título", "Não disponível")
                row["Ano"] = lattes_articles[doi].get("Ano", "Não disponível")
                row["Presente no ORCID"] = False
            elif doi in openalex_dict:
                row["Título"] = openalex_dict[doi].get("Título", "Sem título")
                row["Ano"] = openalex_dict[doi].get("Ano", "Sem data")
                row["Presente no ORCID"] = False
            else:
                row["Título"] = "Não disponível"
                row["Ano"] = "Não disponível"
                row["Presente no ORCID"] = False
            row["Presente no Lattes"] = (doi in lattes_dois)
            row["Presente na OpenAlex"] = (doi in openalex_dict)
            # Insere o ISSN-L, se disponível (apenas se o DOI estiver em openalex_dict)
            if doi in openalex_dict:
                row["ISSN-L"] = openalex_dict[doi].get("ISSN-L", "Sem ISSN-L")
            else:
                row["ISSN-L"] = "Não disponível"
            consolidated_rows.append(row)

        txt_content = generate_consolidated_txt_content(consolidated_rows)
        st.download_button(
            label=f"Baixar arquivo consolidado (.txt) para {uploaded_file.name}",
            data=txt_content,
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_consolidado.txt",
            mime="text/plain"
        )
        all_report.extend(consolidated_rows)

    ##############################
    # DIAGRAMA DE SANKEY
    ##############################
    # Nós do Sankey:
    # 0: "Arquivos Enviados"
    # 1: "Tem ORCID"
    # 2: "Sem ORCID"
    # 3: "Tem OpenAlex" (dentre os que têm ORCID)
    # 4: "Sem OpenAlex" (dentre os que têm ORCID)
    sankey_labels = ["Arquivos Enviados", "Tem ORCID", "Sem ORCID", "Tem OpenAlex", "Sem OpenAlex"]
    # Fluxo: do nó 0 para 1 e 2; do nó 1 para 3 e 4.
    sankey_source = [0, 0, 1, 1]
    sankey_target = [1, 2, 3, 4]
    # Valores:
    # - Do nó 0: valor = files_with_orcid e files_without_orcid
    # - Do nó 1: valor = files_with_openalex_count e files_without_openalex_count
    sankey_values = [files_with_orcid, files_without_orcid, files_with_openalex_count, files_without_openalex_count]

    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_labels,
            color=["lightblue", "lightgreen", "salmon", "gold", "orange"]
        ),
        link=dict(
            source=sankey_source,
            target=sankey_target,
            value=sankey_values
        )
    )])
    sankey_fig.update_layout(title_text="Fluxo de Arquivos: ORCID e OpenAlex", font_size=10)
    st.markdown("## Diagrama de Sankey")
    st.plotly_chart(sankey_fig)

    ##############################
    # DIAGRAMA DE VENN-EULER (3 conjuntos)
    ##############################
    # Conjuntos:
    # A: DOIs presentes no ORCID
    # B: DOIs presentes no Lattes
    # C: DOIs presentes na OpenAlex
    A = all_dois_orcid
    B = all_dois_lattes
    C = all_dois_openalex

    #plt.figure(figsize=(7, 7))
    #v = venn3(subsets=(len(A - B - C), len(B - A - C), len(A & B - C),
    #                   len(C - A - B), len(A & C - B), len(B & C - A),
    #                   len(A & B & C)),
    #          set_labels=('ORCID', 'Lattes', 'OpenAlex'))
    #plt.title("Cobertura de Publicações: ORCID vs. Lattes vs. OpenAlex")
    #st.markdown("## Diagrama de Venn-Euler")
    #st.pyplot(plt.gcf())
    fig = create_venn_figure(A, B, C)
    st.markdown("## Diagrama de Venn")
    st.plotly_chart(fig)



    ##############################
    # RELATÓRIO CONSOLIDADO
    ##############################
    if all_report:
        st.markdown("## Relatório Consolidado")
        df = pd.DataFrame(all_report)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar relatório completo em CSV",
            data=csv,
            file_name="relatorio_consolidado.csv",
            mime="text/csv"
        )
    else:
        st.info("Nenhum dado consolidado para exibir.")

footer_html = """<div style='text-align: center;'>
  <p>Plataforma desenvolvida por Wellbar - 2025</p>
</div>"""
st.markdown(footer_html, unsafe_allow_html=True)
