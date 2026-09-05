"""Interface Streamlit da plataforma CesarAlex."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from .clients import (
    OpenAlexClient,
    OpenAlexService,
    OrcidService,
    parse_openalex_api_keys,
)
from .config import DEFAULT_SETTINGS
from .lattes import decode_lattes_html, parse_lattes_html
from .models import FileProcessingResult
from .processor import CesarAlexProcessor
from .reporting import (
    build_application_results,
    compute_venn_counts,
    dataframe_to_csv_bytes,
    make_arrow_friendly,
    uploaded_files_fingerprint,
)
from .usage import (
    OpenAlexBalance,
    aggregate_balances,
    api_keys_fingerprint,
    balance_from_rate_limit_payload,
    format_usd,
)
from .visualizations import create_sankey, create_venn


@dataclass(slots=True)
class SidebarUsageWidgets:
    curricula_placeholder: Any
    daily_placeholder: Any
    prepaid_placeholder: Any
    stats_placeholder: Any
    balance: OpenAlexBalance | None
    balance_cost_baseline: float = 0.0

    def update_curricula(self, processed: int, total: int) -> None:
        fraction = min(1.0, processed / total) if total else 0.0
        total_label = str(total) if total else "—"
        self.curricula_placeholder.progress(
            fraction,
            text=f"Currículos: {processed} / {total_label}",
        )

    def set_balance(
        self,
        balance: OpenAlexBalance | None,
        *,
        cost_baseline: float = 0.0,
    ) -> None:
        self.balance = balance
        self.balance_cost_baseline = max(0.0, cost_baseline)
        self._render_balance(balance)

    def update_usage_stats(self, stats: dict[str, int | float]) -> None:
        st.session_state["openalex_run_stats"] = stats
        reported_cost = float(stats.get("reported_cost_usd", 0.0) or 0.0)
        current_balance = self.balance
        if current_balance:
            estimated_cost = max(0.0, reported_cost - self.balance_cost_baseline)
            current_balance = current_balance.estimate_after_cost(estimated_cost)
        self._render_balance(current_balance)
        self.stats_placeholder.caption(
            f"Chamadas HTTP: {int(stats.get('http_requests', 0) or 0)} | "
            f"listas: {int(stats.get('list_requests', 0) or 0)} | "
            f"individuais: {int(stats.get('singleton_requests', 0) or 0)} | "
            f"custo reportado: {format_usd(reported_cost)}"
        )

    def _render_balance(self, balance: OpenAlexBalance | None) -> None:
        if balance is None:
            self.daily_placeholder.progress(0.0, text="Saldo diário: —")
            self.prepaid_placeholder.progress(0.0, text="Saldo pré-pago: —")
            return
        self.daily_placeholder.progress(
            balance.daily_remaining_fraction,
            text=(
                f"Saldo diário: {format_usd(balance.daily_remaining_usd)} / "
                f"{format_usd(balance.daily_budget_usd)} | "
                f"usado: {format_usd(balance.daily_used_usd)}"
            ),
        )
        self.prepaid_placeholder.progress(
            balance.prepaid_remaining_fraction,
            text=(
                f"Saldo pré-pago: {format_usd(balance.prepaid_remaining_usd)} / "
                f"{format_usd(balance.prepaid_reference_usd)}"
            ),
        )


def run_app() -> None:
    st.set_page_config(
        page_title="Plataforma CesarAlex",
        page_icon=None,
        layout="centered",
        initial_sidebar_state="auto",
    )
    _initialize_state()
    _render_introduction()
    api_keys, check_openalex_by_doi, usage_widgets = _render_sidebar()
    uploaded_files = st.file_uploader(
        "Selecione os arquivos de currículo Lattes (HTML)",
        accept_multiple_files=True,
    )
    _invalidate_results_when_input_changes(
        uploaded_files,
        check_openalex_by_doi=check_openalex_by_doi,
    )
    usage_widgets.update_curricula(
        st.session_state["processed_file_count"],
        len(uploaded_files or []),
    )

    process_clicked = st.button(
        "Processar currículos",
        type="primary",
        disabled=not uploaded_files or not api_keys,
    )
    if uploaded_files and not api_keys:
        st.warning(
            "Informe a chave OpenAlex no menu lateral para iniciar o processamento."
        )
    if uploaded_files and api_keys and not st.session_state["processed"]:
        st.info(
            "Envie os currículos e clique em **Processar currículos** "
            "para iniciar o processamento."
        )

    if uploaded_files and process_clicked and not st.session_state["processed"]:
        _process_files(
            uploaded_files,
            api_keys=api_keys,
            check_openalex_by_doi=check_openalex_by_doi,
            usage_widgets=usage_widgets,
        )

    if st.session_state["processed"] and st.session_state["results"]:
        _render_results(st.session_state["results"])

    st.markdown(
        "<div style='text-align: center;'><p>"
        "Plataforma desenvolvida por Wellbar - 2026"
        "</p></div>",
        unsafe_allow_html=True,
    )


def _render_introduction() -> None:
    st.title("Integração de Currículo Lattes, ORCID e OpenAlex")
    st.write(
        """
Este aplicativo recebe arquivos de currículo Lattes (HTML) e, para cada um:
- Identifica se o ORCID está no Lattes;
- Consulta a API do ORCID para obter publicações;
- Consulta a API da OpenAlex usando o ORCID para obter publicações (DOIs + ISSN-L);
- Extrai DOIs diretamente do Lattes quando existe link na publicação;
- Consolida as publicações por DOI e permite download em CSV: por pesquisador e geral;
- Apresenta um gráfico presença ORCID/OpenAlex e um diagrama de cobertura multibase.
"""
    )


def _render_sidebar() -> tuple[list[str], bool, SidebarUsageWidgets]:
    st.sidebar.header("Opções")
    keys_text = st.sidebar.text_input(
        "Chave OpenAlex",
        value="",
        type="password",
    )
    check_by_doi = st.sidebar.checkbox(
        "Verificar OpenAlex mesmo sem ORCID (por DOI)",
        value=False,
        help=(
            "Se ativado, pode demorar muito mais porque o sistema consulta "
            "a OpenAlex para cada DOI individualmente."
        ),
    )
    api_keys = parse_openalex_api_keys(keys_text)
    key_fingerprint = api_keys_fingerprint(api_keys) if api_keys else None
    if st.session_state["openalex_balance_key"] != key_fingerprint:
        st.session_state["openalex_balance_key"] = key_fingerprint
        st.session_state["openalex_balance"] = None
        st.session_state["openalex_run_stats"] = _empty_usage_stats()
        st.session_state["openalex_balance_cost_baseline"] = 0.0

    st.sidebar.divider()
    st.sidebar.subheader("Uso da OpenAlex")
    refresh_balance = st.sidebar.button(
        "Atualizar saldos",
        disabled=not api_keys,
        use_container_width=True,
    )
    if refresh_balance:
        balance, errors = _fetch_openalex_balance(api_keys)
        _show_balance_errors(errors)
        if balance:
            _store_balance(
                balance,
                key_fingerprint=key_fingerprint,
                cost_baseline=float(
                    st.session_state["openalex_run_stats"].get("reported_cost_usd", 0.0)
                    or 0.0
                ),
            )

    widgets = SidebarUsageWidgets(
        curricula_placeholder=st.sidebar.empty(),
        daily_placeholder=st.sidebar.empty(),
        prepaid_placeholder=st.sidebar.empty(),
        stats_placeholder=st.sidebar.empty(),
        balance=st.session_state["openalex_balance"],
        balance_cost_baseline=st.session_state["openalex_balance_cost_baseline"],
    )
    widgets.update_curricula(
        st.session_state["processed_file_count"],
        st.session_state["total_file_count"],
    )
    widgets.set_balance(
        st.session_state["openalex_balance"],
        cost_baseline=st.session_state["openalex_balance_cost_baseline"],
    )
    widgets.update_usage_stats(st.session_state["openalex_run_stats"])
    if widgets.balance and widgets.balance.key_count > 1:
        st.sidebar.caption(
            f"Saldos agregados de {widgets.balance.key_count} chaves válidas."
        )
    return api_keys, check_by_doi, widgets


def _initialize_state() -> None:
    st.session_state.setdefault("fingerprint", None)
    st.session_state.setdefault("processed", False)
    st.session_state.setdefault("results", None)
    st.session_state.setdefault("openalex_balance_key", None)
    st.session_state.setdefault("openalex_balance", None)
    st.session_state.setdefault("openalex_balance_cost_baseline", 0.0)
    st.session_state.setdefault("openalex_run_stats", _empty_usage_stats())
    st.session_state.setdefault("processed_file_count", 0)
    st.session_state.setdefault("total_file_count", 0)


def _empty_usage_stats() -> dict[str, int | float]:
    return {
        "http_requests": 0,
        "successful_requests": 0,
        "retry_requests": 0,
        "list_requests": 0,
        "singleton_requests": 0,
        "rate_requests": 0,
        "reported_cost_usd": 0.0,
    }


def _fetch_openalex_balance(
    api_keys: list[str],
) -> tuple[OpenAlexBalance | None, list[str]]:
    balances: list[OpenAlexBalance] = []
    errors: list[str] = []
    balance_settings = replace(
        DEFAULT_SETTINGS,
        openalex_max_retries=2,
        openalex_backoff_base=1.0,
    )
    for position, api_key in enumerate(api_keys, start=1):
        client = OpenAlexClient([api_key], settings=balance_settings)
        payload = client.get_rate_limit()
        if payload:
            balances.append(balance_from_rate_limit_payload(payload))
        else:
            errors.append(
                f"Chave {position}: {client.last_error or 'saldo indisponível'}."
            )
    return aggregate_balances(balances), errors


def _show_balance_errors(errors: list[str]) -> None:
    if errors:
        st.sidebar.warning("Não foi possível ler todos os saldos. " + " ".join(errors))


def _store_balance(
    balance: OpenAlexBalance,
    *,
    key_fingerprint: str | None,
    cost_baseline: float,
) -> None:
    st.session_state["openalex_balance"] = balance
    st.session_state["openalex_balance_key"] = key_fingerprint
    st.session_state["openalex_balance_cost_baseline"] = max(0.0, cost_baseline)


def _invalidate_results_when_input_changes(
    uploaded_files,
    *,
    check_openalex_by_doi: bool,
) -> None:
    if not uploaded_files:
        return
    fingerprint = (
        f"{uploaded_files_fingerprint(uploaded_files)}:"
        f"check_doi={int(check_openalex_by_doi)}"
    )
    if st.session_state["fingerprint"] != fingerprint:
        st.session_state["fingerprint"] = fingerprint
        st.session_state["processed"] = False
        st.session_state["results"] = None
        st.session_state["processed_file_count"] = 0
        st.session_state["total_file_count"] = len(uploaded_files)


def _process_files(
    uploaded_files,
    *,
    api_keys: list[str],
    check_openalex_by_doi: bool,
    usage_widgets: SidebarUsageWidgets,
) -> None:
    st.session_state["openalex_run_stats"] = _empty_usage_stats()
    st.session_state["processed_file_count"] = 0
    st.session_state["total_file_count"] = len(uploaded_files)
    usage_widgets.update_curricula(0, len(uploaded_files))
    usage_widgets.update_usage_stats(_empty_usage_stats())

    initial_balance, initial_errors = _fetch_openalex_balance(api_keys)
    _show_balance_errors(initial_errors)
    if initial_balance:
        _store_balance(
            initial_balance,
            key_fingerprint=api_keys_fingerprint(api_keys),
            cost_baseline=0.0,
        )
        usage_widgets.set_balance(initial_balance, cost_baseline=0.0)

    client = OpenAlexClient(
        api_keys,
        on_warning=st.warning,
        on_usage_change=usage_widgets.update_usage_stats,
    )
    openalex = OpenAlexService(client, on_error=st.error)
    processor = CesarAlexProcessor(
        openalex,
        orcid=OrcidService(on_error=st.error),
    )
    outcomes: list[FileProcessingResult] = []
    file_progress = st.progress(0)
    status = st.empty()
    total_files = len(uploaded_files)

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        status.write(f"Processando {index}/{total_files}: **{uploaded_file.name}**")
        st.markdown(f"---\n### Processando: **{uploaded_file.name}**")
        try:
            content = decode_lattes_html(uploaded_file.getvalue())
            curriculum = parse_lattes_html(content)
        except Exception as exc:  # noqa: BLE001 - fronteira de arquivos enviados
            st.error(f"Erro ao ler {uploaded_file.name}: {exc}")
            file_progress.progress(int(index / total_files * 100))
            st.session_state["processed_file_count"] = index
            usage_widgets.update_curricula(index, total_files)
            continue

        if curriculum.orcid_url:
            st.success(f"ORCID encontrado: {curriculum.orcid_url}")
        else:
            st.warning("ORCID não encontrado no HTML do Lattes.")

        doi_progress = st.progress(0) if check_openalex_by_doi else None
        doi_status = st.empty() if check_openalex_by_doi else None

        def update_doi_progress(
            current: int,
            total: int,
            profiles: int,
            progress_bar=doi_progress,
            status_placeholder=doi_status,
        ) -> None:
            if progress_bar is None or status_placeholder is None or total == 0:
                return
            status_placeholder.write(
                f"Consultando OpenAlex/cache por DOI: {current}/{total} "
                f"| perfis inferidos: {profiles}"
            )
            progress_bar.progress(int(current / total * 100))

        outcome = processor.process_curriculum(
            uploaded_file.name,
            curriculum,
            check_openalex_by_doi=check_openalex_by_doi,
            on_doi_progress=update_doi_progress,
        )
        outcomes.append(outcome)

        if doi_status and outcome.rows:
            doi_status.write(
                f"Consulta concluída | DOIs encontrados na OpenAlex: "
                f"{len(outcome.openalex_dois)} | perfis inferidos mantidos em cache"
            )
        if not outcome.has_orcid and outcome.openalex_dois:
            st.info(
                "**Sem ORCID no Lattes, mas com presença na OpenAlex via DOI**: "
                f"{len(outcome.openalex_dois)} DOI(s) encontrado(s)."
            )
        file_progress.progress(int(index / total_files * 100))
        st.session_state["processed_file_count"] = index
        usage_widgets.update_curricula(index, total_files)

    status.write("Processamento concluído.")
    file_progress.empty()
    final_stats = client.usage_stats()
    final_balance, final_errors = _fetch_openalex_balance(api_keys)
    _show_balance_errors(final_errors)
    if final_balance:
        reported_cost = float(final_stats.get("reported_cost_usd", 0.0) or 0.0)
        _store_balance(
            final_balance,
            key_fingerprint=api_keys_fingerprint(api_keys),
            cost_baseline=reported_cost,
        )
        usage_widgets.set_balance(final_balance, cost_baseline=reported_cost)
    usage_widgets.update_usage_stats(final_stats)
    st.session_state["results"] = build_application_results(
        outcomes,
        check_openalex_by_doi=check_openalex_by_doi,
    )
    st.session_state["processed"] = True
    st.success(
        "Processamento concluído. Agora você pode baixar os arquivos sem reprocessar."
    )


def _render_results(results: dict) -> None:
    st.markdown("## Arquivos por pesquisador")
    for filename, dataframe in results["per_file_df"].items():
        st.download_button(
            label=f"Baixar CSV de: {filename}",
            data=dataframe_to_csv_bytes(dataframe),
            file_name=f"{filename.rsplit('.', 1)[0]}_individual.csv",
            mime="text/csv",
            key=f"dl_file_{filename}",
        )

    st.markdown("## Identificação nas bases")
    st.plotly_chart(create_sankey(results["sankey"]), width="stretch")

    st.markdown("## Cobertura entre as bases")
    report_rows = results["all_report_df"].to_dict(orient="records")
    counts, total = compute_venn_counts(report_rows)
    venn_figure = create_venn(
        counts,
        total,
        title="Cobertura: Lattes × ORCID × OpenAlex",
    )
    st.pyplot(venn_figure)
    plt.close(venn_figure)

    _render_no_orcid_warning(results)

    st.markdown("## Relatório Consolidado")
    st.dataframe(make_arrow_friendly(results["all_report_df"]), width="stretch")
    st.download_button(
        label="Baixar relatório completo em CSV",
        data=dataframe_to_csv_bytes(results["all_report_df"]),
        file_name="relatorio_consolidado.csv",
        mime="text/csv",
        key="dl_all_report",
    )


def _render_no_orcid_warning(results: dict) -> None:
    st.markdown("## Com OpenAlex, mas sem ORCID no Lattes (identificado por DOI)")
    if not results["options"].get("check_openalex_without_orcid", True):
        st.info(
            "Ative a opção 'Verificar OpenAlex mesmo sem ORCID (por DOI)' "
            "para detectar estes casos."
        )
        return
    if results["warn_files_df"].empty:
        st.info(
            "Nenhum caso encontrado (ou a opção de consulta por DOI está desativada)."
        )
        return

    st.dataframe(make_arrow_friendly(results["warn_files_df"]), width="stretch")
    st.download_button(
        label="Baixar CSV (arquivos sem ORCID, mas com OpenAlex via DOI)",
        data=dataframe_to_csv_bytes(results["warn_files_df"]),
        file_name="sem_orcid_mas_com_openalex_por_doi.csv",
        mime="text/csv",
        key="dl_warn_files",
    )
    if results["warn_rows_df"].empty:
        return
    st.dataframe(make_arrow_friendly(results["warn_rows_df"]), width="stretch")
    st.download_button(
        label="Baixar CSV (DOIs encontrados na OpenAlex sem ORCID no Lattes)",
        data=dataframe_to_csv_bytes(results["warn_rows_df"]),
        file_name="dois_openalex_sem_orcid_no_lattes.csv",
        mime="text/csv",
        key="dl_warn_rows",
    )
