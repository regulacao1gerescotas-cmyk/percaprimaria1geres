import streamlit as st
import pandas as pd
import numpy as np
import unicodedata

# ---------------- UTILITÁRIOS (Sem alterações) ----------------
def normalize_text(s):
    """Normaliza strings: remove acentos, colapsa espaços e converte para lowercase."""
    if pd.isna(s):
        return ""
    s = str(s)
    s = " ".join(s.split())
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()

def normalizar_colunas(df):
    """Normaliza nomes de colunas do DataFrame (remove acentos e lower)."""
    df = df.copy()
    df.columns = [normalize_text(col) for col in df.columns]
    return df

def proportional_allocation(sizes, total):
    """
    Aloca 'total' inteiros proporcionalmente aos valores em 'sizes'.
    Garante que soma(alocação) == total, distribuindo restos pelos maiores
    restos fracionários.
    """
    sizes = np.array(sizes, dtype=float)
    if total <= 0 or sizes.sum() == 0:
        return [0] * len(sizes)
    raw = sizes / sizes.sum() * total
    floored = np.floor(raw).astype(int)
    remainder = int(total - floored.sum())
    if remainder > 0:
        frac = raw - floored
        order = np.argsort(-frac)  # índices por frac desc
        for i in range(remainder):
            floored[order[i]] += 1
    return floored.tolist()

# ---------------- LISTA DE UNIDADES E MAPEAMENTO (Sem alterações) ----------------
unidades_permitidas = [
    "COTEL CENTRO DE OBSERV. E TRIAG. PROF. EVERALDO LUNA",
    "UNIDADE BASICA DE SAUDE PRISIONAL PFAL - ABREU E LIMA",
    "UNIDADE BASICA DE SAUDE PRISIONAL PLL - RECIFE",
    "SECRETARIA MUNICIPAL DE SAUDE DE ABREU E LIMA",
    "SECRETARIA MUNICIPAL DE SAUDE DE ARACOIABA",
    "SECRETARIA MUNICIPAL DE SAUDE DO CABO DE SANTO AGOSTINHO",
    "SECRETARIA MUNICIPAL DE SAUDE DE CAMARAGIBE",
    "SECRETARIA MUNICIPAL DE SAUDE DE CHA DE ALEGRIA",
    "SECRETARIA MUNICIPAL DE SAUDE DE CHA GRANDE",
    "HOSPITAL SAO LUCAS - FERNANDO DE NORONHA",
    "SECRETARIA MUNICIPAL DE SAUDE DE GLORIA DO GOITA",
    "HOSPITAL DE CUSTODIA E TRATAMENTO PSIQUIATRICO - ITAMARACA",
    "SECRETARIA MUNICIPAL DE SAUDE DA ILHA DE ITAMARACA",
    "PENITENCIÁRIA AGRO - INDUSTRIAL SÃO JOÃO - PAISJ",
    "UNIDADE BASICA DE SAUDE PRISIONAL PAISJ - ILHA DE ITAMARACA",
    "PPBC-PENITENCIÁRIA PROF. BARRETO CAMPELO",
    "UNIDADE DE SAÚDE PRISIONAL/PRESIDIO DE IGARASSU-PE",
    "SECRETARIA MUNICIPAL DE SAUDE DE IGARASSU",
    "SECRETARIA MUNICIPAL DE SAUDE DO IPOJUCA",
    "SECRETARIA DE SAUDE DE ITAPISSUMA",
    "POLICLINICA CONEGO PEDRO DE SOUZA LEAO - JABOATAO",
    "POLICLINICA DA CRIANCA E DO ADOLESCENTE",
    "POLICLINICA JOSE CARNEIRO LINS - JABOATAO",
    "POLICLINICA LEOPOLDINA LEAO TENORIO - JABOATAO",
    "POLICLINICA MANOEL CALHEIROS CURADO IV - JABOATAO",
    "POLICLINICA MARIINHA MELO - JABOATAO",
    "UBS CAJUEIRO SECO",
    "SECRETARIA DE SAUDE DO JABOATAO DOS GUARARAPES",
    "SECRETARIA MUNICIPAL DE SAUDE DE MORENO",
    "CENTRO DE SAUDE JARDIM FRAGOSO - OLINDA",
    "CRO (CENTRO DE REABILITAÇÃO DE OLINDA)",
    "NUCLEO DE FISIOTERAPIA DE OURO PRETO - OLINDA",
    "POLICLINICA BARROS BARRETO - OLINDA",
    "POLICLINICA DA MULHER - OLINDA",
    "POLICLINICA DA PESSOA IDOSA - OLINDA",
    "POLICLINICA MARTAGAO GESTEIRA - OLINDA",
    "POLICLINICA OURO PRETO - OLINDA",
    "POLICLINICA PEIXINHOS - OLINDA",
    "POLICLINICA RIO DOCE - IV ETAPA - OLINDA",
    "POLICLINICA RIO DOCE II ETAPA - OLINDA",
    "POLICLINICA SAO BENEDITO - OLINDA",
    "SECRETARIA MUNICIPAL DE SAUDE DE OLINDA",
    "SECRETARIA MUNICIPAL DE SAUDE DE PAULISTA",
    "SECRETARIA MUNICIPAL DE SAUDE DE POMBOS",
    "US 166 POLICLINICA CENTRO - RECIFE",
    "US 376 POLICLINICA SALOMAO KELNER - RECIFE",
    "US 293 POLIClinica do pina - recife",
    "US 275 CEST RECIFE DR EDSON HATEN - RECIFE",
    "US 217 CENTRO MEDICO SEN JOSE ERMIRIO DE MORAES - RECIFE",
    "US 169 POLICLINICA AMAURY COUTINHO -CAMPINA BARRETO - RECIFE",
    "US 101 POLICLINICA PROF WALDEMAR DE OLIVEIRA - RECIFE",
    "US 128 POLICLINICA LESSA DE ANDRADE - RECIFE",
    "US 141 SECRETARIA MUNICIPAL DE SAUDE DO RECIFE",
    "US 144 POLICLINICA CLEMENTINO FRAGA - RECIFE",
    "US 153 POLICLINICA ARNALDO MARQUES - RECIFE",
    "US 159 POLICLINICA AGAMENON MAGALHAES - AFOGADOS - RECIFE",
    "US 160 POLICLINICA GOUVEIA DE BARROS - RECIFE",
    "US 162 POLICLINICA ALBERT SABIN - RECIFE",
    "US 163 UNIDADE PEDIATRICA HELENA MOURA - RECIFE",
    "CPFR - COLONIA PENAL FEMININA DO RECIFE - BOM PASTOR",
    "PRESIDIO ASP MARCELO FRANCISCO ARAUJO - PAMFA",
    "PRESIDIO FREI DAMIAO DE BOZZANO - PFDB - RECIFE",
    "PRESIDIO JUIZ ANTONIO LUIZ DE BARROS - PJALLB - RECIFE",
    "SECRETARIA MUNICIPAL DE SAUDE DE SAO LOURENCO DA MATA",
    "PVSA - PRESÍDIO DE VITORIA DE SANTO ANTAO",
    "SECRETARIA DE SAUDE DA VITORIA DE SANTO ANTAO",
    "UNIDADE BASICA DE SAUDE PRISIONAL PLL - RECIFE",
]
unidades_permitidas_norm = set(normalize_text(u) for u in unidades_permitidas)

mapa_agrupamento = {
    "CENTRO DE SAUDE JARDIM FRAGOSO - OLINDA": "Olinda", "CRO (CENTRO DE REABILITAÇÃO DE OLINDA)": "Olinda",
    "NUCLEO DE FISIOTERAPIA DE OURO PRETO - OLINDA": "Olinda", "POLICLINICA BARROS BARRETO - OLINDA": "Olinda",
    "POLICLINICA DA MULHER - OLINDA": "Olinda", "POLICLINICA DA PESSOA IDOSA - OLINDA": "Olinda",
    "POLICLINICA MARTAGAO GESTEIRA - OLINDA": "Olinda", "POLICLINICA OURO PRETO - OLINDA": "Olinda",
    "POLICLINICA PEIXINHOS - OLINDA": "Olinda", "POLICLINICA RIO DOCE - IV ETAPA - OLINDA": "Olinda",
    "POLICLINICA RIO DOCE II ETAPA - OLINDA": "Olinda", "POLICLINICA SAO BENEDITO - OLINDA": "Olinda",
    "SECRETARIA MUNICIPAL DE SAUDE DE OLINDA": "Olinda", "US 166 POLICLINICA CENTRO - RECIFE": "Recife",
    "US 376 POLICLINICA SALOMAO KELNER - RECIFE": "Recife", "US 293 POLICLINICA DO PINA - RECIFE": "Recife",
    "US 275 CEST RECIFE DR EDSON HATEN - RECIFE": "Recife", "US 217 CENTRO MEDICO SEN JOSE ERMIRIO DE MORAES - RECIFE": "Recife",
    "US 169 POLICLINICA AMAURY COUTINHO -CAMPINA BARRETO - RECIFE": "Recife", "US 101 POLICLINICA PROF WALDEMAR DE OLIVEIRA - RECIFE": "Recife",
    "US 128 POLICLINICA LESSA DE ANDRADE - RECIFE": "Recife", "US 141 SECRETARIA MUNICIPAL DE SAUDE DO RECIFE": "Recife",
    "US 144 POLICLINICA CLEMENTINO FRAGA - RECIFE": "Recife",
    "US 153 POLICLINICA ARNALDO MARQUES - RECIFE": "Recife", "US 159 POLICLINICA AGAMENON MAGALHAES - AFOGADOS - RECIFE": "Recife",
    "US 160 POLICLINICA GOUVEIA DE BARROS - RECIFE": "Recife", "US 162 POLICLINICA ALBERT SABIN - RECIFE": "Recife",
    "US 163 UNIDADE PEDIATRICA HELENA MOURA - RECIFE": "Recife",
    "POLICLINICA CONEGO PEDRO DE SOUZA LEAO - JABOATAO": "Jaboatão dos Guararapes", "POLICLINICA DA CRIANCA E DO ADOLESCENTE": "Jaboatão dos Guararapes",
    "POLICLINICA JOSE CARNEIRO LINS - JABOATAO": "Jaboatão dos Guararapes", "POLICLINICA LEOPOLDINA LEAO TENORIO - JABOATAO": "Jaboatão dos Guararapes",
    "POLICLINICA MANOEL CALHEIROS CURADO IV - JABOATAO": "Jaboatão dos Guararapes", "POLICLINICA MARIINHA MELO - JABOATAO": "Jaboatão dos Guararapes",
    "UBS CAJUEIRO SECO": "Jaboatão dos Guararapes", "SECRETARIA DE SAUDE DO JABOATAO DOS GUARARAPES": "Jaboatão dos Guararapes",
    "COTEL CENTRO DE OBSERV. E TRIAG. PROF. EVERALDO LUNA": "COTEL CENTRO DE OBSERV. E TRIAG. PROF. EVERALDO LUNA",
    "UNIDADE BASICA DE SAUDE PRISIONAL PFAL - ABREU E LIMA": "UNIDADE BASICA DE SAUDE PRISIONAL PFAL - ABREU E LIMA",
    "SECRETARIA MUNICIPAL DE SAUDE DE ABREU E LIMA": "SECRETARIA MUNICIPAL DE SAUDE DE ABREU E LIMA",
    "SECRETARIA MUNICIPAL DE SAUDE DE ARACOIABA": "SECRETARIA MUNICIPAL DE SAUDE DE ARACOIABA",
    "SECRETARIA MUNICIPAL DE SAUDE DO CABO DE SANTO AGOSTINHO": "SECRETARIA MUNICIPAL DE SAUDE DO CABO DE SANTO AGOSTINHO",
    "SECRETARIA MUNICIPAL DE SAUDE DE CAMARAGIBE": "SECRETARIA MUNICIPAL DE SAUDE DE CAMARAGIBE",
    "SECRETARIA MUNICIPAL DE SAUDE DE CHA DE ALEGRIA": "SECRETARIA MUNICIPAL DE SAUDE DE CHA DE ALEGRIA",
    "SECRETARIA MUNICIPAL DE SAUDE DE CHA GRANDE": "SECRETARIA MUNICIPAL DE SAUDE DE CHA GRANDE",
    "HOSPITAL SAO LUCAS - FERNANDO DE NORONHA": "HOSPITAL SAO LUCAS - FERNANDO DE NORONHA",
    "SECRETARIA MUNICIPAL DE SAUDE DE GLORIA DO GOITA": "SECRETARIA MUNICIPAL DE SAUDE DE GLORIA DO GOITA",
    "HOSPITAL DE CUSTODIA E TRATAMENTO PSIQUIATRICO - ITAMARACA": "HOSPITAL DE CUSTODIA E TRATAMENTO PSIQUIATRICO - ITAMARACA",
    "SECRETARIA MUNICIPAL DE SAUDE DA ILHA DE ITAMARACA": "SECRETaria municipal de saude da ilha de itamaraca",
    "PENITENCIÁRIA AGRO - INDUSTRIAL SÃO JOÃO - PAISJ": "PENITENCIÁRIA AGRO - INDUSTRIAL SÃO JOÃO - PAISJ",
    "UNIDADE BASICA DE SAUDE PRISIONAL PAISJ - ILHA DE ITAMARACA": "UNIDADE BASICA DE SAUDE PRISIONAL PAISJ - ILHA DE ITAMARACA",
    "PPBC-PENITENCIÁRIA PROF. BARRETO CAMPELO": "PPBC-PENITENCIÁRIA PROF. BARRETO CAMPELO",
    "UNIDADE DE SAÚDE PRISIONAL/PRESIDIO DE IGARASSU-PE": "UNIDADE DE SAÚDE PRISIONAL/PRESIDIO DE IGARASSU-PE",
    "SECRETARIA MUNICIPAL DE SAUDE DE IGARASSU": "SECRETARIA MUNICIPAL DE SAUDE DE IGARASSU",
    "SECRETARIA MUNICIPAL DE SAUDE DO IPOJUCA": "SECRETARIA MUNICIPAL DE SAUDE DO IPOJUCA",
    "SECRETARIA DE SAUDE DE ITAPISSUMA": "SECRETARIA DE SAUDE DE ITAPISSUMA",
    "SECRETARIA MUNICIPAL DE SAUDE DE MORENO": "SECRETARIA MUNICIPAL DE SAUDE DE MORENO",
    "SECRETARIA MUNICIPAL DE SAUDE DE PAULISTA": "SECRETARIA MUNICIPAL DE SAUDE DE PAULISTA",
    "SECRETARIA MUNICIPAL DE SAUDE DE POMBOS": "SECRETARIA MUNICIPAL DE SAUDE DE POMBOS",
    "CPFR - COLONIA PENAL FEMININA DO RECIFE - BOM PASTOR": "CPFR - COLONIA PENAL FEMININA DO RECIFE - BOM PASTOR",
    "PRESIDIO ASP MARCELo francisco araujo - pamfa": "presidio asp marcelo francisco araujo - pamfa",
    "PRESIDIO FREI DAMIAO DE BOZZANO - PFDB - RECIFE": "PRESIDIO FREI DAMIAO DE BOZZANO - PFDB - RECIFE",
    "PRESIDIO JUIZ ANTONIO LUIZ DE BARROS - PJALLB - RECIFE": "PRESIDIO JUIZ ANTONIO LUIZ DE BARROS - PJALLB - RECIFE",
    "SECRETARIA MUNICIPAL DE SAUDE DE SAO LOURENCO DA MATA": "SECRETARIA MUNICIPAL DE SAUDE DE SAO LOURENCO DA MATA",
    "PVSA - PRESÍDIO DE VITORIA DE SANTO ANTAO": "PVSA - PRESÍDIO DE VITORIA DE SANTO ANTAO",
    "SECRETARIA DE SAUDE DA VITORIA DE SANTO ANTAO": "SECRETARIA DE SAUDE DA VITORIA DE SANTO ANTAO",
    "UNIDADE BASICA DE SAUDE PRISIONAL PLL - RECIFE": "UNIDADE BASICA DE SAUDE PRISIONAL PLL - RECIFE",
}
mapa_agrupamento_norm = {normalize_text(k): v for k, v in mapa_agrupamento.items()}

# ---------------- FUNÇÕES DE DISTRIBUIÇÃO (Sem alterações) ----------------
def organizar_dados_60(df):
    """Converte data, extrai mês e normaliza prioridade para ordenação."""
    df = df.copy()
    df['data da solicitacao'] = pd.to_datetime(df['data da solicitacao'], dayfirst=True, errors='coerce')
    df['mes'] = df['data da solicitacao'].dt.to_period('M')
    df['prioridade'] = df['prioridade'].astype(str).str.lower()
    prioridades = {"muito alta": 1, "alta": 2, "media": 3, "baixa": 4}
    df['prioridade_ord'] = df['prioridade'].map(prioridades).fillna(99).astype(int)
    return df

def distribuir_vagas_60(df, vagas_totais):
    """
    Distribuição 60%: percorre linhas ordenadas e aloca 1 vaga por solicitação.
    Retorna df (com coluna 'vaga alocada') e resumo por unidade.
    """
    df = df.copy()
    tamanho_fila = df['unidade solicitante'].value_counts().to_dict()
    df['fila unidade'] = df['unidade solicitante'].map(tamanho_fila)
    df = df.sort_values(by=['mes', 'prioridade_ord', 'fila unidade'], ascending=[True, True, False])
    
    unidades = df['unidade solicitante'].unique()
    vagas_por_unidade = {unidade: 0 for unidade in unidades}
    df['vaga alocada'] = False
    vagas = int(vagas_totais)
    
    for idx, row in df.iterrows():
        if vagas <= 0:
            break
        unidade = row['unidade solicitante']
        vagas_por_unidade[unidade] += 1
        df.at[idx, 'vaga alocada'] = True
        vagas -= 1
        
    resumo = pd.DataFrame([
        {"Unidade": unidade, "Vagas Alocadas": vagas_aloc, "Fila Total": tamanho_fila.get(unidade, 0)}
        for unidade, vagas_aloc in vagas_por_unidade.items()
    ])
    
    df['vaga alocada'] = df['vaga alocada'].map({True: 'Sim', False: 'Não'})
    return df, resumo

# ---------------- NOVA INTERFACE DO STREAMLIT ----------------
st.set_page_config(layout="wide")
st.title("Distribuição Unificada de Cotas - I GERES")

# --- ENTRADA ÚNICA ---
total_cotas = st.number_input(
    "Insira o número total de cotas a serem distribuídas",
    min_value=1,
    value=100,
    key="total_cotas"
)

# --- CÁLCULO AUTOMÁTICO 60/40 ---
vagas_60 = int(total_cotas * 0.6)
vagas_40 = total_cotas - vagas_60
st.info(f"Do total de {total_cotas} cotas, **{vagas_60}** serão distribuídas pelo critério de 60% e **{vagas_40}** pelo critério de 40%.")

# --- UPLOAD DE ARQUIVOS ---
st.markdown("---")
st.header("Carregamento de Arquivos")
col1, col2 = st.columns(2)
with col1:
    uploaded_file_60 = st.file_uploader(
        "1. Arquivo para distribuição de 60% (lista de solicitações)",
        type=["csv"],
        key="arquivo_solicitacoes"
    )
with col2:
    uploaded_file_40 = st.file_uploader(
        "2. Arquivo para distribuição de 40% (fila resumida por unidade com 'fila' para cálculo e 'real' para resumo)",
        type=["csv"],
        key="arquivo_fila"
    )

# --- LÓGICA DE PROCESSAMENTO PRINCIPAL ---
if uploaded_file_60 is not None and uploaded_file_40 is not None:
    try:
        # --- ETAPA 1: Processamento da distribuição de 60% ---
        df_60_base = pd.read_csv(uploaded_file_60)
        df_60_base = normalizar_colunas(df_60_base)
        required_cols_60 = {'unidade solicitante', 'data da solicitacao', 'prioridade'}
        if not required_cols_60.issubset(df_60_base.columns):
            st.error("Arquivo para 60% não contém as colunas esperadas: 'Unidade solicitante', 'Data da solicitacao', 'Prioridade'.")
        else:
            df_60_base['unidade solicitante'] = df_60_base['unidade solicitante'].astype(str) # Garante que a coluna é string
            df_60_base['unidade_norm'] = df_60_base['unidade solicitante'].apply(normalize_text)
            df_60_filtrado = df_60_base[df_60_base['unidade_norm'].isin(unidades_permitidas_norm)].copy()
            
            df_60_organizado = organizar_dados_60(df_60_filtrado)
            df_distribuicao_60_detalhado, resumo_60 = distribuir_vagas_60(df_60_organizado, vagas_60)

            # Preparar resumo de 60% para a junção
            resumo_60_final = resumo_60.rename(columns={'Unidade': 'Unidade Solicitante', 'Vagas Alocadas': 'Cota 60%', 'Fila Total': 'Fila 60% (Nº de Solicitações)'})

            # --- ETAPA 2: Processamento da distribuição de 40% ---
            df_40_base = pd.read_csv(uploaded_file_40)
            df_40_base = normalizar_colunas(df_40_base)
            
            # Verificar se ambas as colunas 'fila' e 'real' existem
            if 'fila' not in df_40_base.columns or 'real' not in df_40_base.columns or 'unidade solicitante' not in df_40_base.columns:
                st.error("Arquivo para 40% deve conter as colunas: 'Unidade solicitante', 'Fila' (para cálculo) e 'Real' (para resumo).")
            else:
                # Converter as colunas para numérico
                df_40_base['fila'] = pd.to_numeric(df_40_base['fila'], errors='coerce').fillna(0).astype(int)
                df_40_base['real'] = pd.to_numeric(df_40_base['real'], errors='coerce').fillna(0).astype(int)
                df_40_base['unidade_original'] = df_40_base['unidade solicitante'].astype(str) # Garante que a coluna é string
                df_40_base['unidade_norm'] = df_40_base['unidade solicitante'].apply(normalize_text)
                df_40_filtrado = df_40_base[df_40_base['unidade_norm'].isin(unidades_permitidas_norm)].copy()

                df_40_filtrado['agrupamento'] = df_40_filtrado['unidade_norm'].map(mapa_agrupamento_norm).fillna("SEM AGRUPAMENTO")
                
                # Agrupar e somar pela COLUNA 'FILA' para a proporção da alocação
                df_grouped = df_40_filtrado.groupby('agrupamento', as_index=False)['fila'].sum().rename(columns={'fila': 'fila_total'})
                
                if df_grouped['fila_total'].sum() > 0:
                    alloc_group = proportional_allocation(df_grouped['fila_total'].tolist(), int(vagas_40))
                    df_grouped['vagas alocadas'] = alloc_group
                    
                    detalhes = []
                    for _, row in df_grouped.iterrows():
                        agrup = row['agrupamento']
                        vagas_para_agrup = int(row['vagas alocadas'])
                        subset = df_40_filtrado[df_40_filtrado['agrupamento'] == agrup].copy()
                        
                        # Usar a COLUNA 'FILA' para a alocação proporcional dentro do agrupamento
                        fila_por_unidade_calculo = subset.groupby('unidade_original', as_index=False)['fila'].sum().rename(columns={'fila': 'fila_unidade_calculo'})
                        
                        # E pegar o valor da COLUNA 'REAL' para o resumo/exibição
                        real_por_unidade_resumo = subset.groupby('unidade_original', as_index=False)['real'].sum().rename(columns={'real': 'real_unidade_resumo'})
                        
                        df_temp = pd.merge(fila_por_unidade_calculo, real_por_unidade_resumo, on='unidade_original', how='left')

                        if vagas_para_agrup > 0:
                            allocs = proportional_allocation(df_temp['fila_unidade_calculo'].tolist(), vagas_para_agrup)
                            df_temp['vagas_alocadas'] = allocs
                        else:
                            df_temp['vagas_alocadas'] = 0
                        
                        detalhes.append(df_temp)

                    df_detalhado_40 = pd.concat(detalhes, ignore_index=True)
                    
                    # No resumo, renomear 'real_unidade_resumo' para ser a fila de 40%
                    resumo_40_final = df_detalhado_40.rename(columns={
                        'unidade_original': 'Unidade Solicitante',
                        'real_unidade_resumo': 'Fila 40% (Valor Real)', # Usando a coluna 'real' para o resumo
                        'vagas_alocadas': 'Cota 40%'
                    })
                    # Selecionar apenas as colunas relevantes para o resumo final e garantir que 'fila_unidade_calculo' não apareça indevidamente
                    resumo_40_final = resumo_40_final[['Unidade Solicitante', 'Fila 40% (Valor Real)', 'Cota 40%']]


                    # --- ETAPA 3: Consolidação dos Resultados ---
                    df_final = pd.merge(
                        resumo_60_final[['Unidade Solicitante', 'Cota 60%']],
                        resumo_40_final[['Unidade Solicitante', 'Cota 40%']],
                        on='Unidade Solicitante',
                        how='outer'
                    ).fillna(0)

                    df_final['Cota 60%'] = df_final['Cota 60%'].astype(int)
                    df_final['Cota 40%'] = df_final['Cota 40%'].astype(int)
                    df_final['Cota Total'] = df_final['Cota 60%'] + df_final['Cota 40%']
                    df_final_sorted = df_final.sort_values(by="Cota Total", ascending=False)
                    
                    # --- ETAPA 4: Geração do Arquivo de Filas Combinadas ---
                    # Merge para as filas, usando 'Fila 40% (Valor Real)' do resumo_40_final
                    df_filas_combinadas = pd.merge(
                        resumo_60_final[['Unidade Solicitante', 'Fila 60% (Nº de Solicitações)']],
                        resumo_40_final[['Unidade Solicitante', 'Fila 40% (Valor Real)']],
                        on='Unidade Solicitante',
                        how='outer'
                    ).fillna(0)
                    df_filas_combinadas['Fila 60% (Nº de Solicitações)'] = df_filas_combinadas['Fila 60% (Nº de Solicitações)'].astype(int)
                    df_filas_combinadas['Fila 40% (Valor Real)'] = df_filas_combinadas['Fila 40% (Valor Real)'].astype(int)

                    # --- EXIBIÇÃO E DOWNLOADS ---
                    st.markdown("---")
                    st.header("Resultado Consolidado da Distribuição de Cotas")
                    
                    # Filtra o dataframe apenas para exibição em tela
                    df_para_exibir = df_final_sorted[df_final_sorted['Cota Total'] > 0]
                    st.dataframe(df_para_exibir)

                    st.markdown("---")
                    st.header("Downloads")
                    
                    # Preparar arquivos para download (usa o dataframe completo, sem o filtro de 'Cota Total > 0')
                    csv_final = df_final_sorted.to_csv(index=False).encode('utf-8')
                    csv_distribuicao_60_detalhado = df_distribuicao_60_detalhado.to_csv(index=False).encode('utf-8')
                    csv_filas_combinadas = df_filas_combinadas.to_csv(index=False).encode('utf-8')

                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    with col_dl1:
                        st.download_button(
                            "Baixar Consolidado Final",
                            data=csv_final,
                            file_name="consolidado_total_cotas.csv",
                            mime="text/csv"
                        )
                    with col_dl2:
                        st.download_button(
                            "Baixar Detalhes da Fila 60% (com alocação)",
                            data=csv_distribuicao_60_detalhado,
                            file_name="detalhes_distribuicao_60.csv",
                            mime="text/csv"
                        )
                    with col_dl3:
                        st.download_button(
                            "Baixar Resumo das Filas (60% e 40%)",
                            data=csv_filas_combinadas,
                            file_name="resumo_filas_60_e_40.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("A soma da coluna 'Fila' no arquivo de 40% é 0. Não é possível distribuir as cotas desta etapa.")
    except Exception as e:
        st.error(f"Ocorreu um erro durante o processamento. Verifique os arquivos e os dados inseridos. Detalhe do erro: {e}")
else:
    st.warning("Por favor, carregue ambos os arquivos CSV para iniciar o processamento.")
