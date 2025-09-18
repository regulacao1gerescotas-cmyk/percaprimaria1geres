import streamlit as st
import pandas as pd
import numpy as np
import unicodedata

# ---------------- UTILITÁRIOS (Reutilizados do código anterior) ----------------
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

# ---------------- LISTA DE UNIDADES PERMITIDAS (Reutilizada) ----------------
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


# ---------------- NOVA INTERFACE DO STREAMLIT ----------------
st.set_page_config(layout="wide")
st.title("Perca Primária - I GERES")

st.markdown("---")
st.header("Carregamento de Arquivos")
col1, col2 = st.columns(2)
with col1:
    uploaded_file_cotas = st.file_uploader(
        "1. Arquivo de Cotas Recebidas (com 'Unidade', 'Item' e 'Cotas Recebidas')",
        type=["csv"],
        key="arquivo_cotas"
    )
with col2:
    uploaded_file_fila_pacientes = st.file_uploader(
        "2. Arquivo da Fila de Pacientes (com 'Unidade', 'Item' e 'Situação')",
        type=["csv"],
        key="arquivo_fila_pacientes"
    )

# --- LÓGICA DE PROCESSAMENTO PRINCIPAL ---
if uploaded_file_cotas is not None and uploaded_file_fila_pacientes is not None:
    try:
        # --- ETAPA 1: Processamento do Arquivo de Cotas Recebidas ---
        df_cotas_base = pd.read_csv(uploaded_file_cotas)
        df_cotas_base = normalizar_colunas(df_cotas_base)
        
        required_cols_cotas = {'unidade', 'item', 'cotas recebidas'}
        if not required_cols_cotas.issubset(df_cotas_base.columns):
            st.error(f"Arquivo de Cotas Recebidas não contém as colunas esperadas: {', '.join(required_cols_cotas)}.")
            st.stop()
            
        df_cotas_base['unidade_norm'] = df_cotas_base['unidade'].astype(str).apply(normalize_text)
        df_cotas_filtrado = df_cotas_base[df_cotas_base['unidade_norm'].isin(unidades_permitidas_norm)].copy()
        
        df_cotas_filtrado['item_norm'] = df_cotas_filtrado['item'].astype(str).apply(normalize_text)
        df_cotas_filtrado['cotas recebidas'] = pd.to_numeric(df_cotas_filtrado['cotas recebidas'], errors='coerce').fillna(0).astype(int)
        
        # Agrupar cotas recebidas por unidade e item
        cotas_recebidas_agrupado = df_cotas_filtrado.groupby(['unidade_norm', 'item_norm'], as_index=False)['cotas recebidas'].sum()
        cotas_recebidas_agrupado = cotas_recebidas_agrupado.rename(columns={'cotas recebidas': 'Cotas Recebidas'})

        # --- ETAPA 2: Processamento do Arquivo da Fila de Pacientes ---
        df_fila_base = pd.read_csv(uploaded_file_fila_pacientes)
        df_fila_base = normalizar_colunas(df_fila_base)

        required_cols_fila = {'unidade', 'item', 'situacao'}
        if not required_cols_fila.issubset(df_fila_base.columns):
            st.error(f"Arquivo da Fila de Pacientes não contém as colunas esperadas: {', '.join(required_cols_fila)}.")
            st.stop()
            
        df_fila_base['unidade_norm'] = df_fila_base['unidade'].astype(str).apply(normalize_text)
        df_fila_filtrado = df_fila_base[df_fila_base['unidade_norm'].isin(unidades_permitidas_norm)].copy()
        
        df_fila_filtrado['item_norm'] = df_fila_filtrado['item'].astype(str).apply(normalize_text)
        df_fila_filtrado['situacao_norm'] = df_fila_filtrado['situacao'].astype(str).apply(normalize_text)
        
        # Filtrar apenas os itens que apareceram na planilha de cotas (para garantir consistência)
        itens_com_cotas = cotas_recebidas_agrupado['item_norm'].unique()
        df_fila_filtrado = df_fila_filtrado[df_fila_filtrado['item_norm'].isin(itens_com_cotas)]

        # Contar pacientes com status 'MARCADO' por unidade e item
        pacientes_marcados = df_fila_filtrado[df_fila_filtrado['situacao_norm'] == 'marcado']
        total_pacientes_marcados = pacientes_marcados.groupby(['unidade_norm', 'item_norm'], as_index=False).size().rename('Total Pacientes Marcados')

        # --- ETAPA 3: Consolidação dos Resultados ---
        df_consolidado = pd.merge(
            cotas_recebidas_agrupado,
            total_pacientes_marcados,
            on=['unidade_norm', 'item_norm'],
            how='left'
        ).fillna(0) # Preenche 0 para unidades/itens que não tiveram pacientes marcados

        df_consolidado['Total Pacientes Marcados'] = df_consolidado['Total Pacientes Marcados'].astype(int)
        
        df_consolidado['Cotas para Redistribuir'] = df_consolidado['Cotas Recebidas'] - df_consolidado['Total Pacientes Marcados']
        
        # Filtrar apenas as cotas positivas para redistribuição
        df_redistribuir = df_consolidado[df_consolidado['Cotas para Redistribuir'] > 0].copy()
        
        # Mapear de volta para os nomes originais das unidades e itens, se desejar
        # Para unidades, podemos tentar um mapeamento simples da primeira ocorrência
        unidade_map_back = df_cotas_base.set_index('unidade_norm')['unidade'].to_dict()
        item_map_back = df_cotas_base.set_index('item_norm')['item'].to_dict()

        df_redistribuir['Unidade Original'] = df_redistribuir['unidade_norm'].map(unidade_map_back).fillna(df_redistribuir['unidade_norm'])
        df_redistribuir['Item Original'] = df_redistribuir['item_norm'].map(item_map_back).fillna(df_redistribuir['item_norm'])

        df_redistribuir_final = df_redistribuir[[
            'Unidade Original', 'Item Original', 'Cotas Recebidas', 'Total Pacientes Marcados', 'Cotas para Redistribuir'
        ]].sort_values(by=['Unidade Original', 'Item Original'])
        
        # --- EXIBIÇÃO E DOWNLOADS ---
        st.markdown("---")
        st.header("Cotas Disponíveis para Redistribuição")
        
        if not df_redistribuir_final.empty:
            st.write("A tabela abaixo mostra as unidades e itens que possuem cotas não utilizadas, disponíveis para redistribuição:")
            st.dataframe(df_redistribuir_final)
            
            total_geral_redistribuir = df_redistribuir_final['Cotas para Redistribuir'].sum()
            st.success(f"**Total geral de cotas disponíveis para redistribuição:** {total_geral_redistribuir}")

            st.markdown("---")
            st.header("Download do Resultado")
            csv_redistribuir = df_redistribuir_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Baixar Tabela de Cotas para Redistribuição",
                data=csv_redistribuir,
                file_name="cotas_para_redistribuir.csv",
                mime="text/csv"
            )
        else:
            st.info("Nenhuma cota foi identificada para redistribuição com base nos arquivos fornecidos.")

    except Exception as e:
        st.error(f"Ocorreu um erro durante o processamento. Verifique os arquivos e os dados. Detalhe do erro: {e}")
else:
    st.warning("Por favor, carregue ambos os arquivos CSV para iniciar o processamento.")

