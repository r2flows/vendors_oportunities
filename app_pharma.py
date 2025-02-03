import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def process_data_with_mappings():
    df_actual = pd.read_csv('orders_delivered_pos_vendor_geozone.csv')
    df_potential = pd.read_csv('top_5_productos_geozona.csv')
    vendor_relations = pd.read_csv('vendor_pos_relations.csv')
    
    # Vendor mapping
    vendor_mapping = {
        '10249':'1141','10250':'1142','10260':'1148','10267':'1150',
        '10268':'1151','10269':'1152','10270':'1153','10271':'1154',
        '10272':'1155','10273':'1156','10274':'1157','10275':'1158',
        '10276':'1159','10277':'1160','10278':'1161','10279':'1162',
        '10280':'1163','10281':'1164','10282':'1165','10479':'1275',
        '10608':'1303'
    }
    
    # Preparar df_actual
    df_actual['vendor_id'] = df_actual['vendor_id'].astype(str)
    df_actual['vendor_id_original'] = df_actual['vendor_id']
    df_actual['vendor_id_mapped'] = df_actual['vendor_id'].map(vendor_mapping).fillna(df_actual['vendor_id'])
    
    # Calcular valor actual una sola vez
    df_actual['valor_actual'] = df_actual['unidades_pedidas'] * df_actual['precio_minimo']
    
    # Agrupar por PdV, vendor y producto para evitar duplicados
    df_actual_grouped = df_actual.groupby(
        ['point_of_sale_id', 'vendor_id_mapped', 'super_catalog_id']
    )['valor_actual'].sum().reset_index()
    
    return df_actual_grouped, df_potential, vendor_relations

def calculate_savings_analysis():
    df_actual, df_potential, vendor_relations = process_data_with_mappings()
    savings_analysis = []

    # Análisis para PdV con relaciones comerciales
    for _, relation in vendor_relations.iterrows():
        pdv = relation['point_of_sale_id']
        vendor = str(relation['vendor_id'])
        status = relation['status']
        
        # Valor actual solo para este vendor
        actual_value = df_actual[
            (df_actual['point_of_sale_id'] == pdv) & 
            (df_actual['vendor_id_mapped'] == vendor)
        ]['valor_actual'].sum()
        
        # Valor potencial
        potential_value = df_potential[
            (df_potential['point_of_sale_id'] == pdv) & 
            (df_potential['vendor_id'] == vendor)
        ]['precio_total_vendedor'].sum()
        
        if actual_value > 0 or potential_value > 0:
            savings_analysis.append({
                'point_of_sale_id': pdv,
                'vendor_id': vendor,
                'status': status,
                'valor_actual': actual_value,
                'valor_potencial': potential_value,
                'ahorro_potencial': actual_value - potential_value if potential_value > 0 else 0,
                'tipo': 'Con Relación'
            })

    # Análisis para PdV sin relaciones comerciales mapeadas
    all_pdvs = set(df_actual['point_of_sale_id'].unique())
    related_pdvs = set(vendor_relations['point_of_sale_id'].unique())
    unrelated_pdvs = all_pdvs - related_pdvs

    for pdv in unrelated_pdvs:
        actual_value = df_actual[
            df_actual['point_of_sale_id'] == pdv
        ]['valor_actual'].sum()
        
        if actual_value > 0:
            # Encontrar mejor oferta potencial
            potential_options = df_potential[df_potential['point_of_sale_id'] == pdv]
            best_potential = potential_options['precio_total_vendedor'].min() if not potential_options.empty else 0
            
            savings_analysis.append({
                'point_of_sale_id': pdv,
                'vendor_id': 'Sin Relación',
                'status': None,
                'valor_actual': actual_value,
                'valor_potencial': best_potential,
                'ahorro_potencial': actual_value - best_potential if best_potential > 0 else 0,
                'tipo': 'Sin Relación'
            })
    
    return pd.DataFrame(savings_analysis)

def show_savings_dashboard():
    st.title("📊 Análisis de Ahorro Potencial")
    
    try:
        savings_df = calculate_savings_analysis()
        
        # Métricas generales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valor Total Actual", 
                     f"${savings_df['valor_actual'].sum():,.2f}")
        with col2:
            st.metric("Valor Total Potencial", 
                     f"${savings_df['valor_potencial'].sum():,.2f}")
        with col3:
            st.metric("Ahorro Total Posible", 
                     f"${savings_df['ahorro_potencial'].sum():,.2f}")
        
        # Análisis Detallado
        st.subheader("🔍 Análisis Detallado")
        tipo_relacion = st.selectbox(
            "Filtrar por tipo de relación",
            ["Todos", "Con Relación", "Sin Relación"]
        )
        
        filtered_df = savings_df if tipo_relacion == "Todos" else savings_df[savings_df['tipo'] == tipo_relacion]
        
        st.dataframe(
            filtered_df.sort_values('ahorro_potencial', ascending=False)
            .style.format({
                'valor_actual': '${:,.2f}',
                'valor_potencial': '${:,.2f}',
                'ahorro_potencial': '${:,.2f}'
            })
        )
        
        # Gráfico de distribución
        fig = go.Figure()
        for tipo in filtered_df['tipo'].unique():
            data = filtered_df[filtered_df['tipo'] == tipo]['ahorro_potencial']
            fig.add_trace(go.Box(
                y=data,
                name=tipo,
                boxpoints='all'
            ))
        
        fig.update_layout(
            title='Distribución de Ahorros Potenciales por Tipo de Relación',
            yaxis_title='Ahorro Potencial ($)',
            showlegend=True
        )
        
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_savings_dashboard()