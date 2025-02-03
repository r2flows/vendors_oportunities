import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def process_data_with_mappings():
    # Cargar datos
    df_actual = pd.read_csv('orders_delivered_pos_vendor_geozone.csv')
    df_potential = pd.read_csv('top_5_productos_geozona.csv')
    vendor_relations = pd.read_csv('vendor_pos_relations.csv')
    
    # Mapping de vendors
    vendor_mapping = {
        '10249':'1141','10250':'1142','10260':'1148','10267':'1150',
        '10268':'1151','10269':'1152','10270':'1153','10271':'1154',
        '10272':'1155','10273':'1156','10274':'1157','10275':'1158',
        '10276':'1159','10277':'1160','10278':'1161','10279':'1162',
        '10280':'1163','10281':'1164','10282':'1165','10479':'1275',
        '10608':'1303'
    }
    
    # Aplicar mapping a df_actual
    df_actual['vendor_id'] = df_actual['vendor_id'].astype(str)
    df_actual['vendor_id'] = df_actual['vendor_id'].map(vendor_mapping).fillna(df_actual['vendor_id'])
    
    # Separar compras actuales entre vendors mapeados y no mapeados
    mapped_vendors = set(vendor_mapping.values())
    df_actual_mapped = df_actual[df_actual['vendor_id'].isin(mapped_vendors)]
    df_actual_unmapped = df_actual[~df_actual['vendor_id'].isin(mapped_vendors)]
    
    return df_actual_mapped, df_actual_unmapped, df_potential, vendor_relations

def calculate_savings_analysis():
    df_actual_mapped, df_actual_unmapped, df_potential, vendor_relations = process_data_with_mappings()
    
    # Calcular valor actual para vendors mapeados
    df_actual_mapped['valor_actual'] = df_actual_mapped['unidades_pedidas'] * df_actual_mapped['precio_minimo']
    
    # Calcular valor actual para vendors no mapeados
    df_actual_unmapped['valor_actual'] = df_actual_unmapped['unidades_pedidas'] * df_actual_unmapped['precio_minimo']
    
    # Análisis por PdV con relación comercial
    savings_analysis = []
    
    for _, relation in vendor_relations.iterrows():
        pdv = relation['point_of_sale_id']
        vendor = str(relation['vendor_id'])
        status = relation['status']
        
        # Ventas actuales
        actual_sales = df_actual_mapped[
            (df_actual_mapped['point_of_sale_id'] == pdv) & 
            (df_actual_mapped['vendor_id'] == vendor)
        ]['valor_actual'].sum()
        
        # Valor potencial
        potential_value = df_potential[
            (df_potential['point_of_sale_id'] == pdv) & 
            (df_potential['vendor_id'] == vendor)
        ]['precio_total_vendedor'].sum()
        
        savings_analysis.append({
            'point_of_sale_id': pdv,
            'vendor_id': vendor,
            'status': status,
            'valor_actual': actual_sales,
            'valor_potencial': potential_value,
            'ahorro_potencial': actual_sales - potential_value if actual_sales > 0 else 0,
            'tipo': 'Con Relación'
        })
    
    # Análisis para vendors no mapeados
    unmapped_pdvs = df_actual_unmapped['point_of_sale_id'].unique()
    
    for pdv in unmapped_pdvs:
        # Valor actual con vendors no mapeados
        actual_value = df_actual_unmapped[
            df_actual_unmapped['point_of_sale_id'] == pdv
        ]['valor_actual'].sum()
        
        # Mejor valor potencial disponible
        best_potential = df_potential[
            df_potential['point_of_sale_id'] == pdv
        ]['precio_total_vendedor'].min() if len(df_potential[df_potential['point_of_sale_id'] == pdv]) > 0 else 0
        
        if actual_value > 0:
            savings_analysis.append({
                'point_of_sale_id': pdv,
                'vendor_id': 'N/A',
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
        total_actual = savings_df['valor_actual'].sum()
        total_potential = savings_df['valor_potencial'].sum()
        total_savings = savings_df['ahorro_potencial'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valor Total Actual", f"${total_actual:,.2f}")
        with col2:
            st.metric("Valor Total Potencial", f"${total_potential:,.2f}")
        with col3:
            st.metric("Ahorro Total Posible", f"${total_savings:,.2f}")
        
        # Filtros
        st.subheader("🔍 Análisis Detallado")
        tipo_relacion = st.selectbox(
            "Filtrar por tipo de relación",
            ["Todos", "Con Relación", "Sin Relación"]
        )
        
        if tipo_relacion != "Todos":
            filtered_df = savings_df[savings_df['tipo'] == tipo_relacion]
        else:
            filtered_df = savings_df
        
        # Mostrar tabla de ahorros
        st.dataframe(
            filtered_df.sort_values('ahorro_potencial', ascending=False)
            .style.format({
                'valor_actual': '${:,.2f}',
                'valor_potencial': '${:,.2f}',
                'ahorro_potencial': '${:,.2f}'
            })
        )
        
        # Gráfico de distribución de ahorros
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=filtered_df['ahorro_potencial'],
            name='Distribución de Ahorros',
            boxpoints='all'
        ))
        
        fig.update_layout(
            title='Distribución de Ahorros Potenciales',
            yaxis_title='Ahorro Potencial ($)',
            showlegend=False
        )
        
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error en el procesamiento: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_savings_dashboard()