import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def calculate_detailed_savings(df_actual, df_potential):
    # Calcular valor actual por PdV, vendor y producto
    df_actual['valor_actual'] = df_actual['unidades_pedidas'] * df_actual['precio_minimo']
    
    # Agrupar datos actuales
    actual_sales = df_actual.groupby(['point_of_sale_id', 'vendor_id', 'super_catalog_id']).agg({
        'valor_actual': 'sum',
        'unidades_pedidas': 'sum'
    }).reset_index()
    
    # Preparar datos potenciales
    potential_sales = df_potential[['point_of_sale_id', 'vendor_id', 'super_catalog_id', 
                                  'precio_total_vendedor', 'unidades_pedidas', 'precio_vendedor']]
    
    # Merge para comparar
    savings_analysis = pd.merge(
        actual_sales,
        potential_sales,
        on=['point_of_sale_id', 'vendor_id', 'super_catalog_id'],
        how='outer',
        suffixes=('_actual', '_potential')
    ).fillna(0)
    
    # Calcular ahorro por producto
    savings_analysis['ahorro_por_producto'] = savings_analysis['valor_actual'] - savings_analysis['precio_total_vendedor']
    
    # Agregar porcentaje de ahorro
    savings_analysis['porcentaje_ahorro'] = np.where(
        savings_analysis['valor_actual'] > 0,
        (savings_analysis['ahorro_por_producto'] / savings_analysis['valor_actual']) * 100,
        0
    )
    
    return savings_analysis

def create_savings_dashboard(savings_df):
    # Agrupar por PdV y vendor
    pdv_summary = savings_df.groupby(['point_of_sale_id', 'vendor_id']).agg({
        'valor_actual': 'sum',
        'precio_total_vendedor': 'sum',
        'ahorro_por_producto': 'sum'
    }).reset_index()
    
    pdv_summary['porcentaje_ahorro_total'] = np.where(
        pdv_summary['valor_actual'] > 0,
        (pdv_summary['ahorro_por_producto'] / pdv_summary['valor_actual']) * 100,
        0
    )
    
    return pdv_summary

def show_savings_analysis():
    st.title("💰 Análisis de Ahorro Potencial por PdV y Vendor")
    
    try:
        # Cargar datos
        df_actual = pd.read_csv('orders_delivered_pos_vendor_geozone.csv')
        df_potential = pd.read_csv('top_5_productos_geozona.csv')
        
        # Calcular ahorros detallados
        savings_analysis = calculate_detailed_savings(df_actual, df_potential)
        pdv_summary = create_savings_dashboard(savings_analysis)
        
        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            selected_pdv = st.selectbox(
                "Seleccionar Punto de Venta",
                options=sorted(pdv_summary['point_of_sale_id'].unique())
            )
        
        with col2:
            selected_vendor = st.selectbox(
                "Seleccionar Vendor",
                options=sorted(pdv_summary['vendor_id'].unique())
            )
        
        # Mostrar resumen para el PdV y vendor seleccionados
        filtered_summary = pdv_summary[
            (pdv_summary['point_of_sale_id'] == selected_pdv) &
            (pdv_summary['vendor_id'] == selected_vendor)
        ]
        
        if not filtered_summary.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Valor Actual Total",
                    f"${filtered_summary['valor_actual'].iloc[0]:,.2f}"
                )
            with col2:
                st.metric(
                    "Valor Potencial Total",
                    f"${filtered_summary['precio_total_vendedor'].iloc[0]:,.2f}"
                )
            with col3:
                st.metric(
                    "Ahorro Total Potencial",
                    f"${filtered_summary['ahorro_por_producto'].iloc[0]:,.2f}",
                    f"{filtered_summary['porcentaje_ahorro_total'].iloc[0]:.1f}%"
                )
        
        # Mostrar detalle de productos
        st.subheader("Detalle de Productos y Ahorros")
        filtered_products = savings_analysis[
            (savings_analysis['point_of_sale_id'] == selected_pdv) &
            (savings_analysis['vendor_id'] == selected_vendor)
        ]
        
        if not filtered_products.empty:
            st.dataframe(
                filtered_products[['super_catalog_id', 'unidades_pedidas_actual', 
                                 'valor_actual', 'precio_total_vendedor', 
                                 'ahorro_por_producto', 'porcentaje_ahorro']]
                .sort_values('ahorro_por_producto', ascending=False)
                .style.format({
                    'valor_actual': '${:,.2f}',
                    'precio_total_vendedor': '${:,.2f}',
                    'ahorro_por_producto': '${:,.2f}',
                    'porcentaje_ahorro': '{:,.1f}%',
                    'unidades_pedidas_actual': '{:,.0f}'
                })
            )
        
        # Mostrar top PdV con mayor potencial de ahorro
        st.subheader("Top 20 Combinaciones PdV-Vendor con Mayor Potencial de Ahorro")
        top_savings = pdv_summary.nlargest(20, 'ahorro_por_producto')
        
        st.dataframe(
            top_savings[['point_of_sale_id', 'vendor_id', 'valor_actual', 
                        'precio_total_vendedor', 'ahorro_por_producto', 'porcentaje_ahorro_total']]
            .style.format({
                'valor_actual': '${:,.2f}',
                'precio_total_vendedor': '${:,.2f}',
                'ahorro_por_producto': '${:,.2f}',
                'porcentaje_ahorro_total': '{:,.1f}%'
            })
        )
        
    except Exception as e:
        st.error(f"Error al procesar los datos: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_savings_analysis()