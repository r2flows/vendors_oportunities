import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Ahorros en Farmacias",
    page_icon="💊",
    layout="wide"
)

# Función para cargar datos
@st.cache_data
def load_data():
    orders = pd.read_csv('orders_delivered_pos_vendor_geozone.csv')
    top_products = pd.read_csv('top_5_productos_geozona.csv')
    return orders, top_products

def calculate_savings(orders_df, top_products_df, selected_pharmacy):
    # Filtrar datos para la farmacia seleccionada
    pharmacy_orders = orders_df[orders_df['point_of_sale_id'] == selected_pharmacy]
    pharmacy_top_products = top_products_df[top_products_df['point_of_sale_id'] == selected_pharmacy]
    
    # Calcular gastos actuales
    current_spend = (pharmacy_orders['precio_minimo'] * pharmacy_orders['unidades_pedidas']).sum()
    
    # Calcular gastos optimizados usando los precios de top_products
    optimized_spend = (pharmacy_top_products['precio_vendedor'] * pharmacy_top_products['unidades_pedidas']).sum()
    
    # Calcular métricas de ahorro
    total_savings = current_spend - optimized_spend
    savings_percentage = (total_savings / current_spend) * 100 if current_spend > 0 else 0
    
    return {
        'current_spend': current_spend,
        'optimized_spend': optimized_spend,
        'total_savings': total_savings,
        'savings_percentage': savings_percentage,
        'total_orders': len(pharmacy_orders),
        'optimized_products': len(pharmacy_top_products)
    }

def main():
    # Título de la aplicación
    st.title("💊 Análisis de Ahorros en Farmacias")
    
    # Cargar datos
    orders_df, top_products_df = load_data()
    
    # Selector de farmacia
    pharmacies = sorted(orders_df['point_of_sale_id'].unique())
    selected_pharmacy = st.selectbox(
        "Selecciona una farmacia:",
        pharmacies,
        format_func=lambda x: f"Farmacia {x}"
    )
    
    # Calcular métricas
    metrics = calculate_savings(orders_df, top_products_df, selected_pharmacy)
    
    # Mostrar métricas en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Gasto Actual",
            f"${metrics['current_spend']:,.2f}",
        )
    
    with col2:
        st.metric(
            "Gasto Optimizado",
            f"${metrics['optimized_spend']:,.2f}",
        )
    
    with col3:
        st.metric(
            "Ahorro Total",
            f"${metrics['total_savings']:,.2f}",
        )
    
    with col4:
        st.metric(
            "Porcentaje de Ahorro",
            f"{metrics['savings_percentage']:.1f}%",
        )
    
    # Análisis detallado
    st.header("Análisis Detallado")
    
    # Crear dos columnas para los gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 productos con mayor ahorro potencial
        pharmacy_products = top_products_df[
            top_products_df['point_of_sale_id'] == selected_pharmacy
        ].copy()
        
        pharmacy_products['ahorro_por_unidad'] = (
            pharmacy_products['precio_minimo'] - pharmacy_products['precio_vendedor']
        )
        pharmacy_products['ahorro_total'] = (
            pharmacy_products['ahorro_por_unidad'] * pharmacy_products['unidades_pedidas']
        )
        
        top_savings = pharmacy_products.nlargest(10, 'ahorro_total')
        
        fig = px.bar(
            top_savings,
            x='super_catalog_id',
            y='ahorro_total',
            title='Top 10 Productos con Mayor Ahorro Potencial',
            labels={'super_catalog_id': 'ID del Producto', 'ahorro_total': 'Ahorro Total ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Comparación de precios actuales vs optimizados
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Precio Actual',
            x=top_savings['super_catalog_id'],
            y=top_savings['precio_minimo'],
        ))
        
        fig.add_trace(go.Bar(
            name='Precio Optimizado',
            x=top_savings['super_catalog_id'],
            y=top_savings['precio_vendedor'],
        ))
        
        fig.update_layout(
            title='Comparación de Precios: Actual vs Optimizado',
            xaxis_title='ID del Producto',
            yaxis_title='Precio ($)',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de productos optimizados
    st.header("Detalle de Productos Optimizados")
    
    detailed_products = pharmacy_products[
        ['super_catalog_id', 'unidades_pedidas', 'precio_minimo', 
         'precio_vendedor', 'ahorro_por_unidad', 'ahorro_total']
    ].copy()
    
    detailed_products.columns = [
        'ID Producto', 'Unidades Pedidas', 'Precio Actual', 
        'Precio Optimizado', 'Ahorro por Unidad', 'Ahorro Total'
    ]
    
    # Formatear números en la tabla
    for col in ['Precio Actual', 'Precio Optimizado', 'Ahorro por Unidad', 'Ahorro Total']:
        detailed_products[col] = detailed_products[col].map('${:,.2f}'.format)
    
    st.dataframe(
        detailed_products.sort_values('Ahorro Total', ascending=False),
        use_container_width=True
    )

if __name__ == "__main__":
    main()