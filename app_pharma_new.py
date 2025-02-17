import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Análisis de Compras y Productos POS", layout="wide")

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Compras Reales vs Potenciales"
if 'show_winners' not in st.session_state:
    st.session_state.show_winners = True
if 'show_semi_winners' not in st.session_state:
    st.session_state.show_semi_winners = False
if 'show_medium_winners' not in st.session_state:
    st.session_state.show_medium_winners = False

# Título
st.title("Análisis de Compras Reales vs Potenciales por Punto de Venta")

# Función para mapear status a texto descriptivo
def get_status_description(status):
    if pd.isna(status):
        return "Sin Status"
    status_map = {
        1: "Activo",
        2: "Pendiente"
    }
    return status_map.get(status, f"Status {status}")

# Función para obtener el status del vendor desde df_products
def get_vendor_status(vendor_id, pos_id, df_products):
    vendor_status = df_products[
        (df_products['vendor_id'] == vendor_id) & 
        (df_products['point_of_sale_id'] == pos_id)
    ]['status'].unique()
    
    return [get_status_description(s) for s in vendor_status] if len(vendor_status) > 0 else ["Sin Status"]

def get_slider_range(potential_value, min_purchase):
    """
    Calcula un rango seguro para el slider basado en los valores disponibles.
    
    Args:
        potential_value (float): Valor de compra potencial
        min_purchase (float): Valor de compra mínima
    
    Returns:
        tuple: (min_value, max_value, default_value)
    """
    # Asegurar que los valores sean numéricos y no negativos
    potential_value = float(max(0, potential_value or 0))
    min_purchase = float(max(0, min_purchase or 0))
    
    # Establecer un valor base mínimo
    BASE_MIN = 100.0
    
    # Calcular el máximo como el mayor entre varios valores
    max_value = max(
        potential_value * 2,  # 2 veces la compra potencial
        min_purchase * 2,     # 2 veces la compra mínima
        BASE_MIN * 2         # 2 veces el valor base
    )
    
    # Asegurar que el máximo nunca sea 0 o igual al mínimo
    max_value = max(max_value, BASE_MIN)
    
    # Establecer el valor por defecto
    default_value = max(min_purchase, BASE_MIN/10)
    
    return (0.0, max_value, default_value)
# Cargar y procesar datos
@st.cache_data
def load_and_process_data():
    # Cargar ambos CSVs
    df_orders = pd.read_csv('orders_delivered_pos_vendor_geozone.csv')
    df_products = pd.read_csv('top_5_productos_geozona.csv')
    df_min_purchase = pd.read_csv('minimum_purchase.csv')
    df_semi_ganadores = pd.read_csv('productos_clasificados.csv')

    # Filtrar registros donde status != 0
    #df_products = df_products[df_products['status'] != 0]
    
    # Calcular el valor total por cada combinación POS-Vendor
    df_orders['total_compra'] = df_orders['unidades_pedidas'] * df_orders['precio_minimo']
    
    # Calcular promedios por orden para cada POS
    order_totals = df_orders.groupby(['point_of_sale_id', 'order_id'])['total_compra'].sum().reset_index()
    pos_order_stats = order_totals.groupby('point_of_sale_id').agg({
        'total_compra': ['mean', 'count']
    }).reset_index()
    pos_order_stats.columns = ['point_of_sale_id', 'promedio_por_orden', 'numero_ordenes']
    
    # Agregar por POS y Vendor
    pos_vendor_totals = df_orders.groupby(['point_of_sale_id', 'vendor_id'])['total_compra'].sum().reset_index()
    
    # Obtener status por POS-Vendor (excluyendo status = 0)
    pos_vendor_status = df_products.groupby(['point_of_sale_id', 'vendor_id'])['status'].agg(lambda x: list(x.unique())).reset_index()
    
    return pos_vendor_totals, df_orders, df_products, pos_vendor_status, pos_order_stats, df_min_purchase, df_semi_ganadores

try:
    pos_vendor_totals, df_original, df_products, pos_vendor_status, pos_order_stats, df_min_purchase, df_semi_ganadores = load_and_process_data()

    # Sidebar para filtros
    st.sidebar.header("Filtros")
    
    # Filtro de punto de venta
    pos_list = sorted(list(set(pos_vendor_totals['point_of_sale_id']) & set(df_products['point_of_sale_id'])))
    selected_pos = st.sidebar.selectbox(
        "Seleccionar Punto de Venta",
        options=pos_list
    )
    category_container = st.sidebar.container()

    # Crear pestañas
    tab1, tab2, tab3 = st.tabs(["Compras Reales vs Potenciales", "Productos Ganadores", "Análisis de Órdenes vs Mínimos"])
    show_winners = True
    show_semi_winners = False
    show_medium_winners = False
    with tab1:
        st.session_state.current_tab = "Compras Reales vs Potenciales"
        # Filtrar datos para el POS seleccionado
        pos_data = pos_vendor_totals[pos_vendor_totals['point_of_sale_id'] == selected_pos]
        pos_data = pos_data.sort_values('total_compra', ascending=False)

        #Obtener estadísticas de órdenes
        pos_stats = pos_order_stats[pos_order_stats['point_of_sale_id'] == selected_pos].iloc[0]
        st.subheader("Información del Punto de Venta")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                f"Total de Compras - POS {selected_pos}", 
                f"${pos_data['total_compra'].sum():,.2f}"
            )
        with col2:
            st.metric(
                "Promedio por Orden",
                f"${pos_stats['promedio_por_orden']:,.2f}"
            )
        with col3:
            st.metric(
                "Número de Órdenes",
                f"{int(pos_stats['numero_ordenes']):,}"
            )
        pos_info = df_original[df_original['point_of_sale_id'] == selected_pos].iloc[0]
        
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("País", pos_info['country'])
        with info_col2:
            st.metric("Zona Geográfica", pos_info['geo_zone'])
        with info_col3:
            st.metric("Total Vendors", len(pos_data))

        # Tabla detallada de vendors
        
        st.subheader("Detalle de Compras por Droguería/Vendor")
        pos_data['porcentaje'] = (pos_data['total_compra'] / pos_data['total_compra'].sum()) * 100
        detail_table = pos_data.copy()
        detail_table.columns = ['POS ID', 'Droguería/Vendor ID', 'Total Comprado', 'Porcentaje']
        detail_table = detail_table.round({'Porcentaje': 2})
        
        st.dataframe(
            detail_table.style.format({
                'Total Comprado': '${:,.2f}',
                'Porcentaje': '{:.2f}%'
            })
        )
        st.subheader("Análisis de Intersección de Productos")

# Calcular intersección para el POS seleccionado
        df_products['precio_total_droguería']=df_products['unidades_pedidas']*df_products['precio_minimo']

        orders_pos = df_original[df_original['point_of_sale_id'] == selected_pos]
        products_pos = df_products[df_products['point_of_sale_id'] == selected_pos]
        orders_products = set(orders_pos['super_catalog_id'])
        top_products = set(products_pos['super_catalog_id'])
        intersection = pd.merge(df_products, orders_pos, on=['super_catalog_id', 'point_of_sale_id'], how='inner')

        intersection_percentage = (len(intersection) / len(orders_products) * 100) if orders_products else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Productos en Compras Reales", f"{len(orders_products):,}")
        with col2:
            st.metric("Total Productos Ganadores Disponibles", f"{len(top_products):,}")
        with col3:
            st.metric("Productos en Intersección", f"{len(intersection):,} ({intersection_percentage:.2f}%)")

# Mostrar productos en la intersección
        if not intersection.empty:  # <- Aquí está la corrección
            st.subheader("Productos en la Intersección")
            #intersection_products = df_products[
             #   (df_products['point_of_sale_id'] == selected_pos) & 
              #  (df_products['super_catalog_id'].isin(intersection['super_catalog_id']))
        #]

    # Calcular valores totales
        #orders_total = orders_pos[orders_pos['super_catalog_id'].isin(intersection['super_catalog_id'])]['unidades_pedidas'].mul(
         #   orders_pos[orders_pos['super_catalog_id'].isin(intersection['super_catalog_id'])]['precio_minimo']
        #).sum()
        orders_total = intersection['precio_total_droguería'].sum()
        products_total = intersection['valor_total_vendedor'].sum()

        value_col1, value_col2, value_col3 = st.columns(3)
        with value_col1:
            st.metric("Valor Total en Compras Reales", f"${orders_total:,.2f}")
        with value_col2:
            st.metric("Valor Total en Productos Ganadores", f"${products_total:,.2f}")
        with value_col3:
            # Calcular el porcentaje de ahorro
            savings_percentage = ((orders_total - products_total) / orders_total * 100) if orders_total > 0 else 0
            st.metric("Ahorro Potencial", f"{savings_percentage:.2f}%")


    with tab2:
        st.session_state.current_tab = "Productos Ganadores"

        # Añadir checkboxes en el sidebar para Tab2
        with category_container:
            st.markdown("---")
            st.subheader("Categorías de Productos")
            show_winners = st.checkbox("Productos Ganadores", value=True, key='cb_winners_tab2')
            show_semi_winners = st.checkbox("Productos Semi Ganadores", value=False, key='cb_semi_winners_tab2')
            show_medium_winners = st.checkbox("Productos Medianamente Ganadores", value=False, key='cb_medium_winners_tab2')

        if not any([show_winners, show_semi_winners, show_medium_winners]):
            st.warning("Por favor seleccione al menos una categoría de productos para visualizar.")
            st.stop()

        # Filtrar productos para el POS seleccionado
        products_data = df_products[df_products['point_of_sale_id'] == selected_pos].copy()
        products_data = products_data.sort_values('valor_total_vendedor', ascending=False)

        # Crear un conjunto de productos ganadores para este POS
        productos_ganadores = set(products_data['super_catalog_id'].unique())
        #st.dataframe(df_semi_ganadores)
        # Procesar datos de semi ganadores
        df_semi_ganadores['precio_vendedor_total'] = df_semi_ganadores['precio_minimo_orders']*0.99*df_semi_ganadores['unidades_pedidas']
        semi_ganadores_data = df_semi_ganadores[
            (df_semi_ganadores['point_of_sale_id'] == selected_pos) &
            (df_semi_ganadores['categoria_diferencia'] == 'hasta_5_porciento')
        ].copy()
        
        # Eliminar productos que ya están en ganadores
        semi_ganadores_data = semi_ganadores_data[
            ~semi_ganadores_data['super_catalog_id'].isin(productos_ganadores)
        ]
        semi_ganadores_data = semi_ganadores_data.sort_values('precio_vendedor_total', ascending=False)

        # Crear conjunto de productos semi ganadores
        productos_semi_ganadores = set(semi_ganadores_data['super_catalog_id'].unique())

        # Procesar datos de medianamente ganadores
        medianamente_ganadores_data = df_semi_ganadores[
            (df_semi_ganadores['point_of_sale_id'] == selected_pos) &
            (df_semi_ganadores['categoria_diferencia'] == 'hasta_15_porciento')
        ].copy()
        
        # Eliminar productos que están en ganadores o semi ganadores
        medianamente_ganadores_data = medianamente_ganadores_data[
            ~medianamente_ganadores_data['super_catalog_id'].isin(productos_ganadores) &
            ~medianamente_ganadores_data['super_catalog_id'].isin(productos_semi_ganadores)
        ]
        medianamente_ganadores_data = medianamente_ganadores_data.sort_values('precio_vendedor_total', ascending=False)

        # Mostrar productos ganadores si está seleccionado
        if show_winners:
            st.subheader("Venta Potencial Productos Ganadores")
            total_valor = products_data['valor_total_vendedor'].sum()
            st.metric("Valor Total de Productos Ganadores", f"${total_valor:,.2f}")

            products_table = products_data[[
                'super_catalog_id', 
                'unidades_pedidas', 
                'vendor_id',
                'valor_total_vendedor'
            ]].copy()
        
            products_table.columns = [
                'ID Producto', 
                'Unidades Pedidas', 
                'Vendor ID',
                'Valor Total'
            ]
        
            st.dataframe(
                products_table.style.format({
                    'Unidades Pedidas': '{:.0f}',
                    'Valor Total': '${:.2f}'
                })
            )

        # Mostrar semi ganadores si está seleccionado
        if show_semi_winners:
            st.subheader("Venta Potencial Semi Ganadores")
            total_valor = semi_ganadores_data['precio_vendedor_total'].sum()
            st.metric("Valor Total de Productos Semi Ganadores", f"${total_valor:,.2f}")

            semi_ganadores_table = semi_ganadores_data[[
                'super_catalog_id', 
                'unidades_pedidas', 
                'vendor_id',
                'precio_vendedor_total'
            ]].copy()
        
            semi_ganadores_table.columns = [
                'ID Producto', 
                'Unidades Pedidas', 
                'Vendor ID',
                'Valor Total'
            ]
        
            st.dataframe(
                semi_ganadores_table.style.format({
                    'Unidades Pedidas': '{:.0f}',
                    'Valor Total': '${:.2f}'
                })
            )

        # Mostrar medianamente ganadores si está seleccionado
        if show_medium_winners:
            st.subheader("Venta Potencial Medianamente Ganadores")
            total_valor = medianamente_ganadores_data['precio_vendedor_total'].sum()
            st.metric("Valor Total de Productos Medianamente Ganadores", f"${total_valor:,.2f}")

            medianamente_ganadores_table = medianamente_ganadores_data[[
                'super_catalog_id', 
                'unidades_pedidas', 
                'vendor_id',
                'precio_vendedor_total'
            ]].copy()
        
            medianamente_ganadores_table.columns = [
                'ID Producto', 
                'Unidades Pedidas', 
                'Vendor ID',
                'Valor Total'
            ]
        
            st.dataframe(
                medianamente_ganadores_table.style.format({
                    'Unidades Pedidas': '{:.0f}',
                    'Valor Total': '${:.2f}'
                })
            )

    with tab3:
        st.session_state.current_tab = "Análisis de Órdenes vs Mínimos"
        
        # Checkboxes en el sidebar
        with category_container:
            st.markdown("---")
            st.subheader("Categorías de Productos")
            show_winners = st.checkbox("Productos Ganadores", value=True, key='cb_winners_tab3')
            show_semi_winners = st.checkbox("Productos Semi Ganadores", value=False, key='cb_semi_winners_tab3')
            show_medium_winners = st.checkbox("Productos Medianamente Ganadores", value=False, key='cb_medium_winners_tab3')

        if not any([show_winners, show_semi_winners, show_medium_winners]):
            st.warning("Por favor seleccione al menos una categoría de productos para visualizar.")
            st.stop()

        st.subheader("Análisis de Compra Potencial al Mejor Precio")
    
        # Preparar datos de órdenes para el POS seleccionado
        orders_data = df_original[df_original['point_of_sale_id'] == selected_pos].copy()
        orders_data['order_date'] = pd.to_datetime(orders_data['order_date'])
        orders_data['date'] = orders_data['order_date'].dt.date
    
        # Obtener productos de todas las categorías para el POS seleccionado
        winning_products = df_products[df_products['point_of_sale_id'] == selected_pos].copy()
        semi_winning_products = df_semi_ganadores[
            (df_semi_ganadores['point_of_sale_id'] == selected_pos) & 
            (df_semi_ganadores['categoria_diferencia'] == 'hasta_5_porciento')
        ].copy()
        medium_winning_products = df_semi_ganadores[
            (df_semi_ganadores['point_of_sale_id'] == selected_pos) & 
            (df_semi_ganadores['categoria_diferencia'] == 'hasta_15_porciento')
        ].copy()
    
        # Crear mappings para cada categoría
        product_vendor_mapping = winning_products.groupby('super_catalog_id').agg({
            'vendor_id': 'first',
            'precio_minimo': 'first',
            'valor_total_vendedor': 'first'
        }).to_dict('index')

        semi_product_mapping = semi_winning_products.groupby('super_catalog_id').agg({
            'vendor_id': 'first',
            'precio_vendedor': 'first',
            'precio_vendedor_total': 'first'
        }).to_dict('index')

        medium_product_mapping = medium_winning_products.groupby('super_catalog_id').agg({
            'vendor_id': 'first',
            'precio_vendedor': 'first',
            'precio_vendedor_total': 'first'
        }).to_dict('index')
    
        # Preparar datos por fecha y orden
        orders_summary = orders_data.groupby(['order_id', 'date']).agg({
            'geo_zone': 'first',
            'super_catalog_id': list,
            'unidades_pedidas': list
        }).reset_index()
    
        # Filtros de fecha y orden
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.selectbox(
                "Seleccionar Fecha",
                options=sorted(orders_data['date'].unique()),
                format_func=lambda x: x.strftime('%Y-%m-%d'),
                key="date_selector_tab3"
            )
    
        filtered_orders = orders_summary[orders_summary['date'] == selected_date]
    
        with col2:
            selected_order = st.selectbox(
                "Seleccionar Orden",
                options=sorted(filtered_orders['order_id'].unique()),
                format_func=lambda x: f"Orden #{x}",
                key="order_selector_tab3"
            )
    
        order_detail = filtered_orders[filtered_orders['order_id'] == selected_order].iloc[0]
        geozone = order_detail['geo_zone']

        # Inicializar diccionarios para tracking
        vendor_totals = {}
        vendor_category_totals = {}

        # Procesar productos según selecciones
        for product_id, units in zip(order_detail['super_catalog_id'], order_detail['unidades_pedidas']):
            processed = False
            vendor_id = None
            total_value = 0
            category = None

            if show_winners and product_id in product_vendor_mapping:
                vendor_info = product_vendor_mapping[product_id]
                vendor_id = vendor_info['vendor_id']
                total_value = vendor_info['valor_total_vendedor']
                category = "Ganador"
                processed = True
            elif show_semi_winners and product_id in semi_product_mapping:
                vendor_info = semi_product_mapping[product_id]
                vendor_id = vendor_info['vendor_id']
                total_value = vendor_info['precio_vendedor_total']
                category = "Semi Ganador"
                processed = True
            elif show_medium_winners and product_id in medium_product_mapping:
                vendor_info = medium_product_mapping[product_id]
                vendor_id = vendor_info['vendor_id']
                total_value = vendor_info['precio_vendedor_total']
                category = "Medianamente Ganador"
                processed = True

            if processed:
                if vendor_id not in vendor_totals:
                    vendor_totals[vendor_id] = 0
                    vendor_category_totals[vendor_id] = {
                        'Ganador': 0,
                        'Semi Ganador': 0,
                        'Medianamente Ganador': 0
                    }
                vendor_totals[vendor_id] += total_value
                vendor_category_totals[vendor_id][category] += total_value

        # Preparar datos para visualización
        vendor_data = []
        for vendor_id, total_value in vendor_totals.items():
            min_purchase_info = df_min_purchase[
                (df_min_purchase['vendor_id'] == vendor_id) & 
                (df_min_purchase['name'] == geozone)
            ]
            min_purchase_value = min_purchase_info['min_purchase'].iloc[0] if not min_purchase_info.empty else 0
            vendor_status = get_vendor_status(vendor_id, selected_pos, winning_products)

            vendor_data.append({
                'Vendor ID': f'Vendor {vendor_id}',
                'Compra Potencial': total_value,
                'Compra Mínima': min_purchase_value,
                'Status': vendor_status[0],
                'Total Ganadores': vendor_category_totals[vendor_id]['Ganador'],
                'Total Semi Ganadores': vendor_category_totals[vendor_id]['Semi Ganador'],
                'Total Medianamente Ganadores': vendor_category_totals[vendor_id]['Medianamente Ganador']
            })

        if vendor_data:
            vendor_df = pd.DataFrame(vendor_data)
            vendor_df = vendor_df.sort_values('Compra Potencial', ascending=True)


            st.subheader("Ajuste de Compras Mínimas")
            input_cols = st.columns(3)
            col_idx = 0
            adjusted_min_purchases = {}

            for idx, row in vendor_df.iterrows():
                with input_cols[col_idx]:
                    default_value = float(row['Compra Mínima']) if pd.notnull(row['Compra Mínima']) else 0.0
                    adjusted_value = st.number_input(
                        f"{row['Vendor ID']}",
                        min_value=0.0,
                        value=default_value,
                        format="%.2f",
                        key=f"input_{row['Vendor ID']}"
                    )
                    adjusted_min_purchases[row['Vendor ID']] = adjusted_value
                col_idx = (col_idx + 1) % 3

            # Crear gráfico de barras apiladas
            fig = go.Figure()
            
            if show_winners:
                fig.add_trace(go.Bar(
                    name='Ganadores',
                    y=vendor_df['Vendor ID'],
                    x=vendor_df['Total Ganadores'],
                    orientation='h',
                    marker_color='rgb(55, 83, 109)',
                    hovertemplate='<b>%{y}</b><br>Ganadores: $%{x:,.2f}<extra></extra>'
                ))

            if show_semi_winners:
                fig.add_trace(go.Bar(
                    name='Semi Ganadores',
                    y=vendor_df['Vendor ID'],
                    x=vendor_df['Total Semi Ganadores'],
                    orientation='h',
                    marker_color='rgb(26, 118, 255)',
                    hovertemplate='<b>%{y}</b><br>Semi Ganadores: $%{x:,.2f}<extra></extra>'
                ))

            if show_medium_winners:
                fig.add_trace(go.Bar(
                    name='Medianamente Ganadores',
                    y=vendor_df['Vendor ID'],
                    x=vendor_df['Total Medianamente Ganadores'],
                    orientation='h',
                    marker_color='rgb(95, 158, 245)',
                    hovertemplate='<b>%{y}</b><br>Medianamente Ganadores: $%{x:,.2f}<extra></extra>'
                ))

            # Configurar layout
            fig.update_layout(
                barmode='stack',
                title=f'Compra Potencial vs Compra Mínima por Vendor - Orden #{selected_order}',
                xaxis_title='Monto ($)',
                yaxis_title='Vendor ID',
                height=max(400, len(vendor_df) * 80),
                showlegend=True,
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1)
            )

            # Añadir líneas de compra mínima ajustada
            for idx, row in vendor_df.iterrows():
                adjusted_min = adjusted_min_purchases[row['Vendor ID']]
                y_pos = vendor_df.index.get_loc(idx)  # Obtener la posición correcta en el eje Y
                
                fig.add_shape(
                    type='line',
                    x0=adjusted_min,
                    x1=adjusted_min,
                    y0=y_pos-0.3,
                    y1=y_pos+0.3,
                    line=dict(color='red', width=2, dash='dash')
                )
                fig.add_annotation(
                    x=adjusted_min,
                    y=y_pos,
                    text=f'${adjusted_min:,.0f}',
                    showarrow=False,
                    yshift=20
                )

            # Mostrar métricas actualizadas con valores ajustados
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Compra Potencial de la orden", f"${vendor_df['Compra Potencial'].sum():,.2f}")
            with col2:
                #st.metric("Total Compra Real", f"{intersection['precio_total_droguería'].sum():,.2f}")
                st.metric("Total Compra Real de la orden", f"${intersection[(intersection['point_of_sale_id'] == selected_pos) & (intersection['order_id'] == selected_order)]['precio_total_droguería'].sum():,.2f}")

            with col3:
                vendors_cumplen = sum(
                    vendor_df['Compra Potencial'] >= vendor_df['Vendor ID'].map(adjusted_min_purchases)
                )
                st.metric("Vendors que Cumplen Mínimo Ajustado", f"{vendors_cumplen}/{len(vendor_df)}")
            #with col3:
             #   st.metric("Total Vendors", str(len(vendor_df)))

            # Mostrar gráfico
            st.plotly_chart(fig, use_container_width=True)

            # Mostrar tabla detallada con valores ajustados
            st.subheader("Detalle por Vendor")
            display_df = vendor_df.copy()
            display_df['Compra Mínima Ajustada'] = display_df['Vendor ID'].map(adjusted_min_purchases)
            
            # Corregir la verificación de cumplimiento
            display_df['Cumple Mínimo'] = display_df.apply(
                lambda x: "Sí" if x['Compra Potencial'] >= x['Compra Mínima Ajustada'] else "No", 
                axis=1
            )

            # Función para aplicar colores según el status
            def get_status_color(status):
                if status == "Activo":
                    return 'background-color: #90EE90'  # Verde
                elif status == "Pendiente":
                    return 'background-color: #FFD700'  # Amarillo
                else:  # Sin Status
                    return 'background-color: #ffcccb'  # Rosado

            st.dataframe(
                display_df.style.format({
                    'Compra Potencial': '${:,.2f}',
                    'Compra Mínima': '${:,.2f}',
                    'Compra Mínima Ajustada': '${:,.2f}',
                    'Total Ganadores': '${:,.2f}',
                    'Total Semi Ganadores': '${:,.2f}',
                    'Total Medianamente Ganadores': '${:,.2f}'
                }).applymap(
                    lambda x: 'background-color: #90EE90' if x == 'Sí' else 'background-color: #ffcccb',
                    subset=['Cumple Mínimo']
                ).applymap(
                    get_status_color,
                    subset=['Status']
                )
            )
        else:
            st.warning("No se encontraron productos para esta orden en las categorías seleccionadas.")
except Exception as e:
    st.error(f"Error al procesar los datos: {str(e)}")
    st.write("Asegúrate de que ambos archivos CSV estén en el directorio correcto y tengan el formato esperado.")
    