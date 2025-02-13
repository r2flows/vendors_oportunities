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
    df_semi_ganadores = pd.read_csv('productos_clasificados_test.csv')

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
        orders_pos = df_original[df_original['point_of_sale_id'] == selected_pos]
        products_pos = df_products[df_products['point_of_sale_id'] == selected_pos]
        
        orders_products = set(orders_pos['super_catalog_id'])
        top_products = set(products_pos['super_catalog_id'])
        intersection = orders_products.intersection(top_products)
        
        intersection_percentage = (len(intersection) / len(orders_products) * 100) if orders_products else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Productos en Compras Reales", f"{len(orders_products):,}")
        with col2:
            st.metric("Total Productos Ganadores", f"{len(top_products):,}")
        with col3:
            st.metric("Productos en Intersección", 
                     f"{len(intersection):,} ({intersection_percentage:.2f}%)")

        # Mostrar productos en la intersección
        if intersection:
            st.subheader("Productos en la Intersección")
            intersection_products = df_products[
                (df_products['point_of_sale_id'] == selected_pos) & 
                (df_products['super_catalog_id'].isin(intersection))
            ]
            
            # Calcular valores totales
            orders_total = orders_pos[orders_pos['super_catalog_id'].isin(intersection)]['unidades_pedidas'].mul(
                orders_pos[orders_pos['super_catalog_id'].isin(intersection)]['precio_minimo']
            ).sum()
            
            products_total = intersection_products['valor_total_vendedor'].sum()
            
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


        # Filtrar productos para el POS seleccionado
        products_data = df_products[df_products['point_of_sale_id'] == selected_pos]
        products_data = products_data.sort_values('valor_total_vendedor', ascending=False)

        df_semi_ganadores['precio_vendedor_total'] = df_semi_ganadores['precio_vendedor']*df_semi_ganadores['unidades_pedidas']
        semi_ganadores_data = df_semi_ganadores[df_semi_ganadores['point_of_sale_id'] == selected_pos]
        semi_ganadores_data = semi_ganadores_data.sort_values('precio_vendedor_total', ascending =False)


        st.subheader("Venta Potencial Productos Ganadores")
        
                # Calcular y mostrar el valor total
        total_valor = products_data['valor_total_vendedor'].sum()
        st.metric("Valor Total de Productos Ganadores", f"${total_valor:,.2f}")

        products_table = products_data[[
            'super_catalog_id', 
            'unidades_pedidas', 
            'vendor_id',
            'valor_total_vendedor'
        ]].copy()
        
        #products_table['status_desc'] = products_table['status'].apply(get_status_description)
        
        products_table.columns = [
            'ID Producto', 
            'Unidades Pedidas', 
            'Vendor ID',
            'Valor Total'
        ]
        
        st.dataframe(
            products_table.style.format({
                'Unidades Pedidas': '{:.0f}',
                'Precio Mínimo': '${:.2f}',
                'Valor Total': '${:.2f}'
            })
        )

        st.subheader("Venta Potencial Semi Ganadores")

        total_valor = semi_ganadores_data['precio_vendedor_total'].sum()
        st.metric("Valor Total de Productos Semi Ganadores", f"${total_valor:,.2f}")


        semi_ganadores_table = semi_ganadores_data[[
            'super_catalog_id', 
            'unidades_pedidas', 
            'vendor_id',
            'precio_vendedor_total',
            'categoria_diferencia'
        ]].copy()
        
        #products_table['status_desc'] = products_table['status'].apply(get_status_description)
        
        semi_ganadores_table.columns = [
            'ID Producto', 
            'Unidades Pedidas', 
            'Vendor ID',
            'Valor Total',
            'categoria'
        ]
        
        st.dataframe(
            semi_ganadores_table[semi_ganadores_table['categoria']=='hasta_5_porciento'].style.format({
                'Unidades Pedidas': '{:.0f}',
                'Precio Mínimo': '${:.2f}',
                'Valor Total': '${:.2f}'
            })
        )

        st.subheader("Venta Potencial Medianamente Ganadores")

        total_valor = semi_ganadores_data[semi_ganadores_table['categoria']=='hasta_15_porciento']['precio_vendedor_total'].sum()
        st.metric("Valor Total de Productos Medianamente Ganadores", f"${total_valor:,.2f}")
        semi_ganadores_table = semi_ganadores_data[[
            'super_catalog_id', 
            'unidades_pedidas', 
            'vendor_id',
            'precio_vendedor_total',
            'categoria_diferencia'
        ]].copy()
        
        #products_table['status_desc'] = products_table['status'].apply(get_status_description)
        
        semi_ganadores_table.columns = [
            'ID Producto', 
            'Unidades Pedidas', 
            'Vendor ID',
            'Valor Total',
            'categoria'
        ]
        
        st.dataframe(
            semi_ganadores_table[semi_ganadores_table['categoria']=='hasta_15_porciento'].style.format({
                'Unidades Pedidas': '{:.0f}',
                'Precio Mínimo': '${:.2f}',
                'Valor Total': '${:.2f}'
            })
        )

    #with tab3:

    #with tab4:
#        st.subheader("Categorización por Status y Vendor")
        
        # Filtrar status para el POS seleccionado
 #       pos_status = pos_vendor_status[pos_vendor_status['point_of_sale_id'] == selected_pos].copy()
        
        # Obtener la geozona del POS seleccionado
  #      pos_geozone = df_original[df_original['point_of_sale_id'] == selected_pos]['geo_zone'].iloc[0]
        
        # Crear tabla de status por vendor
   #     status_table = []
    #    for _, row in pos_status.iterrows():
     #       vendor_id = row['vendor_id']
      #      status_list = row['status']
            
            # Obtener valor total de compras para este vendor
       #     vendor_total = pos_vendor_totals[
        #        (pos_vendor_totals['point_of_sale_id'] == selected_pos) & 
         #       (pos_vendor_totals['vendor_id'] == vendor_id)
          #  ]['total_compra'].iloc[0] if len(pos_vendor_totals[
           #     (pos_vendor_totals['point_of_sale_id'] == selected_pos) & 
            #    (pos_vendor_totals['vendor_id'] == vendor_id)
            #]) > 0 else 0
            
            # Obtener productos asociados
#            vendor_products = df_products[
 #               (df_products['point_of_sale_id'] == selected_pos) & 
  #              (df_products['vendor_id'] == vendor_id)
   #         ]
            
    #        min_purchase_info = df_min_purchase[
     #           (df_min_purchase['vendor_id'] == vendor_id) & 
      #          (df_min_purchase['name'] == pos_geozone)
       #     ]
        #    min_purchase_value = min_purchase_info['min_purchase'].iloc[0] if not min_purchase_info.empty else 0
            
            # Calcular si cumple con la compra mínima
         #   cumple_minimo = vendor_products['valor_total_vendedor'].sum() >= min_purchase_value if min_purchase_value > 0 else True
            


          #  status_table.append({
           #     'Vendor ID': vendor_id,
            #    'Status': [get_status_description(s) for s in status_list],
             #   'Cantidad de Productos': len(vendor_products),
#                'Valor Total Vendedor': vendor_products['valor_total_vendedor'].sum(),
 #               'Compra Mínima': min_purchase_value,
  #              'Cumplimiento del Mínimo': "Sí" if cumple_minimo else "No"
   #         })
        
    #    status_df = pd.DataFrame(status_table)
        
        # Mostrar tabla de status
     #   st.dataframe(
      #      status_df.style.format({
       #         'Valor Total Vendedor': '${:,.2f}',
        #        'Compra Mínima': '${:,.2f}',
         #       'Compra Real': '${:,.2f}'
          #  }).applymap(
           #     lambda x: 'background-color: #ffcccb' if x == "No" else '',
            #    subset=['Cumplimiento del Mínimo']
            #)
        #)
        
        # Mostrar distribución de status
        
        # Métricas de cumplimiento
#        st.subheader("Métricas de Cumplimiento de Compra Mínima")
 #       total_vendors = len(status_df)
  #      vendors_cumplen = len(status_df[status_df['Cumplimiento del Mínimo'] == "Sí"])
        
   #     col1, col2, col3 = st.columns(3)
    #    with col1:
     #       st.metric("Total Vendors", total_vendors)
      #  with col2:
       #     st.metric("Cumplimiento Mínimo", vendors_cumplen)
        #with col3:
         #   st.metric("% Cumplimiento", f"{(vendors_cumplen/total_vendors*100):.1f}%")
                
                
#        st.subheader("Distribución de Status por Vendor")
        
 #       status_distribution = {}
  #      for status_list in pos_status['status']:
   #         for status in status_list:
    #            status_desc = get_status_description(status)
     #           status_distribution[status_desc] = status_distribution.get(status_desc, 0) + 1
        
      #  status_dist_df = pd.DataFrame([
       #     {'Status': status, 'Cantidad de Vendors': count}
        #    for status, count in status_distribution.items()
        #])
        
        #st.table(status_dist_df)



#    with tab3:
  #      st.subheader("Análisis de Órdenes vs Compra Mínima (Productos Ganadores)")
    
    # Preparar datos de órdenes para el POS seleccionado
   #     orders_data = df_original[df_original['point_of_sale_id'] == selected_pos].copy()
    
    # Convertir order_date a datetime
    #    orders_data['order_date'] = pd.to_datetime(orders_data['order_date'])
     #   orders_data['date'] = orders_data['order_date'].dt.date
    
    # Obtener productos ganadores para el POS seleccionado
      #  winning_products = df_products[df_products['point_of_sale_id'] == selected_pos].copy()
    
    # Crear un mapping de producto a mejor vendor y precio
       # product_vendor_mapping = winning_products.groupby('super_catalog_id').agg({
        #    'vendor_id': 'first',
         #   'precio_minimo': 'first',
          #  'valor_total_vendedor': 'first'
    #    }).to_dict('index')
    
    # Agrupar órdenes por fecha y order_id
#        orders_summary = orders_data.groupby(['order_id', 'date']).agg({
 #           'geo_zone': 'first',
  #          'super_catalog_id': list,
   #         'unidades_pedidas': list
    #    }).reset_index()
    
    # Obtener fechas únicas para el filtro
    #    unique_dates = sorted(orders_data['date'].unique())
    
    # Filtros en columnas
    #    col1, col2 = st.columns(2)
    
    #    with col1:
    #        selected_date = st.selectbox(
    #        "Seleccionar Fecha",
    #        options=unique_dates,
    #        format_func=lambda x: x.strftime('%Y-%m-%d')
     #       )
    
    # Filtrar órdenes por fecha
#        filtered_orders = orders_summary[orders_summary['date'] == selected_date]
    
 #       with col2:
  #          selected_order = st.selectbox(
   #         "Seleccionar Orden",
    #        options=sorted(filtered_orders['order_id'].unique()),
     #       format_func=lambda x: f"Orden #{x}"
      #  )
    
    # Obtener datos de la orden seleccionada
       # order_detail = filtered_orders[filtered_orders['order_id'] == selected_order].iloc[0]
    
    # Calcular totales por vendor usando productos ganadores
        #vendor_totals = {}
        #for product_id, units in zip(order_detail['super_catalog_id'], order_detail['unidades_pedidas']):
        #    if product_id in product_vendor_mapping:
         #       vendor_info = product_vendor_mapping[product_id]
          #      vendor_id = vendor_info['vendor_id']
            
           #     if vendor_id not in vendor_totals:
            #        vendor_totals[vendor_id] = 0
            
             #   vendor_totals[vendor_id] +=  vendor_info['valor_total_vendedor']
    
    # Obtener compras mínimas para los vendors en la geozona
        #geozone = order_detail['geo_zone']

    # Crear DataFrame con la comparación
#        comparison_data = []
 #       for vendor_id, total_value in vendor_totals.items():
        # Obtener compra mínima para el vendor
  #          min_purchase_info = df_min_purchase[
   #             (df_min_purchase['vendor_id'] == vendor_id) & 
    #            (df_min_purchase['name'] == geozone)
     #       ]
      #      min_purchase_value = min_purchase_info['min_purchase'].iloc[0] if not min_purchase_info.empty else 0
            
       #     vendor_status = get_vendor_status(vendor_id, selected_pos, winning_products)
        
        # Calcular diferencia
        #    difference = total_value - min_purchase_value
         #   cumple_minimo = "Si" if difference >= 0 else "No"
        
        #    comparison_data.append({
         #       'Vendor ID': str(vendor_id),
          #      'Status': vendor_status[0],  # Tomamos el primer status si hay múltiples
           #     'Valor de Orden (Mejor Precio)': total_value,
            #    'Compra Mínima': min_purchase_value,
             #   'Diferencia': difference,
              #  'Cumple Mínimo': cumple_minimo
            #})
    
#        comparison_df = pd.DataFrame(comparison_data)
    
 #       if not comparison_df.empty:
        # Mostrar métricas generales
  #          col1, col2 = st.columns(2)
   #         with col1:
    #            st.metric(
     #               "Total de la orden (mejor precio)",
      #              f"${comparison_df['Valor de Orden (Mejor Precio)'].sum():,.2f}"
       #         )
        #    with col2:
         #       st.metric(
          #          "Vendors en los que se cumple el mínimo",
           #         f"{len(comparison_df[comparison_df['Cumple Mínimo'] == 'Si'])}/{len(comparison_df)}"
            #    )
            #with col3:
             #   st.metric(
              #      "Ahorro Potencial vs Compra Mínima",
               #     f"${comparison_df['Diferencia'].sum():,.2f}",
                #    delta_color="normal"
                #)
    #        comparison_df['Vendor ID'] = comparison_df['Vendor ID'].astype(str)

        # Crear gráfico
     #       fig = go.Figure()
        
        # Añadir barras para Valor de Orden
      #      fig.add_trace(go.Bar(
       #         name='Valor de Orden (Mejor Precio)',
        #        x=comparison_df['Vendor ID'],#.astype(str),
         #       y=comparison_df['Valor de Orden (Mejor Precio)'],
          #      marker_color='rgb(55, 83, 109)'
           # ))
        
        # Añadir barras para Compra Mínima
            #fig.add_trace(go.Bar(
           #     name='Compra Mínima',
          #      x=comparison_df['Vendor ID'].astype(str),
         #       y=comparison_df['Compra Mínima'],
        #        marker_color='rgb(26, 118, 255)'
       #     ))

        # Actualizar el layout
      #      fig.update_layout(
     #           title=f'Comparación de Valores de Orden (Mejor Precio) vs Compra Mínima - Orden #{selected_order}',
    #            xaxis_title='Vendor ID',
   #             yaxis_title='Monto ($)',
  #              barmode='group',
 #               xaxis=dict(type='category'),  # Asegurar que Vendor ID sea categórico
#                height=500
            #)

        # Mostrar el gráfico
            #st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar tabla detallada
           # st.subheader("Detalle por Vendor")
          #  detailed_table = comparison_df.copy()
        
            # Aplicar formato condicional
         #   st.dataframe(
        #        comparison_df.style.format({
       #             'Valor de Orden (Mejor Precio)': '${:,.2f}',
      #              'Compra Mínima': '${:,.2f}',
     #               'Diferencia': '${:,.2f}'
    #            }).applymap(
   #                 lambda x: 'background-color: #90EE90' if x == "Activo" else 
  #                  ('background-color: #FFD700' if x == "Pendiente" else 
 #                    'background-color: #ffcccb'),
              #      subset=['Status']
             #   ).applymap(
            #        lambda x: 'background-color: #90EE90' if x == "Sí" else 'background-color: #ffcccb',
           #         subset=['Cumple Mínimo']
          #      )
         #   )
        #else:
            #st.warning("No se encontraron productos ganadores para esta orden.")

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
                st.metric("Total Compra Potencial", f"${vendor_df['Compra Potencial'].sum():,.2f}")
            with col2:
                vendors_cumplen = sum(
                    vendor_df['Compra Potencial'] >= vendor_df['Vendor ID'].map(adjusted_min_purchases)
                )
                st.metric("Vendors que Cumplen Mínimo Ajustado", f"{vendors_cumplen}/{len(vendor_df)}")
            with col3:
                st.metric("Total Vendors", str(len(vendor_df)))

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
    