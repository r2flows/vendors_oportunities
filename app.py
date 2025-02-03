import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

# Función para crear el dashboard
def create_sales_potential_dashboard(df_actual, df_potential):
    try:
        # Primero obtener los PDVs únicos por zona
        pdv_actual = (df_actual.groupby('geo_zone')
                     ['point_of_sale_id'].unique()
                     .apply(lambda x: set(x))) if not df_actual.empty else pd.Series(dtype='object')
        
        pdv_potential = (df_potential.groupby('geo_zone')
                        ['point_of_sale_id'].unique()
                        .apply(lambda x: set(x))) if not df_potential.empty else pd.Series(dtype='object')
        
        # Crear el análisis base con left outer join
        zone_analysis = pd.merge(
            df_potential.groupby(['geo_zone', 'vendor_id']).agg({
                'precio_total_vendedor': 'sum',
                'unidades_pedidas': 'sum',
            }).reset_index(),
            df_actual.groupby(['geo_zone', 'vendor_id']).agg({
                'precio_total': 'sum',
                'unidades_pedidas': 'sum',
            }).reset_index() if not df_actual.empty else pd.DataFrame(columns=['geo_zone', 'vendor_id', 'precio_total', 'unidades_pedidas']),
            on=['geo_zone', 'vendor_id'],
            how='left',  # Cambiar a left join para mantener todos los datos de potential
            suffixes=('_potential', '_actual')
        ).fillna(0)
        
        def get_pdv_count(zone, status=None):
            if status is None:
                return len(pdv_actual[zone]) if zone in pdv_actual else 0
            key = (zone, status)
            return len(pdv_potential[key]) if key in pdv_potential else 0
        

        # Agregar los conteos correctos de PDV
        zone_analysis['pdv_actual'] = zone_analysis['geo_zone'].apply(get_pdv_count)
        zone_analysis['pdv_status_1'] = zone_analysis['geo_zone'].apply(lambda x: get_pdv_count(x, 1))
        zone_analysis['pdv_status_2'] = zone_analysis['geo_zone'].apply(lambda x: get_pdv_count(x, 2))
        zone_analysis['pdv_sin_relacion'] = zone_analysis['geo_zone'].apply(lambda x: get_pdv_count(x, 0))
        
        status_mapping = {0: 'rechazado', 1: 'aceptado', 2: 'pendiente', None: 'sin_relacion'}        
        
        
        for desc in status_mapping.values():
            zone_analysis[f'potential_status_{desc}'] = 0
        

        for geo_zone in zone_analysis['geo_zone'].unique():
            for status, desc in status_mapping.items():
                if status is None:
                    # For points without relationship - those not in vendor_pos_relations
                    mask = (df_potential['geo_zone'] == geo_zone) & (df_potential['status'].isna())
                else:
                    mask = (df_potential['geo_zone'] == geo_zone) & (df_potential['status'] == status)
                
                if mask.any():
                    total_vendedor = df_potential.loc[mask, 'precio_total_vendedor'].sum()
                    zone_analysis.loc[zone_analysis['geo_zone'] == geo_zone, f'potential_status_{desc}'] = total_vendedor
        

        # Calcular métricas adicionales
        zone_analysis['total_sales'] = zone_analysis['precio_total'] + zone_analysis['precio_total_vendedor']
        zone_analysis['actual_percentage'] = (zone_analysis['precio_total'] / zone_analysis['total_sales'] * 100).round(2)
        zone_analysis['potential_percentage'] = (zone_analysis['precio_total_vendedor'] / zone_analysis['total_sales'] * 100).round(2)
        zone_analysis['growth_percentage'] = ((zone_analysis['precio_total_vendedor'] / zone_analysis['precio_total']) * 100).round(2)
        
        # Calcular porcentajes por status de forma segura
        for status in ['rechazado', 'aceptado', 'pendiente','sin_relacion']:
            col_name = f'potential_status_{status}'
            pct_name = f'percentage_status_{status}'
            zone_analysis[pct_name] = np.where(
                zone_analysis['precio_total_vendedor'] > 0,
                (zone_analysis[col_name] / zone_analysis['precio_total_vendedor'] * 100).round(2),
                0
            )

        # Ordenar por potencial de venta
        zone_analysis = zone_analysis.sort_values('precio_total_vendedor', ascending=False)
        # Get top 5 zones
        top_5_zones = (zone_analysis
                      .sort_values('total_sales', ascending=False)
                      .groupby('vendor_id')
                      .head(7)
                      .sort_values(['vendor_id', 'total_sales'], ascending=[True, False]))

        # Create dashboard layout
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "bar"}]],
            vertical_spacing=0.12
        )
        
        # Calculate main metrics
        total_actual = zone_analysis['precio_total'].sum()
        total_potential = zone_analysis['precio_total_vendedor'].sum()

        # Create stacked bar chart
        # Create stacked bar chart
# Create stacked bar chart
        if not top_5_zones.empty:
            # Add actual sales bars
            fig.add_trace(
                go.Bar(
                    name='Venta Actual',
                    x=[f"{row['geo_zone']}" for _, row in top_5_zones.iterrows()],
                    y=top_5_zones['precio_total'],
                    text=[f"${v:,.0f}<br>{p:.1f}%" for v, p in zip(top_5_zones['precio_total'], top_5_zones['actual_percentage'])],
                    textposition='auto',  # Cambiado a 'auto' para posicionamiento automático
                    marker_color='rgb(55, 83, 109)',
                    textfont=dict(size=9),
                ),
                row=1, col=1
            )
            
            # Add potential sales bars
            fig.add_trace(
                go.Bar(
                    name='Potencial Adicional',
                    x=[f"{row['geo_zone']}" for _, row in top_5_zones.iterrows()],
                    y=top_5_zones['precio_total_vendedor'],
                    text=[f"${v:,.0f}<br>{p:.1f}%" for v, p in zip(top_5_zones['precio_total_vendedor'], top_5_zones['potential_percentage'])],
                    textposition='auto',  # Cambiado a 'auto' para posicionamiento automático
                    marker_color='rgb(26, 118, 255)',
                    textfont=dict(size=9),
                ),
                row=1, col=1
            )
        
        # Actualizar el layout con altura más moderada
        fig.update_layout(
            barmode='stack',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Ajustar altura y márgenes
            height=350,  # Altura más moderada
            margin=dict(t=50, b=50, l=50, r=50),
            uniformtext=dict(mode="show", minsize=8),
            yaxis=dict(
                rangemode='tozero',
                automargin=True,
                tickformat='$,.0f'
            ),
            # Ajustar el espacio entre barras
            bargap=0.2,
            bargroupgap=0.1
        )
        # Update axes
        fig.update_xaxes(title_text="Zona Geográfica", row=2, col=1, tickangle=45)
        fig.update_yaxes(title_text="Ventas (MXN)", row=2, col=1)
        
        # Add hover template
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                         "Monto: $%{y:,.2f}<br>" +
                         "<extra></extra>",
            selector=dict(type='bar')
        )
        
        return fig, zone_analysis

    except Exception as e:
        raise ValueError(f"Error al procesar los datos: {str(e)}")

# Cargar los datos con manejo de tipos

def show_pdv_details(df_actual_filtered, df_potential_filtered, zona):
    # Mostrar resumen de ventas por status con separación
    st.markdown("#### 📊 Desglose de Venta Potencial")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "_PdV Aceptados_", 
            f"${zona['potential_status_aceptado']:,.2f}",
            f"{zona['percentage_status_aceptado']}% del potencial"
        )
    
    with col2:
        st.metric(
            "_PdV Pendientes_", 
            f"${zona['potential_status_pendiente']:,.2f}",
            f"{zona['percentage_status_pendiente']}% del potencial"
        )
    
    with col3:
        st.metric(
            "_PdV Rechazados_", 
            f"${zona['potential_status_rechazado']:,.2f}",
            f"{zona['percentage_status_rechazado']}% del potencial"
        )
    
    with col4:
        st.metric(
            "_PdV Sin Relación_", 
            f"${zona['potential_status_sin_relacion']:,.2f}",
            f"{zona['percentage_status_sin_relacion']}% del potencial"
        )
    
    st.markdown("---")
    st.markdown("#### 📍 Detalle de Puntos de Venta")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.write("_PdV Aceptados Venta Actual_")
        pdv_actuales = df_actual_filtered[
            df_actual_filtered['geo_zone'] == zona['geo_zone']
        ]['point_of_sale_id'].unique()
        for pdv in pdv_actuales:
            st.markdown(f"- `{pdv}`")

    with col2:
        st.markdown("_PdV Aceptados Venta Potencial_")
        pdv_status_1 = df_potential_filtered[
            (df_potential_filtered['geo_zone'] == zona['geo_zone']) &
            (df_potential_filtered['status'] == 1)
        ]['point_of_sale_id'].unique()
        for pdv in pdv_status_1:
            st.markdown(f"- `{pdv}`")

    with col3:
        st.markdown("_PdV Pendientes Venta Potencial_")
        pdv_status_2 = df_potential_filtered[
            (df_potential_filtered['geo_zone'] == zona['geo_zone']) &
            (df_potential_filtered['status'] == 2)
        ]['point_of_sale_id'].unique()
        for pdv in pdv_status_2:
            st.markdown(f"- `{pdv}`")

    with col4:
        st.markdown("_PdV Rechazados_")
        pdv_rechazados = df_potential_filtered[
            (df_potential_filtered['geo_zone'] == zona['geo_zone']) &
            (df_potential_filtered['status'] == 0)
        ]['point_of_sale_id'].unique()
        for pdv in pdv_rechazados:
            st.markdown(f"- `{pdv}`")

    with col5:
        st.markdown("_PdV Sin Relación Comercial_")
        pdv_sin_relacion = df_potential_filtered[
            (df_potential_filtered['geo_zone'] == zona['geo_zone']) &
            (df_potential_filtered['status'].isna())
        ]['point_of_sale_id'].unique()
        for pdv in pdv_sin_relacion:
            st.markdown(f"- `{pdv}`")


def create_vendor_summary(df_actual, df_potential):
    # Crear análisis para todos los vendors
    summary = pd.merge(
        df_potential.groupby('vendor_id').agg({
            'precio_total_vendedor': 'sum',
            'unidades_pedidas': 'sum',
        }).reset_index(),
        df_actual.groupby('vendor_id').agg({
            'precio_total': 'sum',
            'unidades_pedidas': 'sum',
        }).reset_index() if not df_actual.empty else pd.DataFrame(columns=['vendor_id', 'precio_total', 'unidades_pedidas']),
        on=['vendor_id'],
        how='left',
        suffixes=('_potential', '_actual')
    ).fillna(0)
    
    status_mapping = {0: 'rechazado', 1: 'aceptado', 2: 'pendiente', None: 'sin_relacion'}

    # Añadir desglose por status
    for status, desc in status_mapping.items():
        if status is None:
            summary[f'potential_status_{desc}'] = summary['vendor_id'].apply(
                lambda x: df_potential[
                    (df_potential['vendor_id'] == x) & 
                    (df_potential['status'].isna())
                ]['precio_total_vendedor'].sum()
            )
        else:
            summary[f'potential_status_{desc}'] = summary['vendor_id'].apply(
                lambda x: df_potential[
                    (df_potential['vendor_id'] == x) & 
                    (df_potential['status'] == status)
                ]['precio_total_vendedor'].sum()
            )
    # Calcular métricas adicionales
    summary['total_sales'] = summary['precio_total'] + summary['precio_total_vendedor']
    summary['growth_percentage'] = ((summary['precio_total_vendedor'] / summary['precio_total']) * 100).round(2)
    summary['growth_percentage'] = summary['growth_percentage'].replace([np.inf, -np.inf], 100)
    
    return summary

@st.cache_data
def load_data():
    try:
        # Cargar todos los datasets necesarios
        orders_delivered = pd.read_csv('orders_delivered_pos_vendor_geozone.csv')
        top_5_ventas = pd.read_csv('top_5_productos_geozona.csv')
        vendor_pos_relations = pd.read_csv('vendor_pos_relations.csv')
        
        # Convertir vendor_id a string en todos los DataFrames
        orders_delivered['vendor_id'] = orders_delivered['vendor_id'].astype(str)
        top_5_ventas['vendor_id'] = top_5_ventas['vendor_id'].astype(str)
        vendor_pos_relations['vendor_id'] = vendor_pos_relations['vendor_id'].astype(str)

        # Filtrar y mapear vendor_ids
        vendor_mapping = {'10249':'1141','10250':'1142','10260':'1148','10267':'1150','10268':'1151','10269':'1152', '10270':'1153','10271':'1154','10272':'1155','10273':'1156','10274':'1157','10275':'1158','10276':'1159','10277':'1160','10278':'1161', '10279':'1162', '10280':'1163','10281':'1164', '10282':'1165','10479':'1275','10608':'1303','10738':'1350','10778':'1373'}
        orders_delivered['vendor_id'] = orders_delivered['vendor_id'].replace(vendor_mapping)
        orders_delivered = orders_delivered[orders_delivered['vendor_id'].isin(['1148','1150','1151','1152','1153','1154','1155','1156','1157','1158','1159','1160','1161','1162','1163','1164','1275','1303','1350','1373'])]
        top_5_ventas = top_5_ventas[top_5_ventas['vendor_id'].isin(['1148','1150','1151','1152','1153','1154','1155','1156','1157','1158','1159','1160','1161','1162','1163','1164','1275','1303','1350','1373'])]

        # Preparar df_actual
        df_actual = orders_delivered.copy()
        df_actual['precio_total'] = df_actual['unidades_pedidas'].astype(float) * df_actual['precio_minimo'].astype(float)
        df_actual = df_actual[['point_of_sale_id', 'vendor_id', 'geo_zone', 'unidades_pedidas', 'precio_total','super_catalog_id']]
        
        # Preparar df_potential
        df_potential = top_5_ventas.copy()
        df_potential = df_potential[['point_of_sale_id', 'vendor_id', 'geo_zone', 'unidades_pedidas', 'precio_total_vendedor','super_catalog_id']]
        
        # Convertir tipos de datos
        df_actual['unidades_pedidas'] = df_actual['unidades_pedidas'].astype(float)
        df_actual['precio_total'] = df_actual['precio_total'].astype(float)
        df_potential['unidades_pedidas'] = df_potential['unidades_pedidas'].astype(float)
        df_potential['precio_total_vendedor'] = df_potential['precio_total_vendedor'].astype(float)



        # Agregar información de status
        df_potential = pd.merge(
            df_potential,
            vendor_pos_relations[['point_of_sale_id', 'vendor_id', 'status']],
            on=['point_of_sale_id', 'vendor_id'],
            how='left'
        )
        
        # Llenar status faltantes con 0 (sin relación)
        #df_potential['status'] = df_potential['status'].fillna(0)
        
        return df_actual, df_potential
        
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None, None
    
# Título
st.header("🏪 Oportunidades de Ventas por Proveedor y Zona Geográfica")

# Cargar datos

df_actual, df_potential = load_data()

if df_actual is not None and df_potential is not None:
    # Obtener vendor_ids únicos
    vendor_summary = create_vendor_summary(df_actual, df_potential)
    
    st.subheader("📊 Resumen General de Todos los Proveedores")
    
    # Métricas totales
    total_actual = vendor_summary['precio_total'].sum()
    total_potential = vendor_summary['precio_total_vendedor'].sum()
    avg_growth = ((total_potential / total_actual) * 100).round(2) if total_actual > 0 else 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Ventas Actuales", 
            f"${total_actual:,.2f}"
        )
    with col2:
        st.metric(
            "Total Ventas Potenciales", 
            f"${total_potential:,.2f}"
        )
    with col3:
        st.metric(
            "Potencial de Crecimiento Promedio", 
            f"{avg_growth:.2f}%"
        )
    
    # Desglose del potencial total
    total_aceptado = vendor_summary['potential_status_aceptado'].sum()
    total_pendiente = vendor_summary['potential_status_pendiente'].sum()
    total_rechazado = vendor_summary['potential_status_rechazado'].sum()
    total_sin_relacion = vendor_summary['potential_status_sin_relacion'].sum()  # Agregar esta línea

    st.markdown("#### 📈 Desglose del Potencial Total")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "_PdV Aceptados_",
            f"${total_aceptado:,.2f}",
            f"{(total_aceptado/total_potential*100):.1f}% del potencial"
        )
    with col2:
        st.metric(
            "_PdV Pendientes_",
            f"${total_pendiente:,.2f}",
            f"{(total_pendiente/total_potential*100):.1f}% del potencial"
        )
    with col3:
        st.metric(
            "_PdV Rechazados_",
            f"${total_rechazado:,.2f}",
            f"{(total_rechazado/total_potential*100):.1f}% del potencial"
        )
    with col4:  # Agregar nueva columna
        st.metric(
            "_PdV Sin Relación_",
            f"${total_sin_relacion:,.2f}",
            f"{(total_sin_relacion/total_potential*100):.1f}% del potencial"
        )
    # Tabla resumen por vendor

# Reemplazar la parte de la tabla resumen por vendor con este código:
    
    # Tabla resumen por vendor
    #st.markdown("#### 📋 Detalle por Proveedor")
    
    # Crear una copia del DataFrame y resetear el índice
    #display_summary = vendor_summary.copy()
    
    #st.dataframe(
     #   display_summary.style
      #  .format({
       #     'precio_total': '${:,.2f}',
        #    'precio_total_vendedor': '${:,.2f}',
         #   'potential_status_aceptado': '${:,.2f}',
          #  'potential_status_pendiente': '${:,.2f}',
           # 'potential_status_rechazado': '${:,.2f}',
            #'total_sales': '${:,.2f}',
            #'growth_percentage': '{:,.2f}%'
       # })
   # )
    
    st.markdown("---")
    
    vendor_ids = sorted(list(set(df_actual['vendor_id'].unique().tolist()) | 
                           set(df_potential['vendor_id'].unique().tolist())))
    st.subheader("Oportunidad potencial por proveedor")
    selected_vendor = st.selectbox('Seleccionar Proveedor:', vendor_ids)
    
    # Filtrar datos
    df_actual_filtered = df_actual[df_actual['vendor_id'] == selected_vendor].copy()
    df_potential_filtered = df_potential[df_potential['vendor_id'] == selected_vendor].copy()

    # Modificar la validación
    if len(df_potential_filtered) == 0:
        st.warning(f"No hay datos de potencial de venta para el proveedor {selected_vendor}")
    else:
        try:
            fig, analysis = create_sales_potential_dashboard(df_actual_filtered, df_potential_filtered)
            
            # Mostrar métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Ventas Actuales", 
                    f"${analysis['precio_total'].sum():,.2f}"
                )
            with col2:
                st.metric(
                    "Total Ventas Potenciales", 
                    f"${analysis['precio_total_vendedor'].sum():,.2f}"
                )
            with col3:
                if analysis['precio_total'].sum() > 0:
                    crecimiento_promedio = ((analysis['precio_total_vendedor'].sum() / analysis['precio_total'].sum()) * 100).round(2)
                else:
                    crecimiento_promedio = 100  # o cualquier otro valor que tenga sentido para tu caso
                st.metric(
                    "Potencial de Crecimiento", 
                    f"{crecimiento_promedio:.2f}%"
                )
            
            st.markdown("### 📊 Desglose de Venta Potencial Total")
            
            # Calcular totales por status
            total_aceptado = analysis['potential_status_aceptado'].sum()
            total_pendiente = analysis['potential_status_pendiente'].sum()
            total_rechazado = analysis['potential_status_rechazado'].sum()
            total_sin_relacion = analysis['potential_status_sin_relacion'].sum()
            total_potencial = analysis['precio_total_vendedor'].sum()
            
            # Calcular porcentajes
            pct_aceptado = (total_aceptado / total_potencial * 100) if total_potencial > 0 else 0
            pct_pendiente = (total_pendiente / total_potencial * 100) if total_potencial > 0 else 0
            pct_rechazado = (total_rechazado / total_potencial * 100) if total_potencial > 0 else 0
            pct_sin_relacion = (total_sin_relacion / total_potencial * 100) if total_potencial > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "_PdV Aceptados_", 
                    f"${total_aceptado:,.2f}",
                    f"{pct_aceptado:.1f}% del potencial"
                )
            
            with col2:
                st.metric(
                    "_PdV Pendientes_", 
                    f"${total_pendiente:,.2f}",
                    f"{pct_pendiente:.1f}% del potencial"
                )
            
            with col3:
                st.metric(
                    "_PdV Rechazados_", 
                    f"${total_rechazado:,.2f}",
                    f"{pct_rechazado:.1f}% del potencial"
                )
                
            with col4:
                st.metric(
                     "_PdV Sin Relación Comercial_", 
                    f"${total_sin_relacion:,.2f}",  # Corregido para usar total_sin_relacion
                    f"{pct_sin_relacion:.1f}% del potencial"  # Corregido para usar pct_sin_relacion
                )
                
            st.markdown("---")

            
            st.plotly_chart(fig, use_container_width=True)

        # Nueva sección de análisis por zona
            st.subheader("📍 Análisis Detallado por Zona Geográfica")

# Análisis por zona individual
            for _, zona in analysis.iterrows():
                with st.expander(f"🎯 {zona['geo_zone']} - Potencial Total: ${zona['total_sales']:,.2f}"):
                    st.markdown("""
            <style>
            .metric-container {
                background-color: #f0f2f6;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Sección de métricas principales
                    with st.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Venta Actual", 
                    f"${zona['precio_total']:,.2f}", 
                    f"{zona['actual_percentage']:.1f}% del total")
                    #        st.metric("PDV Actuales", 
                    #f"{zona['pdv_actual']:,.0f}")
            
                        with col2:
                            st.metric("Venta Potencial", 
                    f"${zona['precio_total_vendedor']:,.2f}", 
                    f"{zona['potential_percentage']:.1f}% del total")
                            #st.metric("PDV Potenciales", 
                    #f"{zona['pdv_potential']:,.0f}")
            
                        with col3:
                            st.metric("Crecimiento Posible", 
                    f"{zona['growth_percentage']:.1f}%")
                        #st.metric("Nuevos PDV Posibles", 
                    #f"{max(0, zona['pdv_potential'] - zona['pdv_actual']):,.0f}")

        # Sección de PDVs y Productos
                    tab1, tab2 = st.tabs(["📍 Puntos de Venta", "📦 Productos con Potencial"])
        
                    with tab1:
                        show_pdv_details(df_actual_filtered, df_potential_filtered, zona)
                    with tab2:
            # Productos con mayor potencial
                        top_productos = (
                        df_potential_filtered[
                        df_potential_filtered['geo_zone'] == zona['geo_zone']
                        ]
                        .groupby('super_catalog_id')
                        .agg({
                        'precio_total_vendedor': 'sum',
                        'unidades_pedidas': 'sum'
                        })
                        .sort_values('precio_total_vendedor', ascending=False)
                        .head(50)
                        )
            
                        for idx, (prod_id, row) in enumerate(top_productos.iterrows(), 1):
                            with st.container():
                                st.markdown(f"""
                        <div class='metric-container'>
                            <b>{idx}. Producto ID: {prod_id}</b><br>
                            💰 Venta Potencial: ${row['precio_total_vendedor']:,.2f}<br>
                            📦 Unidades Potenciales: {row['unidades_pedidas']:,.0f}
                        </div>
                    """, unsafe_allow_html=True)
# Agregar visualización del DataFrame df_potential
            st.subheader("🔍 Verificación de Datos")
            with st.expander("Ver datos de Ventas Potenciales"):
                st.write("### DataFrame de Ventas Potenciales")
                st.write("Número total de registros:", len(df_potential_filtered))
                st.dataframe(
                    df_potential_filtered.style.format({
                        'precio_total_vendedor': '${:,.2f}',
                        'unidades_pedidas': '{:,.0f}'
                    })
                )

# Mostrar el dataframe
                    #st.dataframe(
        #analysis.style.format({
        #'precio_total': '${:,.2f}',
        #'precio_total_vendedor': '${:,.2f}',
        #'growth_percentage': '{:,.2f}%'
 #   })
#)
        except Exception as e:
            st.error(f"Error al crear el dashboard: {str(e)}")


