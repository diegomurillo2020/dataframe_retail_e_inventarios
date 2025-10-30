import pandas as pd
# Las librerías ReportLab se usan para generar informes PDF de alta calidad
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import green, orange, red
import os
import sys
import datetime

# --- CONFIGURACIÓN DE ARCHIVOS ---
# Nombre del archivo CSV de entrada generado por el script de forecast
CSV_FILE = "forecast_resultados_mejorado.csv"
# Nombre del archivo PDF de salida
PDF_FILE = "Informe_Forecast_Inventario.pdf"


# --- 1. FUNCIÓN DE CARGA Y VERIFICACIÓN DE DATOS ---
def cargar_datos_y_verificar(file_path: str) -> pd.DataFrame | None:
    """
    Carga los datos del archivo CSV, renombra columnas, realiza limpieza, 
    parsea la cadena del forecast para obtener el total y las semanas, 
    y verifica la integridad.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: Archivo no encontrado en la ruta: {file_path}")
        print("Asegúrese de que 'forecast_resultados_mejorado.csv' esté en la misma carpeta.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # 1. Renombrar columnas para consistencia con el proyecto
        df = df.rename(columns={
            'SKU': 'StockCode',
            'Store': 'StoreID',
            'Runtime_sec': 'Duracion_sec',
        }, errors='ignore') # Usamos ignore por si las columnas ya tienen el nombre correcto
        
        # 2. Conversión de tipos y manejo de errores (coerce)
        for col in ['MAPE', 'Safety_Stock', 'Reorder_Point', 'Qty_to_Order', 'Duracion_sec']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. Procesar la columna 'Forecast' (que es una cadena de lista)
        df['Forecast'] = df['Forecast'].fillna('[]').astype(str)
        
        def parse_forecast(forecast_str):
            """Intenta convertir la cadena de lista en una lista de floats, calcula total y semanas."""
            try:
                # Método robusto para limpiar y parsear la cadena
                cleaned = forecast_str.strip().replace('[', '').replace(']', '')
                if not cleaned:
                    return [], 0.0, 0
                
                # Esto maneja números separados por coma o espacio
                forecast_list = [float(x.strip()) for x in cleaned.split(',') if x.strip()]
                
                total = sum(forecast_list)
                weeks = len(forecast_list)
                return forecast_list, total, weeks
            except Exception:
                # Si falla, devuelve valores seguros
                return [], 0.0, 0

        # Aplicar el parseo y generar las nuevas columnas
        results = df['Forecast'].apply(parse_forecast).apply(pd.Series)
        results.columns = ['Forecast_List', 'Forecast_Total', 'Forecast_Weeks']
        df = pd.concat([df, results], axis=1)

        # 4. Calcular la columna derivada 'Need_Reorder'
        df['Need_Reorder'] = df['Qty_to_Order'] > 0
        
        # 5. Añadir RMSE si falta (usando 0 como fallback para que el reporte no falle)
        if 'RMSE' not in df.columns:
            print("⚠️ Advertencia: Columna 'RMSE' no encontrada. Usando valor predeterminado (0) en el informe.")
            df['RMSE'] = 0.0
            
        # 6. Limpieza final
        df = df.dropna(subset=['Forecast_Total', 'MAPE', 'Qty_to_Order'])
        
        # Convertir a enteros las cantidades de inventario (no tiene sentido tener 7.4 unidades)
        for col in ['Safety_Stock', 'Reorder_Point', 'Qty_to_Order', 'Forecast_Total']:
            df[col] = df[col].round(0).astype(int)

        if df.empty:
            print("❌ Error: El DataFrame está vacío después de la carga, limpieza o cálculo de totales.")
            return None

        return df
    except Exception as e:
        print(f"❌ Error al leer o procesar el archivo CSV: {e}")
        return None


# --- 2. FUNCIÓN DE INTERPRETACIÓN DE RESULTADOS (SOLO GENERA EL TEXTO XML/HTML) ---
def generar_interpretacion_sku(row: pd.Series) -> str:
    """
    Genera una interpretación de texto legible para un SKU/Tienda.
    Utiliza etiquetas <b> y <br/> y <font> para formato dentro del <para>.
    """
    
    # Aseguramos el uso de etiquetas <b> en el texto de las variables
    pedido_qty = f"<b>{row['Qty_to_Order']} unidades</b>"
    rop_val = f"<b>{row['Reorder_Point']} unidades</b>"
    ss_val = f"<b>{row['Safety_Stock']} unidades</b>"
    forecast_val = f"<b>{row['Forecast_Total']} unidades</b>"

    # 1. Interpretación del Pedido
    if row['Need_Reorder']: # Usa la columna calculada Need_Reorder
        pedido = (
            f"RECOMENDACIÓN CLAVE: Se debe realizar un pedido de {pedido_qty} "
            f"para evitar una posible rotura de stock."
        )
    else:
        pedido = "RECOMENDACIÓN CLAVE: No se requiere realizar un pedido inmediato (el inventario actual está por encima del ROP)."

    # 2. Interpretación de Logística/Inventario
    logistica = (
        f"El punto de reorden (ROP) es de {rop_val}.<br/>"
        f"El stock de seguridad (SS) calculado es de {ss_val}.<br/>"
        f"La demanda pronosticada para las {row['Forecast_Weeks']} semanas siguientes es de {forecast_val}."
    )

    # 3. Interpretación de la Precisión del Modelo (MAPE)
    mape_val = row['MAPE']
    
    if mape_val < 10.0:
        prec_text = f"El modelo muestra una <b>alta precisión</b> (MAPE: {mape_val:.2f}%). Las predicciones son muy fiables."
        prec_color = "green"
    elif mape_val < 50.0:
        prec_text = f"La precisión del modelo es <b>aceptable</b> (MAPE: {mape_val:.2f}%). Requiere monitorización."
        prec_color = "orange"
    else:
        prec_text = f"El modelo tuvo una <b>baja precisión</b> (MAPE: {mape_val:.2f}%), lo que indica demanda intermitente o volátil. La decisión se basa fuertemente en el Stock de Seguridad."
        prec_color = "red"
        
    
    # Ensamblar el informe en formato XML de ReportLab
    informe = f"""
    <para>
        <font size="14"><b>Análisis Detallado de SKU: {row['StockCode']} en {row['StoreID']}</b></font><br/>
        <font size="10">Duración del procesamiento: {row['Duracion_sec']:.2f} segundos</font>
    </para>
    <para>
        <font size="12"><b>1. Decisión de Pedido:</b></font><br/>
        {pedido}
    </para>
    <para>
        <font size="12"><b>2. Parámetros Logísticos:</b></font><br/>
        {logistica}
    </para>
    <para>
        <font size="12"><b>3. Calidad del Pronóstico:</b></font><br/>
        <font color="{prec_color}"> {prec_text} (RMSE: {row['RMSE']:.2f}).</font>
    </para>
    """
    return informe.strip()


# --- 3. GENERACIÓN DEL INFORME PDF ---
def generar_pdf_informe(df: pd.DataFrame, nombre_archivo: str):
    """Crea un documento PDF con la interpretación de los resultados."""
    
    doc = SimpleDocTemplate(nombre_archivo, pagesize=letter)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Estilos personalizados (mejorados para incluir color)
    styles.add(ParagraphStyle(name='TituloPrincipal', fontSize=18, spaceAfter=20, alignment=1))
    styles.add(ParagraphStyle(name='TextoCuerpo', fontSize=12, spaceAfter=10, leading=14))
    
    # Título Principal
    story.append(Paragraph("Informe de Optimización de Inventario", styles['TituloPrincipal']))
    story.append(Paragraph(f"Fecha de Generación: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Resumen
    story.append(Paragraph(f"Resumen: Se analizaron {len(df)} combinaciones SKU/Tienda. <b>{df['Need_Reorder'].sum()}</b> requieren un pedido de reabastecimiento.", styles['TextoCuerpo']))
    
    # Métricas Globales (Opcional, pero útil)
    story.append(Paragraph(f"MAPE Promedio Global: <b>{df['MAPE'].mean():.2f}%</b>", styles['TextoCuerpo']))
    story.append(Spacer(1, 24))


    # Generar interpretación SKU por SKU
    for index, row in df.iterrows():
        # Obtenemos el texto XML/HTML
        html_content = generar_interpretacion_sku(row)
        
        # Dividimos el contenido por </para> y procesamos cada segmento
        # Esto es crucial para ReportLab: cada <para> debe ser procesado individualmente
        segments = html_content.split('</para>')
        
        for segment in segments:
            if segment.strip():
                # ReportLab necesita que cada <para> tenga su cierre. Añadimos el cierre manualmente.
                p_text = segment.strip() + '</para>'
                
                # Añadir un separador visual antes de cada nuevo SKU
                if "Análisis Detallado" in p_text:
                    story.append(Paragraph("<hr/>", styles['Normal'])) # Separador visual
                    
                # Creamos el objeto Paragraph
                p = Paragraph(p_text, styles['TextoCuerpo'])
                story.append(p)
        
        story.append(Spacer(1, 18))
        
    try:
        doc.build(story)
        print(f"\n✅ ¡Informe PDF generado con éxito: {nombre_archivo}!")
    except Exception as e:
        print(f"\n❌ Error al construir el PDF. Asegúrese de que ReportLab está instalado correctamente y el formato es válido: {e}")


# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    
    # Verificación de librerías
    try:
        # Importamos solo para verificar la instalación
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        print("\n🛑 ERROR: La librería 'reportlab' no está instalada.")
        print("Ejecute: pip install reportlab")
        sys.exit(1)
        
    df_results = cargar_datos_y_verificar(CSV_FILE)
    if df_results is not None:
        generar_pdf_informe(df_results, PDF_FILE)