import sqlite3
import pandas as pd

# 1. Conectar a la base de datos SQLite
conn = sqlite3.connect("traces.db")

# 2. Cargar todos los spans excluyendo el span padre 'rag'
query = """
    SELECT name, start_time, end_time 
    FROM spans 
    WHERE name != 'rag'
"""
df = pd.read_sql_query(query, conn)

# 3. Calcular la duración de cada span (en milisegundos)
df['duration_ms'] = (df['end_time'] - df['start_time']) / 1_000_000

# 4. Agrupar por nombre de span y sumar los tiempos totales
df_grouped = df.groupby('name')['duration_ms'].sum().reset_index()

print("--- TIEMPO TOTAL CONSUMIDO POR TIPO DE SPAN ---")
print(df_grouped.to_string(index=False))

conn.close()
