import sqlite3
import pandas as pd

# 1. Conectar a la base de datos ya creada
conn = sqlite3.connect("traces.db")

# 2. Traer el conteo de cuántas veces aparece cada nombre de span
query = "SELECT name, COUNT(*) as cantidad FROM spans GROUP BY name;"
df = pd.read_sql_query(query, conn)

print("--- NOMBRES DE SPANS ENCONTRADOS EN LA TABLA ---")
print(df.to_string(index=False))

conn.close()
