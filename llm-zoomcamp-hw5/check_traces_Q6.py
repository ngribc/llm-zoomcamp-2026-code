import sqlite3
import pandas as pd

# 1. Conectar a la base de datos de OpenTelemetry
conn = sqlite3.connect("traces.db")

# 2. Leer la tabla de spans con Pandas
df = pd.read_sql_query("SELECT name, input_tokens, output_tokens FROM spans", conn)

# 3. Filtrar solo los registros del LLM (que es donde guardamos los tokens)
df_llm = df[df['name'] == 'llm']

print("--- REGISTROS DE LAS EJECUCIONES EN LA BASE DE DATOS ---")
print(df_llm)

# 4. Verificar si todos los valores son iguales
if df_llm['input_tokens'].nunique() == 1:
    print("\n✅ ¡COMPROBADO!: Todos los input_tokens son perfectamente IDÉNTICOS.")
else:
    print("\n❌ Hay variaciones en los tokens.")

conn.close()
