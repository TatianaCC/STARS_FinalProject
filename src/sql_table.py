import sqlite3

def init_db():
    conn = sqlite3.connect('stars.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            correo TEXT NOT NULL,
            url_modelo TEXT,
            estado INTEGER CHECK (estado IN (0, 1, 2))
        )
    ''')
    conn.commit()
    conn.close()

# Llama a esta función una vez para inicializar la base de datos
init_db()
