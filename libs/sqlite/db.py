import sqlite3

# Connect to a database (creates if it doesn't exist)
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Example usage
cursor.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)')
conn.commit()
conn.close()
