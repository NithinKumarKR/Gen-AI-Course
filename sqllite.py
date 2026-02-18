import sqlite3

conn = sqlite3.connect("hr.db")
cursor = conn.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, course TEXT)")

cursor.execute("INSERT INTO employees (name, age, course) VALUES ('Nithin', 11, 'AI')")
cursor.execute("INSERT INTO employees (name, age, course) VALUES ('Raj', 51, 'DSA')")
cursor.execute("INSERT INTO employees (name, age, course) VALUES ('Priya', 21, 'AI')")
cursor.execute("INSERT INTO employees (name, age, course) VALUES ('Kumar', 41, 'DSA')")
cursor.execute("INSERT INTO employees (name, age, course) VALUES ('Kokila', 41, 'AI')")
cursor.execute("INSERT INTO employees (name, age, course) VALUES ('Nithin', 23, 'AI')")
cursor.execute("INSERT INTO employees (name, age, course) VALUES ('Prajwal', 33, 'DSA')")
cursor.execute("INSERT INTO employees (name, age, course) VALUES ('Dhanush', 23, 'DSA')")
cursor.execute("INSERT INTO employees (name, age, course) VALUES ('Rajesh', 15, 'DSA')")
cursor.execute("INSERT INTO employees (name, age, course) VALUES ('Nithish', 20, 'AI')")

conn.commit()
conn.close()

