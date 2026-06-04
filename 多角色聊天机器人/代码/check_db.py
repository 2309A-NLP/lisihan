import mysql.connector

try:
    # 连接MySQL
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='chatbot'
    )

    cursor = conn.cursor()

    # 查询roles表
    print("=" * 60)
    print("MySQL Database - Roles Table")
    print("=" * 60)

    cursor.execute("SELECT id, name, description FROM roles")
    roles = cursor.fetchall()

    if roles:
        print(f"\nFound {len(roles)} roles:\n")
        for role in roles:
            print(f"ID: {role[0]}")
            print(f"Name: {role[1]}")
            print(f"Description: {role[2]}")
            print("-" * 40)
    else:
        print("\nNo data in roles table")

    # 查询templates表
    print("\n" + "=" * 60)
    print("Templates Table")
    print("=" * 60)

    cursor.execute("SELECT id, name FROM templates")
    templates = cursor.fetchall()

    if templates:
        print(f"\nFound {len(templates)} templates:\n")
        for template in templates:
            print(f"ID: {template[0]}, Name: {template[1]}")

    # 查询knowledge_bases表
    print("\n" + "=" * 60)
    print("Knowledge Bases Table")
    print("=" * 60)

    cursor.execute("SELECT id, name FROM knowledge_bases")
    kbs = cursor.fetchall()

    if kbs:
        print(f"\nFound {len(kbs)} knowledge bases:\n")
        for kb in kbs:
            print(f"ID: {kb[0]}, Name: {kb[1]}")

    cursor.close()
    conn.close()
    print("\n[SUCCESS] Database check completed!")

except mysql.connector.Error as e:
    print(f"MySQL Error: {e}")
except Exception as e:
    print(f"Error: {e}")