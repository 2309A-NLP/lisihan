from app.core.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text('SELECT id, name, description FROM roles'))
    print('数据库中的角色列表:')
    for row in result:
        print(f'{row[0]}. {row[1]} - {row[2]}')