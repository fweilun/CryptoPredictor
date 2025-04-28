from psycopg2 import pool

db_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dbname="finance",
    user="fweilun",
    password="0000",
    host="localhost",
    port="5432"
)

def get_connection():
    try:
        return db_pool.getconn()
    except Exception as e:
        print(f"Error getting connection: {e}")
        raise

def release_connection(conn):
    try:
        db_pool.putconn(conn)
    except Exception as e:
        print(f"Error releasing connection: {e}")
        raise

def close_pool():
    db_pool.closeall()
    
    
    


if __name__ == "__main__":
    conn = get_connection()
    cursor = conn.cursor()
    sql = """
    CREATE TABLE test (
        id int
    )
    """
    cursor.execute(sql)
    conn.commit()
    release_connection(conn)
    pass
