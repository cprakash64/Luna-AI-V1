# verify_db.py
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

# Database connection string
db_url = "postgresql://luna:746437@localhost:5432/luna"

def verify_database():
    print(f"Connecting to database: {db_url}")
    
    try:
        # Create engine and session
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Test connection
        result = session.execute(text("SELECT 1")).fetchone()
        print(f"✅ Database connection successful! Result: {result[0]}")
        
        # Get PostgreSQL version
        version = session.execute(text("SELECT version()")).fetchone()
        print(f"PostgreSQL version: {version[0]}")
        
        # List all tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\nFound {len(tables)} tables in database:")
        for table in tables:
            print(f"  - {table}")
            
            # Get columns for each table
            columns = inspector.get_columns(table)
            print(f"    Columns: {', '.join(col['name'] for col in columns)}")
        
        # Check if we have any users or videos
        if "users" in tables:
            user_count = session.execute(text("SELECT COUNT(*) FROM users")).fetchone()[0]
            print(f"\nUsers table has {user_count} records")
            
        if "videos" in tables:
            video_count = session.execute(text("SELECT COUNT(*) FROM videos")).fetchone()[0]
            print(f"Videos table has {video_count} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
if __name__ == "__main__":
    verify_database()