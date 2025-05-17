"""
Database test script for Luna AI
Run this script to test database connectivity and table structure
"""
import os
import sys
from sqlalchemy import create_engine, text
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("db_test")

def test_database_connection(conn_string):
    """Test database connection and table structure"""
    try:
        # Create engine
        logger.info(f"Connecting to database with: {conn_string}")
        engine = create_engine(conn_string)
        
        # Try to connect
        with engine.connect() as conn:
            logger.info("Connection successful!")
            
            # Check if users table exists
            result = conn.execute(text("SELECT to_regclass('public.users')"))
            table_exists = result.scalar()
            
            if table_exists:
                logger.info("Users table exists!")
                
                # Get table structure
                result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'users'
                ORDER BY ordinal_position
                """))
                
                logger.info("Users table structure:")
                for row in result:
                    logger.info(f"  {row[0]}: {row[1]} (nullable: {row[2]})")
                
                # Count users
                result = conn.execute(text("SELECT COUNT(*) FROM users"))
                count = result.scalar()
                logger.info(f"Users count: {count}")
                
                # List all database tables
                result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables
                WHERE table_schema = 'public'
                """))
                
                tables = [row[0] for row in result]
                logger.info(f"Database tables: {tables}")
                
            else:
                logger.warning("Users table does not exist!")
                
                # Create the users table
                logger.info("Creating users table...")
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    hashed_password VARCHAR(255) NOT NULL,
                    full_name VARCHAR(255),
                    is_active BOOLEAN DEFAULT TRUE,
                    is_superuser BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """))
                conn.commit()
                logger.info("Users table created successfully!")
                
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Get database connection from environment or use default
    db_user = os.environ.get("POSTGRES_USER", "luna")
    db_password = os.environ.get("POSTGRES_PASSWORD", "746437")
    db_host = os.environ.get("POSTGRES_HOST", "localhost")
    db_port = os.environ.get("POSTGRES_PORT", "5432")
    db_name = os.environ.get("POSTGRES_DB", "luna")
    
    conn_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    success = test_database_connection(conn_string)
    
    if success:
        logger.info("All database tests passed!")
        sys.exit(0)
    else:
        logger.error("Database tests failed!")
        sys.exit(1)