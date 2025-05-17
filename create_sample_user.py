# create_sample_user.py
import uuid
import hashlib
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Database connection string
db_url = "postgresql://luna:746437@localhost:5432/luna"

def create_sample_user():
    print(f"Connecting to database: {db_url}")
    
    try:
        # Create engine and session
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Generate a simple password hash (in a real app, use a proper password hashing library)
        password = "password123"
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Insert a sample user
        user_id = str(uuid.uuid4())
        current_time = "2025-03-12T00:00:00.000000"
        
        query = text("""
        INSERT INTO users (
            id, created_at, updated_at, email, username, hashed_password, 
            full_name, is_active, is_superuser
        ) VALUES (
            :id, :created_at, :updated_at, :email, :username, :hashed_password,
            :full_name, :is_active, :is_superuser
        )
        """)
        
        session.execute(query, {
            "id": user_id,
            "created_at": current_time,
            "updated_at": current_time,
            "email": "admin@luna.ai",
            "username": "admin",
            "hashed_password": hashed_password,
            "full_name": "Admin User",
            "is_active": True,
            "is_superuser": True
        })
        
        session.commit()
        print(f"✅ Sample user created with ID: {user_id}")
        print(f"Email: admin@luna.ai")
        print(f"Username: admin")
        print(f"Password: {password}")
        
        # Verify user was created
        user = session.execute(text("SELECT * FROM users WHERE email = 'admin@luna.ai'")).fetchone()
        print(f"\nUser record: {user}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        session.rollback()
        return False
    
if __name__ == "__main__":
    create_sample_user()