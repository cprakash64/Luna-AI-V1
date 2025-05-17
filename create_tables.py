import os
import sys
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Boolean, ForeignKey, Float, Integer, JSON, create_engine, inspect, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Create SQLAlchemy engine directly
db_url = "postgresql://luna:746437@localhost:5432/luna"
engine = create_engine(db_url)

# Create base class for models
Base = declarative_base()

# Define models directly in this script
class BaseModel(Base):
    """Base model with common fields for all models"""
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at = Column(String, default=lambda: datetime.utcnow().isoformat())

class User(BaseModel):
    """User model"""
    __tablename__ = "users"
    
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    bio = Column(Text)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # Relationships - commenting out to avoid errors for now
    # videos = relationship("Video", back_populates="user")

class Video(BaseModel):
    """Video model to store video metadata and analysis results"""
    __tablename__ = "videos"
    
    # Basic metadata
    title = Column(String(255), index=True)
    description = Column(Text)
    filename = Column(String(255))
    storage_path = Column(String(255))
    external_url = Column(String(512))  # For YouTube videos
    
    # Technical metadata
    duration = Column(Float)   # In seconds
    width = Column(Integer)
    height = Column(Integer)
    format = Column(String(50))
    file_size = Column(Integer)  # In bytes
    
    # Thumbnails - using JSONB instead of ARRAY(String) for compatibility
    thumbnail_urls = Column(JSONB)
    
    # Processing status
    status = Column(String(50), index=True, default="pending")  # pending, processing, completed, failed
    error = Column(Text)  # Store error messages if processing fails
    
    # Analysis results
    transcription = Column(Text)
    transcription_segments = Column(JSONB)  # Timestamped segments
    detected_objects = Column(JSONB)  # Objects with timestamps and bounding boxes
    detected_scenes = Column(JSONB)  # Scene changes with timestamps
    extracted_text = Column(JSONB)  # OCR results with timestamps
    sentiment_analysis = Column(JSONB)  # Sentiment scores with timestamps
    
    # Summary and insights
    summary = Column(Text)
    topics = Column(JSONB)  # Extracted topics with scores
    key_moments = Column(JSONB)  # Important timestamps with descriptions
    
    # Relations
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    # user = relationship("User", back_populates="videos")

def create_tables():
    print("Creating database tables...")
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Verify tables were created
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        print(f"✅ Successfully created {len(tables)} tables:")
        for table in tables:
            print(f"  - {table}")
            
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def add_username_column():
    print("Checking if username column exists in users table...")
    
    try:
        # Check if column exists
        inspector = inspect(engine)
        columns = [col['name'] for col in inspector.get_columns('users')]
        
        if 'username' in columns:
            print("✅ Username column already exists.")
            return True
        
        print("Username column doesn't exist. Adding it...")
        
        # Add username column
        with engine.connect() as connection:
            # Add column first
            connection.execute(text("ALTER TABLE users ADD COLUMN username VARCHAR(255) UNIQUE;"))
            
            # Populate username from full_name or email
            print("Populating username column...")
            # First try using full_name
            if 'full_name' in columns:
                connection.execute(text("""
                    UPDATE users 
                    SET username = COALESCE(full_name, 'user_' || substring(cast(id as varchar), 1, 8))
                    WHERE username IS NULL;
                """))
            else:
                # Use email prefix as fallback
                connection.execute(text("""
                    UPDATE users 
                    SET username = COALESCE(
                        split_part(email, '@', 1), 
                        'user_' || substring(cast(id as varchar), 1, 8)
                    )
                    WHERE username IS NULL;
                """))
            
            # Make unique by appending numbers to duplicates
            connection.execute(text("""
                CREATE OR REPLACE FUNCTION make_usernames_unique() RETURNS void AS $$
                DECLARE
                    rec RECORD;
                    counter INTEGER;
                    new_username VARCHAR;
                BEGIN
                    FOR rec IN (
                        SELECT id, username FROM users
                        WHERE username IN (
                            SELECT username FROM users GROUP BY username HAVING COUNT(*) > 1
                        )
                        ORDER BY id
                    ) LOOP
                        counter := 1;
                        new_username := rec.username || '_' || counter;
                        
                        -- Keep incrementing counter until we find a unique username
                        WHILE EXISTS (SELECT 1 FROM users WHERE username = new_username) LOOP
                            counter := counter + 1;
                            new_username := rec.username || '_' || counter;
                        END LOOP;
                        
                        -- Update the username
                        UPDATE users SET username = new_username WHERE id = rec.id;
                    END LOOP;
                END;
                $$ LANGUAGE plpgsql;
                
                SELECT make_usernames_unique();
                DROP FUNCTION make_usernames_unique();
            """))
            
            # Add NOT NULL constraint
            connection.execute(text("ALTER TABLE users ALTER COLUMN username SET NOT NULL;"))
            
            connection.commit()
            
        print("✅ Successfully added and populated username column.")
        return True
    except Exception as e:
        print(f"❌ Error adding username column: {e}")
        return False

if __name__ == "__main__":
    # First make sure tables exist
    create_tables()
    
    # Then add username column if needed
    add_username_column()