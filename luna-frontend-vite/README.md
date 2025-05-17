# Luna Backend (FastAPI)

This is the FastAPI backend for Luna AI video analysis application.

## Features

- User authentication (login, signup, password reset)
- Video processing (upload, YouTube URL processing)
- Automatic video transcription with Whisper
- Frame extraction and analysis
- OCR and object detection on frames
- AI-powered Q&A functionality
- Real-time chat via WebSockets

## Tech Stack

- **FastAPI**: Modern, fast API framework
- **SQLAlchemy**: ORM for database operations
- **Pydantic**: Data validation and settings management
- **WebSockets**: Real-time communication
- **Whisper**: Audio transcription
- **YOLOv5**: Object detection
- **PyTesseract**: OCR functionality
- **OpenAI**: AI-powered responses

## Setup Instructions

### Prerequisites

- Python 3.8+
- PostgreSQL (recommended) or SQLite for development
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/Cprakash64/luna-backend.git
   cd luna-backend
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following environment variables:
   ```
   # API settings
   SECRET_KEY=your_secret_key_here
   SERVER_HOST=0.0.0.0
   SERVER_PORT=8000
   DEBUG_MODE=True
   
   # Database settings
   DATABASE_URL=postgresql://user:password@localhost/luna
   # Or for SQLite: DATABASE_URL=sqlite:///./luna.db
   
   # Email settings
   MAIL_SERVER=smtp.gmail.com
   MAIL_PORT=587
   MAIL_USE_TLS=True
   MAIL_USERNAME=your_email@gmail.com
   MAIL_PASSWORD=your_app_password
   MAIL_FROM=your_email@gmail.com
   
   # OpenAI settings
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-4o-mini-2024-07-18
   ```

5. Run the application
   ```bash
   uvicorn main:app --reload
   ```
   
6. Open your browser and navigate to `http://localhost:8000/api/docs` to view the API documentation.

## Project Structure

```
luna-backend/
├── app/
│   ├── api/
│   │   └── api_v1/
│   │       ├── endpoints/
│   │       │   ├── auth.py
│   │       │   ├── videos.py
│   │       │   └── chat.py
│   │       └── api.py
│   ├── core/
│   │   ├── auth.py
│   │   ├── config.py
│   │   └── security.py
│   ├── db/
│   │   ├── base.py
│   │   └── session.py
│   ├── models/
│   │   └── models.py
│   ├── schemas/
│   │   ├── user.py
│   │   └── video.py
│   ├── services/
│   │   ├── ai.py
│   │   ├── transcription.py
│   │   └── video_processing.py
│   ├── utils/
│   │   ├── email.py
│   │   └── storage.py
│   └── websockets/
│       └── connection_manager.py
├── main.py
├── requirements.txt
└── README.md
```

## API Endpoints

### Authentication

- `POST /api/auth/signup`: Create a new user
- `POST /api/auth/login`: Login and get access token
- `POST /api/auth/logout`: Logout and clear session
- `GET /api/auth/me`: Get current user information
- `POST /api/auth/password-reset/request`: Request a password reset
- `POST /api/auth/password-reset/confirm`: Reset password using token

### Videos

- `POST /api/videos/upload-video`: Upload and process a video file
- `POST /api/videos/process-url`: Process a YouTube URL
- `GET /api/videos/get-transcription`: Get the transcription for a video
- `GET /api/videos/videos`: Get all videos for the current user
- `GET /api/videos/videos/{video_id}`: Get a specific video
- `DELETE /api/videos/videos/{video_id}`: Delete a video

### Chat

- `POST /api/chat/ask`: Ask the AI a question about a video
- `GET /api/chat/chat-history`: Get chat history for a specific video

### WebSocket

- `WebSocket /ws/chat`: WebSocket endpoint for real-time AI chat

## License

[MIT](LICENSE)