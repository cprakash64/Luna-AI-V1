# Luna AI - Video Analysis Platform

Luna AI is a comprehensive video analysis platform that processes videos to extract valuable insights. This repository contains the FastAPI backend implementation of Luna AI.

## Features

- **Automatic Transcription**: Convert spoken content in videos into text with timestamps
- **Frame Extraction & Analysis**: Capture key frames to analyze visual content
- **Content Search & Timestamped Insights**: Search within video content using keywords
- **Summarization & Key Insights**: Generate concise summaries and highlight main topics
- **Q&A System**: Ask questions about the video with timestamped references
- **Topic Segmentation & Categorization**: Break videos into meaningful sections
- **Emotion & Sentiment Analysis**: Detect the tone of conversations
- **Text Extraction & OCR**: Read and extract text from video frames

## Architecture

Luna AI has been migrated from Flask to FastAPI to take advantage of:

- Asynchronous request handling for better performance
- Automatic API documentation with Swagger UI
- Type validation with Pydantic
- Dependency injection system
- Background tasks for video processing
- WebSockets for real-time communication

## Setup

### Prerequisites

- Python 3.9 or higher
- PostgreSQL database
- ffmpeg (for video processing)
- Tesseract OCR (for text extraction)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/luna-ai.git
   cd luna-ai
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with the following variables:
   ```
   SECRET_KEY=your_secret_key
   DATABASE_URL=postgresql+asyncpg://username:password@localhost/luna_ai
   OPENAI_API_KEY=your_openai_api_key
   MAIL_SERVER=smtp.example.com
   MAIL_PORT=587
   MAIL_USE_TLS=True
   MAIL_USERNAME=your_email@example.com
   MAIL_PASSWORD=your_email_password
   ```

5. Create the database:
   ```bash
   createdb luna_ai  # Using PostgreSQL CLI
   ```

### Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. Open your browser and navigate to:
   - API Documentation: http://localhost:8000/docs
   - Alternative API docs: http://localhost:8000/redoc

## Project Structure

```
luna-ai/
├── main.py                 # FastAPI application entry point
├── models.py               # SQLAlchemy database models
├── schemas.py              # Pydantic models for request/response
├── utils/                  # Utility functions
│   ├── object_detection.py # Object detection utilities
│   ├── video_processing.py # Video processing utilities
├── templates/              # Jinja2 templates for HTML rendering
├── static/                 # Static files (CSS, JS, images)
├── uploads/                # Directory for uploaded videos
├── extracted_frames/       # Directory for extracted video frames
├── transcriptions/         # Directory for saved transcriptions
└── requirements.txt        # Project dependencies
```

## API Endpoints

### Authentication
- `POST /token` - Get access token (login)
- `POST /signup` - Create a new user account

### Video Processing
- `POST /upload-video` - Upload and process a video file
- `GET /process-url` - Process a YouTube video URL
- `POST /detect-object` - Detect a specific object in a video
- `GET /get-transcription` - Get the transcription for a video

### WebSocket
- `WebSocket /ws/{client_id}` - WebSocket connection for real-time communication

## Frontend Integration

The FastAPI backend can be integrated with any frontend framework (React, Vue, Angular, etc.) by making HTTP requests to the API endpoints. For real-time updates during video processing, connect to the WebSocket endpoint.

## Future Improvements

- Implement database migrations with Alembic
- Add unit and integration tests
- Set up CI/CD pipeline
- Add role-based access control
- Implement video processing queue with Celery
- Add more advanced video analysis features
- Optimize performance for larger videos

## License

[MIT License](LICENSE)# Luna-AI-V1
