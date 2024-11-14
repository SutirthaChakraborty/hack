import os
import json
from pathlib import Path
import whisper
from whisper.utils import get_writer
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from anthropic import AnthropicBedrock

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create required directories
UPLOAD_DIR = Path("uploads")
TRANSCRIPTION_DIR = Path("transcriptions")
UPLOAD_DIR.mkdir(exist_ok=True)
TRANSCRIPTION_DIR.mkdir(exist_ok=True)

# Load Whisper model
model = whisper.load_model("turbo")

# Initialize Claude client
client = AnthropicBedrock(
    aws_access_key="",
    aws_secret_key="",
    aws_session_token="",
    aws_region="us-west-2",
)


def get_file_extension(filename: str) -> str:
    """Get the file extension from a filename."""
    return Path(filename).suffix.lower()


def transcribe_audio(input_file: Path) -> Path:
    """Transcribes the audio using Whisper and saves the transcription as an SRT file."""
    try:
        # Generate output filename
        output_file = TRANSCRIPTION_DIR / f"{input_file.stem}.srt"

        # Perform transcription
        result = model.transcribe(str(input_file))
        srt_writer = get_writer("srt", str(TRANSCRIPTION_DIR))
        srt_writer(result, str(input_file))

        print(f"Transcription saved as SRT: '{output_file}'")
        return output_file
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


def query_claude(query: str) -> dict:
    """Query Claude API and return the response."""
    try:
        message = client.messages.create(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            max_tokens=8192,
            messages=[{"role": "user", "content": query}],
        )
        response_text = (
            message.content[0].text
            if isinstance(message.content, list)
            else message.content
        )
        return json.loads(response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)}")


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Handle video upload, transcription, and analysis."""
    # Validate file extension
    if get_file_extension(file.filename) not in ['.mp4', '.avi', '.mov', '.mkv']:
        raise HTTPException(status_code=400, detail="Invalid video file format")

    try:
        # Save uploaded file
        input_path = UPLOAD_DIR / file.filename
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Transcribe video
        srt_path = transcribe_audio(input_path)

        # Read transcription
        with open(srt_path, 'r', encoding='utf-8') as srt_file:
            srt_text = srt_file.read()

        # Generate prompt for Claude
        prompt = f"""As an intelligent video content script editor and marketing professional, analyze the SRT content and perform the following tasks:
            1. Select a list of small segments with the best sections' START and END timestamps to create a highly engaging highlight of this video, keeping important parts without repeating.
            2. Provide 6 category names that best describe the video.
            3. List 6 interests or preferences the viewer might have.
            4. Identify the countries where viewers might be watching from.
            5. Suggest 5 commercial brands or products that might be relevant or useful to the viewer.
            6. Describe the possible mood or emotional state of the viewer.
            7. Estimate the age range of the viewer.
            8. Characterize the personality traits of the viewer.
            9. Offer additional insights about the viewer's preferences and behaviors that can help us understand them better, including:
            - Preferred social media platforms.
            - Likely purchasing behaviors (e.g., impulse buyer, value seeker).
            - Possible hobbies or leisure activities.
            - Technology usage patterns.
            - Content consumption habits (e.g., binge-watching, casual viewing).
            - Potential life events or milestones influencing their interests.
            10. Provide a summary of the main themes and topics covered in the video.
            11. Extract 10 keywords or phrases that are most significant in the video content.
            12. Analyze the overall sentiment of the video (e.g., positive, negative, neutral).
            13. Identify any trending topics or timely subjects mentioned in the video.
            14. Describe the style and tone of the content (e.g., humorous, educational, inspirational).
            15. Suggest related content or topics that viewers might be interested in after watching this video.

            Remember, DO NOT WRITE EXTRA COMMENTARY and REPLY IN JSON FORMAT.
            Ensure the response contains the full JSON query.
            For each item, include a confidence score after each entry.
            The SRT of the video is:

            {srt_text}
            """

        # Get analysis from Claude
        analysis = query_claude(prompt)

        # Clean up files
        os.remove(input_path)
        os.remove(srt_path)

        return analysis

    except Exception as e:
        # Ensure cleanup in case of errors
        if 'input_path' in locals():
            os.remove(input_path)
        if 'srt_path' in locals():
            os.remove(srt_path)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
