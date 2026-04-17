"""CLI for conv - file converter."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="conv",
    help="Fast file converter from terminal",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()

# Supported conversions
VIDEO_FORMATS = {"mp4", "webm", "mov", "avi", "mkv"}
AUDIO_FORMATS = {"mp3", "wav", "m4a", "aac", "ogg", "flac"}
IMAGE_FORMATS = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "ico", "heic"}
SPECIAL_TARGETS = {"noaudio"}  # special conversion targets

# Whisper config
WHISPER_PATH = os.getenv("WHISPER_PATH", "/opt/homebrew/opt/whisper-cpp/bin/whisper-cli")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_MODELS_DIR = Path(os.getenv("WHISPER_MODELS_DIR", str(Path.home() / "whisper.cpp" / "models")))


def _clean_repetitions(text: str) -> str:
    """Remove repeated phrases/sentences from transcript."""
    # Remove exact duplicate sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    cleaned = []
    for s in sentences:
        s_normalized = s.strip().lower()
        if s_normalized and s_normalized not in seen:
            seen.add(s_normalized)
            cleaned.append(s)

    text = ' '.join(cleaned)

    # Remove repeated phrases (3+ consecutive same words/phrases)
    text = re.sub(r'\b(\w+(?:\s+\w+){0,3})\s+(?:\1\s*){2,}', r'\1 ', text)

    # Remove repeated single words (4+ times)
    text = re.sub(r'\b(\w+)\s+(?:\1\s+){3,}', r'\1 ', text)

    return text.strip()


def check_whisper() -> bool:
    """Check if whisper-cpp is installed."""
    return Path(WHISPER_PATH).exists()


def get_whisper_model_path() -> Path:
    """Get path to whisper model file."""
    model_file = f"ggml-{WHISPER_MODEL}.bin"
    return WHISPER_MODELS_DIR / model_file


def extract_audio_for_whisper(input_path: Path, output_path: Path) -> bool:
    """Extract audio and convert to WAV 16kHz mono for whisper."""
    if not check_ffmpeg():
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",      # mono
        "-c:a", "pcm_s16le",  # 16-bit PCM
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def transcribe_audio(audio_path: Path, language: Optional[str] = None) -> Optional[str]:
    """Transcribe audio file using whisper.cpp."""
    model_path = get_whisper_model_path()
    if not model_path.exists():
        console.print(f"[red]Error: Whisper model not found at {model_path}[/red]")
        console.print(f"Download with: curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{WHISPER_MODEL}.bin")
        return None

    cmd = [
        WHISPER_PATH,
        "-m", str(model_path),
        "-f", str(audio_path),
        "--no-timestamps",
        "--entropy-thold", "2.4",
        "--no-fallback",
    ]

    if language:
        # Normalize language codes (eng -> en, rus -> ru, etc.)
        lang_map = {"eng": "en", "rus": "ru", "jpn": "ja", "deu": "de", "fra": "fr", "spa": "es"}
        lang = lang_map.get(language.lower(), language.lower())
        cmd.extend(["-l", lang])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[red]Whisper error: {result.stderr}[/red]")
            return None

        transcript = result.stdout.strip()

        # Check for .txt file output
        txt_path = audio_path.with_suffix(".txt")
        if txt_path.exists():
            transcript = txt_path.read_text().strip()
            txt_path.unlink()

        # Clean up repetitions
        transcript = _clean_repetitions(transcript)
        return transcript
    except Exception as e:
        console.print(f"[red]Whisper error: {e}[/red]")
        return None


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_output_path(input_path: Path, target_format: str, output: Optional[Path], source_format: str = "") -> Path:
    """Generate output path."""
    if output:
        return output
    if target_format == "noaudio":
        # Keep same extension, add _noaudio suffix
        return input_path.with_stem(f"{input_path.stem}_noaudio")
    return input_path.with_suffix(f".{target_format}")


def convert_with_pillow(input_path: Path, output_path: Path) -> bool:
    """Convert image using Pillow."""
    try:
        from PIL import Image

        img = Image.open(input_path)
        target_ext = output_path.suffix.lower()

        # Check if animated
        is_animated = getattr(img, "is_animated", False)

        if is_animated and target_ext == ".gif":
            # Save animated image as GIF
            frames = []
            try:
                while True:
                    frame = img.copy()
                    if frame.mode != "RGBA":
                        frame = frame.convert("RGBA")
                    frames.append(frame)
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            if frames:
                # Get duration from original
                duration = img.info.get("duration", 100)
                frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=duration,
                    loop=0,
                    disposal=2,
                )
                return True

        # Handle RGBA to RGB conversion for formats that don't support alpha
        if target_ext in {".jpg", ".jpeg"} and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img.save(output_path)
        return True
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


def convert_animated_to_video(
    input_path: Path,
    output_path: Path,
    fps: int = 10,
) -> bool:
    """Convert animated image (webp/gif) to video using Pillow + ffmpeg."""
    if not check_ffmpeg():
        console.print("[red]Error: ffmpeg not found. Install it with:[/red]")
        console.print("  brew install ffmpeg")
        return False

    try:
        from PIL import Image

        img = Image.open(input_path)
        duration = img.info.get("duration", 100)  # ms per frame
        actual_fps = 1000 / duration if duration > 0 else fps

        # Create temp directory for frames
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            frame_num = 0

            try:
                while True:
                    frame = img.copy()
                    if frame.mode != "RGB":
                        frame = frame.convert("RGB")
                    frame.save(tmppath / f"frame_{frame_num:05d}.png")
                    frame_num += 1
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            if frame_num == 0:
                console.print("[red]Error: No frames found in image[/red]")
                return False

            # Use ffmpeg to create video from frames
            # pad filter ensures dimensions are divisible by 2 (required for h264)
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(actual_fps),
                "-i", str(tmppath / "frame_%05d.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                console.print(f"[red]ffmpeg error: {result.stderr}[/red]")
                return False

            return True
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


def remove_audio(input_path: Path, output_path: Path) -> bool:
    """Remove audio from video file."""
    if not check_ffmpeg():
        console.print("[red]Error: ffmpeg not found. Install it with:[/red]")
        console.print("  brew install ffmpeg")
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-an",  # no audio
        "-c:v", "copy",  # copy video stream without re-encoding
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[red]ffmpeg error: {result.stderr}[/red]")
            return False
        return True
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


def convert_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    source_format: str,
    target_format: str,
    fps: int = 10,
    scale: Optional[int] = None,
) -> bool:
    """Convert using ffmpeg."""
    if not check_ffmpeg():
        console.print("[red]Error: ffmpeg not found. Install it with:[/red]")
        console.print("  brew install ffmpeg")
        return False

    cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    # Special handling for GIF output
    if target_format == "gif":
        filter_parts = [f"fps={fps}"]
        if scale:
            filter_parts.append(f"scale={scale}:-1:flags=lanczos")
        else:
            filter_parts.append("scale=480:-1:flags=lanczos")

        filter_str = ",".join(filter_parts)
        cmd.extend([
            "-vf", f"{filter_str},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop", "0",
        ])

    cmd.append(str(output_path))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[red]ffmpeg error: {result.stderr}[/red]")
            return False
        return True
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


def show_help():
    """Show usage help."""
    console.print("[bold]conv[/bold] - Fast file converter\n")
    console.print("Usage: conv to <target> <file>\n")
    console.print("Examples:")
    console.print("  conv to gif video.mp4")
    console.print("  conv to png image.webp")
    console.print("  conv to noaudio video.mp4")
    console.print("  conv transcript video.mp4")
    console.print("\nRun [cyan]conv formats[/cyan] to see all supported formats")


def run_conversion(
    source_format: str,
    target_format: str,
    path: Path,
    output: Optional[Path],
    fps: int,
    scale: Optional[int],
):
    """Run the actual conversion."""
    source_format = source_format.lower().lstrip(".")
    target_format = target_format.lower().lstrip(".")

    if not path.exists():
        console.print(f"[red]Error: File not found: {path}[/red]")
        raise typer.Exit(1)

    output_path = get_output_path(path, target_format, output, source_format)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Converting {path.name} to {target_format}...", total=None)

        # Determine conversion method
        if target_format == "noaudio":
            if source_format not in VIDEO_FORMATS:
                console.print(f"[red]Error: noaudio only works with video files[/red]")
                raise typer.Exit(1)
            success = remove_audio(path, output_path)
        elif source_format in VIDEO_FORMATS:
            # Video input -> use ffmpeg
            success = convert_with_ffmpeg(
                path, output_path, source_format, target_format, fps, scale
            )
        elif target_format in VIDEO_FORMATS:
            # Image to video -> extract frames and use ffmpeg
            success = convert_animated_to_video(path, output_path, fps)
        else:
            # Image to image -> use Pillow
            success = convert_with_pillow(path, output_path)

    if success:
        size_kb = output_path.stat().st_size / 1024
        console.print(f"[green]✓[/green] Converted to: {output_path} ({size_kb:.1f} KB)")
    else:
        raise typer.Exit(1)


@app.command("formats")
def formats_cmd():
    """Show supported formats."""
    console.print("\n[bold]Supported formats:[/bold]\n")
    console.print("[cyan]Video:[/cyan]", ", ".join(sorted(VIDEO_FORMATS)))
    console.print("[cyan]Image:[/cyan]", ", ".join(sorted(IMAGE_FORMATS)))
    console.print("[cyan]Special:[/cyan] noaudio (remove audio from video)")
    console.print("\n[bold]Common conversions:[/bold]")
    console.print("  conv to gif video.mp4")
    console.print("  conv to png image.webp")
    console.print("  conv to noaudio video.mp4")
    console.print("  conv to mp4 video.mov")
    console.print()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    to: Optional[str] = typer.Argument(None, help="Should be 'to' or 'formats'"),
    target_format: Optional[str] = typer.Argument(None, help="Target format (e.g., gif, png)"),
    path: Optional[Path] = typer.Argument(None, help="Path to file to convert"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file path"),
    fps: int = typer.Option(10, "--fps", help="FPS for GIF output"),
    scale: Optional[int] = typer.Option(None, "--scale", help="Width for GIF output"),
):
    """
    Convert files between formats.

    Examples:
        conv to gif video.mp4
        conv to png image.webp
        conv to jpg photo.png -o output.jpg
        conv to gif video.mp4 --fps 15 --scale 320
    """
    # Check if a subcommand is being invoked
    if ctx.invoked_subcommand is not None:
        return

    # Handle special commands
    if to == "formats":
        formats_cmd()
        return

    if to == "transcript":
        # Parse args for transcript command
        args = sys.argv[2:]  # skip 'conv' and 'transcript'
        file_path = None
        language = None
        out_path = None

        i = 0
        while i < len(args):
            arg = args[i]
            if arg in ("-l", "--language") and i + 1 < len(args):
                language = args[i + 1]
                i += 2
            elif arg in ("-o", "--output") and i + 1 < len(args):
                out_path = Path(args[i + 1])
                i += 2
            elif arg in ("--help", "-h"):
                console.print("[bold]conv transcript[/bold] - Transcribe audio/video\n")
                console.print("Usage: conv transcript <file> [OPTIONS]\n")
                console.print("Options:")
                console.print("  -l, --language TEXT   Language code (en, ru, ja)")
                console.print("  -o, --output PATH     Output text file path")
                return
            elif not arg.startswith("-"):
                file_path = Path(arg)
                i += 1
            else:
                i += 1

        if not file_path:
            console.print("Usage: conv transcript <file> [-l language] [-o output]")
            return

        transcript_cmd(file_path, language=language, output=out_path)
        return

    if not all([to, target_format, path]):
        show_help()
        return

    if to.lower() != "to":
        console.print(f"[red]Error: Expected 'to', got '{to}'[/red]")
        console.print("Usage: conv to <target_format> <file>")
        raise typer.Exit(1)

    # Get source format from file extension
    source_format = path.suffix.lower().lstrip(".")
    if not source_format:
        console.print("[red]Error: Cannot determine file format (no extension)[/red]")
        raise typer.Exit(1)

    run_conversion(source_format, target_format, path, output, fps, scale)


@app.command("transcript")
def transcript_cmd(
    path: Path = typer.Argument(..., help="Path to audio/video file"),
    language: Optional[str] = typer.Option(None, "-l", "--language", help="Language code (e.g., en, ru, ja)"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output text file path"),
):
    """
    Transcribe audio/video to text using whisper.

    Examples:
        conv transcript video.mp4
        conv transcript audio.mp3 -l ru
        conv transcript video.mp4 -o transcript.txt
    """
    if not path.exists():
        console.print(f"[red]Error: File not found: {path}[/red]")
        raise typer.Exit(1)

    if not check_whisper():
        console.print(f"[red]Error: whisper-cpp not found at {WHISPER_PATH}[/red]")
        console.print("Install with: brew install whisper-cpp")
        raise typer.Exit(1)

    source_format = path.suffix.lower().lstrip(".")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Extract audio if video
        if source_format in VIDEO_FORMATS:
            progress.add_task("Extracting audio...", total=None)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            if not extract_audio_for_whisper(path, tmp_path):
                console.print("[red]Error: Failed to extract audio[/red]")
                raise typer.Exit(1)

            audio_path = tmp_path
        elif source_format in AUDIO_FORMATS:
            # Convert to WAV 16kHz if needed
            if source_format != "wav":
                progress.add_task("Converting audio...", total=None)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                if not extract_audio_for_whisper(path, tmp_path):
                    console.print("[red]Error: Failed to convert audio[/red]")
                    raise typer.Exit(1)

                audio_path = tmp_path
            else:
                audio_path = path
        else:
            console.print(f"[red]Error: Unsupported format for transcription: {source_format}[/red]")
            raise typer.Exit(1)

        # Transcribe
        progress.add_task("Transcribing with whisper...", total=None)
        transcript = transcribe_audio(audio_path, language)

        # Cleanup temp file
        if audio_path != path and audio_path.exists():
            audio_path.unlink()

    if not transcript:
        console.print("[red]Transcription failed![/red]")
        raise typer.Exit(1)

    # Save to file
    output_path = output or path.with_suffix(".txt")
    output_path.write_text(transcript)

    word_count = len(transcript.split())
    console.print(f"[green]✓[/green] Transcribed to: {output_path} ({word_count} words)")


# Standalone transcript app
transcript_app = typer.Typer(add_completion=False)


@transcript_app.command()
def transcript_main(
    path: Path = typer.Argument(..., help="Path to audio/video file"),
    language: Optional[str] = typer.Option(None, "-l", "--language", help="Language code (e.g., en, ru, ja)"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output text file path"),
):
    """Transcribe audio/video to text using whisper."""
    transcript_cmd(path, language, output)


if __name__ == "__main__":
    app()
