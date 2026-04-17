"""CLI for conv - file converter."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="conv",
    help="Fast file converter from terminal",
    add_completion=False,
)
console = Console()

# Supported conversions
VIDEO_FORMATS = {"mp4", "webm", "mov", "avi", "mkv"}
IMAGE_FORMATS = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "ico", "heic"}
FFMPEG_REQUIRED = {"gif"} | VIDEO_FORMATS


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


def get_output_path(input_path: Path, target_format: str, output: Optional[Path]) -> Path:
    """Generate output path."""
    if output:
        return output
    return input_path.with_suffix(f".{target_format}")


def convert_with_pillow(input_path: Path, output_path: Path) -> bool:
    """Convert image using Pillow."""
    try:
        from PIL import Image

        img = Image.open(input_path)

        # Handle RGBA to RGB conversion for formats that don't support alpha
        if output_path.suffix.lower() in {".jpg", ".jpeg"} and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img.save(output_path)
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


@app.command()
def convert(
    source_format: str = typer.Argument(..., help="Source format (e.g., mp4, webp, jpeg)"),
    to: str = typer.Argument(..., help="Should be 'to'"),
    target_format: str = typer.Argument(..., help="Target format (e.g., gif, png)"),
    path: Path = typer.Argument(..., help="Path to file to convert"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file path"),
    fps: int = typer.Option(10, "--fps", help="FPS for GIF output"),
    scale: Optional[int] = typer.Option(None, "--scale", help="Width for GIF output"),
):
    """
    Convert files between formats.

    Examples:
        conv mp4 to gif video.mp4
        conv webp to png image.webp
        conv jpeg to png photo.jpg -o output.png
        conv mp4 to gif video.mp4 --fps 15 --scale 320
    """
    if to.lower() != "to":
        console.print(f"[red]Error: Expected 'to', got '{to}'[/red]")
        console.print("Usage: conv <source_format> to <target_format> <file>")
        raise typer.Exit(1)

    source_format = source_format.lower().lstrip(".")
    target_format = target_format.lower().lstrip(".")

    if not path.exists():
        console.print(f"[red]Error: File not found: {path}[/red]")
        raise typer.Exit(1)

    actual_ext = path.suffix.lower().lstrip(".")
    if actual_ext != source_format and not (
        actual_ext in {"jpg", "jpeg"} and source_format in {"jpg", "jpeg"}
    ):
        console.print(f"[yellow]Warning: File extension '{actual_ext}' doesn't match '{source_format}'[/yellow]")

    output_path = get_output_path(path, target_format, output)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Converting {path.name} to {target_format}...", total=None)

        # Determine conversion method
        needs_ffmpeg = (
            source_format in VIDEO_FORMATS
            or target_format in VIDEO_FORMATS
            or target_format == "gif"
            or source_format == "gif"
        )

        if needs_ffmpeg:
            success = convert_with_ffmpeg(
                path, output_path, source_format, target_format, fps, scale
            )
        else:
            success = convert_with_pillow(path, output_path)

    if success:
        size_kb = output_path.stat().st_size / 1024
        console.print(f"[green]✓[/green] Converted to: {output_path} ({size_kb:.1f} KB)")
    else:
        raise typer.Exit(1)


@app.command()
def formats():
    """Show supported formats."""
    console.print("\n[bold]Supported formats:[/bold]\n")
    console.print("[cyan]Video:[/cyan]", ", ".join(sorted(VIDEO_FORMATS)))
    console.print("[cyan]Image:[/cyan]", ", ".join(sorted(IMAGE_FORMATS)))
    console.print("\n[bold]Common conversions:[/bold]")
    console.print("  conv mp4 to gif video.mp4")
    console.print("  conv webp to png image.webp")
    console.print("  conv png to jpg image.png")
    console.print("  conv mov to mp4 video.mov")
    console.print()


if __name__ == "__main__":
    app()
