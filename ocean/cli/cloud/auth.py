"""cloud auth — AI Studio token management."""

import os
from pathlib import Path

import click

TOKEN_FILE = Path.home() / ".cache" / "ocean" / ".auth" / "token"


def _ensure_token_dir():
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)


def save_token(token: str) -> None:
    """Save AI Studio token to local file."""
    _ensure_token_dir()
    TOKEN_FILE.write_text(token.strip())
    TOKEN_FILE.chmod(0o600)
    click.echo("✅ Token saved.")


def get_token() -> str:
    """Read AI Studio token from env or local file.

    Raises:
        click.ClickException: If no token is found.
    """
    token = os.environ.get("AISTUDIO_ACCESS_TOKEN")
    if token:
        return token
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    raise click.ClickException(
        "No AI Studio token found. Run 'ocean cloud login --token YOUR_TOKEN' first, "
        "or set AISTUDIO_ACCESS_TOKEN environment variable."
    )


def get_token_optional() -> str | None:
    """Read AI Studio token, returning None if none is configured."""
    token = os.environ.get("AISTUDIO_ACCESS_TOKEN")
    if token:
        return token
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    return None


def clear_token() -> None:
    """Remove saved token file."""
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
        click.echo("✅ Token cleared.")
    else:
        click.echo("No token to clear.")


@click.command()
@click.option("--token", "-t", required=True, help="AI Studio access token.")
def login(token: str) -> None:
    """Log in to AI Studio with an access token.

    Example:

        ocean cloud login --token abc123def456
    """
    save_token(token)


@click.command()
def logout() -> None:
    """Log out by removing saved token."""
    clear_token()
