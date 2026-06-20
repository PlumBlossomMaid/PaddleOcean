import os

from tqdm import tqdm

os.system("")  # Compatible with Windows


def hex_to_ansi(hex_color: str, background: bool = False) -> str:
    """
    Convert hexadecimal color to ANSI escape sequence

    Args:
        hex_color: Hexadecimal color, e.g., '#dda0a0' or 'dda0a0'
        background: True for background color, False for foreground color

    Returns:
        ANSI escape sequence string, e.g., '\033[38;2;221;160;160m'

    Example:
        >>> print(f"{hex_to_ansi('#dda0a0')}Hello{hex_to_ansi('#000000')} World")
        >>> print(f"{hex_to_ansi('dda0a0', background=True)}Background color{hex_to_ansi.reset()}")
    """
    # Remove # symbol and convert to lowercase
    hex_color = hex_color.lower().lstrip("#")

    # Handle shorthand form (#fff -> ffffff)
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])

    # Convert to RGB values
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # ANSI true color sequence
    # 38;2;R;G;B for foreground, 48;2;R;G;B for background
    code = 48 if background else 38
    return f"\033[{code};2;{r};{g};{b}m"


def rgb_to_ansi(r: int, g: int, b: int, background: bool = False) -> str:
    """Convert RGB values directly to ANSI"""
    code = 48 if background else 38
    return f"\033[{code};2;{r};{g};{b}m"


# ANSI code to reset color
hex_to_ansi.reset = "\033[0m"


class ColoredTqdm(tqdm):
    def __init__(
        self,
        *args,
        start_color=(221, 160, 160),  # RGB: #DDA0A0
        end_color=(160, 221, 160),  # RGB: #A0DDA0
        desc_max_width: int = 30,  # max width for desc text; long descs scroll
        **kwargs,
    ):
        # Set attrs BEFORE super().__init__() so display() can read them
        self.start_color = start_color
        self.end_color = end_color
        self.desc_max_width = desc_max_width
        self._full_desc = kwargs.get("desc", args[1] if len(args) > 1 else "") or ""
        self._scroll_interval = 2
        self._scroll_counter = 0
        self._scroll_offset = 0
        self._needs_scroll = len(self._full_desc) > desc_max_width

        super().__init__(*args, **kwargs)

    def _make_bar_format(self) -> str:
        """Build bar_format with pinned desc width and color."""
        style = hex_to_ansi(self.get_current_color())
        return f"{{desc:{self.desc_max_width}}}{{percentage:3.0f}}%|{style}{{bar}}{hex_to_ansi.reset}|{{r_bar}}"

    def _get_scrolled_desc(self) -> str:
        """Return the current visible window of the description (marquee)."""
        if not self._needs_scroll:
            return self._full_desc.ljust(self.desc_max_width)
        end = self._scroll_offset + self.desc_max_width
        if end <= len(self._full_desc):
            return self._full_desc[self._scroll_offset : end]
        # Wrap around: tail + gap + head
        tail = self._full_desc[self._scroll_offset :]
        head = self._full_desc[: end - len(self._full_desc)]
        return (tail + " " + head).ljust(self.desc_max_width)

    def get_current_color(self):

        if self.total is None:
            return "#FFFFFF"

        progress = self.n / self.total if self.total > 0 else 0
        current_rgb = tuple(
            int(start + (end - start) * progress) for start, end in zip(self.start_color, self.end_color)
        )
        result = current_rgb[0] * 16**4 + current_rgb[1] * 16**2 + current_rgb[2] * 16**0
        return "%06x" % result

    def display(self, msg=None, pos=None):
        # Advance marquee scroll on every display call
        if self._needs_scroll:
            self._scroll_counter += 1
            if self._scroll_counter >= self._scroll_interval:
                self._scroll_counter = 0
                self._scroll_offset = (self._scroll_offset + 1) % len(self._full_desc)
            self.desc = self._get_scrolled_desc()
        self.bar_format = self._make_bar_format()
        super().display(msg, pos)

    def update(self, n=1):
        # parent's update() calls display() internally, which our override handles
        super().update(n)
