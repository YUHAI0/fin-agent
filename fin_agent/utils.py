from rich.markdown import Markdown, Heading
from rich.panel import Panel
from rich import box
from rich.text import Text

class LeftAlignedHeading(Heading):
    def __rich_console__(self, console, options):
        text = self.text
        text.justify = "left"
        if self.tag == "h1":
            # Draw a border around h1s
            yield Panel(
                text,
                box=box.HEAVY,
                style="markdown.h1.border",
            )
        else:
            # Styled text for h2 and beyond
            if self.tag == "h2":
                yield Text("")
            yield text

class FinMarkdown(Markdown):
    elements = Markdown.elements.copy()
    elements["heading_open"] = LeftAlignedHeading


