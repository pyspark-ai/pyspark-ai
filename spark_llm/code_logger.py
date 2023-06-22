import re
import logging
import sys
from pygments import highlight
from pygments.lexers import get_lexer_by_name, PythonLexer, SqlLexer
from pygments.formatters import TerminalFormatter

GREEN = "\033[92m"  # terminal code for green
RESET = "\033[0m"  # reset terminal color


# Custom Formatter
class CustomFormatter(logging.Formatter):
    def format(self, record):
        return GREEN + "INFO: " + RESET + super().format(record)


class CodeLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(CustomFormatter("%(message)s"))  # output only the message
        self.logger.addHandler(handler)

    @staticmethod
    def colorize_code(code, language):
        if not language or language.lower() == "python":
            lexer = PythonLexer()
        elif language.lower() == "sql":
            lexer = SqlLexer()
        else:
            raise ValueError(f"Unsupported language: {language}")
        return highlight(code, lexer, TerminalFormatter())

    def log(self, message):
        # Define pattern to match code blocks with optional language specifiers
        pattern = r"```(python|sql)?(.*?)```"
        # Split message into parts. Every 3rd part will be a code block.
        parts = re.split(pattern, message, flags=re.DOTALL)

        colored_message = ""
        for i in range(0, len(parts), 3):
            # Add regular text to the message
            colored_message += parts[i]
            # If there is a code block, colorize it and add it to the message
            if i + 2 < len(parts):
                colored_message += (
                    "\n```\n" + self.colorize_code(parts[i + 2], parts[i + 1]) + "```"
                )
        # Log the message with colored code blocks
        self.logger.info(colored_message)
