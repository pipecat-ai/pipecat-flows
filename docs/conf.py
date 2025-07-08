import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sphinx-build")

# Add source directory to path
docs_dir = Path(__file__).parent
project_root = docs_dir.parent
sys.path.insert(0, str(project_root / "src"))

# Project information
project = "pipecat-ai-flows"
current_year = datetime.now().year
copyright = f"2024-{current_year}, Daily" if current_year > 2024 else "2024, Daily"
author = "Daily"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Suppress warnings about mocked objects
suppress_warnings = [
    "autodoc.mocked_object",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_include_init_with_doc = True

# AutoDoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "no-index": True,
    "show-inheritance": True,
}

# Mock imports for optional dependencies (if any)
autodoc_mock_imports = [
    # Add any optional dependencies here that might not be available during doc builds
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pipecat": ("https://docs.pipecat.ai/api", None),
}

# HTML output settings
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
autodoc_typehints = "signature"
html_show_sphinx = False


def clean_title(title: str) -> str:
    """Automatically clean module titles."""
    # Remove everything after space (like 'module', 'processor', etc.)
    title = title.split(" ")[0]

    # Get the last part of the dot-separated path
    parts = title.split(".")
    title = parts[-1]

    # Special cases for common terms
    special_cases = {
        "ai": "AI",
        "flows": "Flows",
    }

    # Check if the entire title is a special case
    if title.lower() in special_cases:
        return special_cases[title.lower()]

    # Otherwise, capitalize each word
    words = title.split("_")
    cleaned_words = []
    for word in words:
        if word.lower() in special_cases:
            cleaned_words.append(special_cases[word.lower()])
        else:
            cleaned_words.append(word.capitalize())

    return " ".join(cleaned_words)


def setup(app):
    """Generate API documentation during Sphinx build."""
    from sphinx.ext.apidoc import main

    docs_dir = Path(__file__).parent
    project_root = docs_dir.parent
    output_dir = str(docs_dir / "api")
    source_dir = str(project_root / "src" / "pipecat_flows")

    # Clean existing files
    if Path(output_dir).exists():
        import shutil

        shutil.rmtree(output_dir)
        logger.info(f"Cleaned existing documentation in {output_dir}")

    logger.info(f"Generating API documentation...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Source directory: {source_dir}")

    excludes = [
        "**/test_*.py",
        "**/tests/*.py",
    ]

    try:
        main(
            [
                "-f",  # Force overwriting
                "-e",  # Don't generate empty files
                "-M",  # Put module documentation before submodule documentation
                "--no-toc",  # Don't create a table of contents file
                "--separate",  # Put documentation for each module in its own page
                "--module-first",  # Module documentation before submodule documentation
                "--implicit-namespaces",  # Handle implicit namespace packages
                "-o",
                output_dir,
                source_dir,
            ]
            + excludes
        )

        logger.info("API documentation generated successfully!")

        # Process generated RST files to update titles
        for rst_file in Path(output_dir).glob("**/*.rst"):
            content = rst_file.read_text()
            lines = content.split("\n")

            # Find and clean up the title
            if lines and len(lines) > 1 and "=" in lines[1]:
                old_title = lines[0]
                new_title = clean_title(old_title)
                content = content.replace(old_title, new_title)
                rst_file.write_text(content)
                logger.info(f"Updated title: {old_title} -> {new_title}")

    except Exception as e:
        logger.error(f"Error generating API documentation: {e}", exc_info=True)
