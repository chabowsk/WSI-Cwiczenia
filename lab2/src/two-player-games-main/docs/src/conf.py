import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

project = 'Two-player games'
author = 'Jakub Łyskawa'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_markdown_builder',
]