# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import alabaster
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../dronebuddylib'))
sys.path.insert(0, os.path.abspath('../dronebuddylib/offline'))
sys.path.insert(0, os.path.abspath('../dronebuddylib/offline/molecules'))
sys.path.insert(0, os.path.abspath('../dronebuddylib/offline/atoms'))
sys.path.insert(0, os.path.abspath('../dronebuddylib/online'))
sys.path.insert(0, os.path.abspath('../dronebuddylib/online/atoms'))

project = 'DroneBuddy Library'
copyright = '2023, NUS'
author = 'NUS'
release = 'V1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "alabaster", "sphinx.ext.viewcode", "sphinx.ext.napoleon",
              "sphinx_autodoc_typehints"]
# Extension Settings
# autosummary_generate = True  # Turn on sphinx.ext.autosummary
# html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
# autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
# set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
# autodoc_mock_imports = ["dronebuddylib"]
keep_warnings = True  # Keep warnings in output (helpful for debugging)
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'dist/*', 'setup.py', 'venv/*', 'venvnew/*']
all_files = True
warnings_is_error = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
