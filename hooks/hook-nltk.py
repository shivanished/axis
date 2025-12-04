"""
Dummy hook for nltk to prevent PyInstaller from trying to import it.
This prevents scipy/numpy compatibility issues during build analysis.
"""
# Explicitly exclude nltk and all its submodules
excludedimports = ['nltk']
datas = []
hiddenimports = []

