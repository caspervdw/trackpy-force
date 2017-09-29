import os
from setuptools import setup


try:
    descr = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()
except IOError:
    descr = ''

try:
    from pypandoc import convert
    descr = convert(descr, 'rst', format='md')
except ImportError:
    pass


# In some cases, the numpy include path is not present by default.
# Let's try to obtain it.
try:
    import numpy
except ImportError:
    ext_include_dirs = []
else:
    ext_include_dirs = [numpy.get_include(),]

setup_parameters = dict(
    name = "trackpy-force",
    version = "0.1",
    description = "Extract forces from trajectories of Brownian particles",
    author = "Casper van der Wel",
    author_email = "caspervdw@gmail.com",
    url = "https://github.com/caspervdw/trackpy-force",
    install_requires = ['numpy>=1.7', 'scipy>=0.12', 'six>=1.8',
	                    'pandas>=0.13', 'trackpy>=0.3.1'],
    packages = ['tpforce', 'tpforce.test'],
    long_description = descr,
)

setup(**setup_parameters)
