# New_MASCADO

Extention Package for MASCADO.

Install
Use setuptools via

python setup.py install [--user] [--dry-run]
in the project root directory. This makes the mascado package available as import and puts the executable scripts into an appropriate directory for executables. In the output, setuptools tells you where exactly.

Dependencies are

Python3.5 or higher
3.5 or higher needed for matrix multiplication operator (@)
Matplotlib
NumPy >= 1.14
1.14 or higher needed for encoding keyword to numpy.genfromtxt()
SciPy
pandas
Make sure that you are using the Python3 version of pip (explicitly pip3 ... or python3 -m pip ...) or conda, because otherwise the packages are not visible to the scripts.

Documentation
The documentation for MASCADO is available at https://tychons.net/mascado/

License
Copyright 2020 Jonas Sauter jonas.sauter@web.de
at Max-Planck-Institute for Astronomy, Heidelberg.
Licensed under GPL-3.0-or-later.
You can find the full text of the license in COPYING.

If you plan to publish the obtained results, please contact the authors:

Jonas Sauter jonas.sauter@web.de
Hannes Riechert hannes@tychons.net or riechert@mpia.de
Jörg-Uwe Pott jpott@mpia.de

© 2020 GitHub, Inc.
