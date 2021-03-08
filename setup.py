import setuptools

requirements = []
with open('requirements.txt', 'rt') as f:
    for req in f.read().splitlines():
        if req.startswith('git+'):
            pkg_name = req.split('/')[-1].replace('.git', '')
            if "#egg=" in pkg_name:
                pkg_name = pkg_name.split("#egg=")[1]
            requirements.append(f'{pkg_name} @ {req}')
        else:
            requirements.append(req)
            
with open("README.md", "r") as fh:
    long_description = fh.read()
 
          
setuptools.setup(
    name="ParadeRanker",
    version="0.0.1",
    author="GabrielGeorgiev",
    author_email='gabrielgeorgiev1998{at}.gmail.com',
    description="PARADE implementation for PyTerrier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
)
