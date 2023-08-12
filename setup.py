from setuptools import setup, find_packages

from pathlib import Path
curr_dir = Path(__file__).parent
long_description = (curr_dir / "README.md").read_text()

setup(
    name='trendalation',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    version='1.0.2',
    license='MIT',
    author=['Raghav Saboo', 'Kartikey Sinha'],
    author_email='raghs2000@gmail.com',
    url='https://github.com/kartikeysinha/trendalation',
    keywords=['anomaly detection', 'time-series', 'procrustes', 'trends', 'spatial analysis'],
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy',
        'pytest'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    
    # Inherit description from README.md
    long_description=long_description,
    long_description_content_type='text/markdown'
)