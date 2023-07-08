from setuptools import setup, find_packages

setup(
    name='trendalation',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    version='0.1.3',
    license='MIT',
    author="Raghav Saboo, Kartikey Sinha",
    author_email='raghs2000@gmail.com',
    url='https://github.com/kartikeysinha/trendalation',
    keywords=['anomaly detection', 'procrustes', 'time-series', 'trends'],
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
  ],
)