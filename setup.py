from setuptools import setup, find_packages

setup(
    name='trendalation',
    packages=['trendalation'],
    version='0.0.1',
    license='MIT',
    author="Raghav Saboo, Kartikey Sinha",
    author_email='raghs2000@gmail.com',
    # packages=find_packages('src'),
    # package_dir={'': 'src'},
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