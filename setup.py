from setuptools import find_packages, setup

if __name__ == "__main__":
    print('setup executed')
    setup(
        name="mrc_modules",
        version="1.0.0",
        author="sifter",
        description="MRC Search Engine Module Project",
        keywords=['tests', 'modules'],
        install_requires=[],
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent", ],
        python_requires='>=3.9',
    )