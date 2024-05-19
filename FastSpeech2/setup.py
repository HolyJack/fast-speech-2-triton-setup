from setuptools import find_packages, setup

setup(
    name="FastSpeech2",
    version="0.1",
    packages=find_packages(),
    setup_requires=["setuptools_scm"],
    include_package_data=True,
    package_data={
        "FastSpeech2": [
            "configs/**/*.yaml",
            "preprocessed_data/**/*",
            "lexicon/**",
        ]
    },
    description="A brief description of your project",
    python_requires=">=3.6",  # Specify the Python versions you support
)
