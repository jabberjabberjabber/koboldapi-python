from setuptools import setup, find_packages

setup(
    name="koboldapi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "kobold-image=examples.image_example:main",
            "kobold-text=examples.text_example:main",
        ],
    },
    python_requires=">=3.8",
)