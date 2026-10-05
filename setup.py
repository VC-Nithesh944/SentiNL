"""
Setup script to install required packages and train the model
"""
import subprocess
import sys

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    packages = [
        'nltk==3.9.1',
        'scikit-learn==1.5.2',
        'numpy==2.1.2'
    ]

    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

    print("\nAll packages installed successfully!")

if __name__ == "__main__":
    install_requirements()