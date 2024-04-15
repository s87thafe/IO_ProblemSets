# Industrial Organisation Problem Sets

## Introduction
This repository contains the Python code and necessary documentation for the Industrial Organisation module in the spring-semester 2024.

## Setup Instructions

### Prerequisites
- Python 3.x
- pip (Python package installer)
- virtualenv (optional but recommended for creating isolated Python environments)

### Clone the Repository
First, clone the repository to your local machine using:

```bash
git clone https://github.com/s87thafe/IO_ProblemSets.git
cd IO_ProblemSets
```

### Setting Up the Virtual Environment
To set up and activate the virtual environment, follow these steps:

#### For Windows Users
```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create the virtual environment
virtualenv venv

# Activate the virtual environment
venv\Scripts\activate
```

#### For MacOS/Linux Users
```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create the virtual environment
virtualenv venv

# Activate the virtual environment
source venv/bin/activate
```

### Install Required Packages
Install all required packages using the requirements file:

```bash
pip install -r requirements.txt
```
## Daily Usage
Whenever you start working on the project, ensure your virtual environment is activated. This keeps dependencies managed and conflicts minimized.

### Activating the Virtual Environment

#### Windows
```bash
venv\Scripts\activate
```

#### MacOS/Linux
```bash
source venv/bin/activate
```

### Deactivating the Virtual Environment
When you're done working, you can deactivate the virtual environment by typing:
```bash
deactivate
```

### Committing Changes to Git
Regularly commit your changes to keep track of the project's progress and backup your work:

```bash
git status  # View changed files
git add .   # Add all changes to the staging area
git commit -m "Your commit message"  # Commit your changes
git push origin master  # Push changes to remote repository
```
