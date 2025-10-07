# Create the complete folder structure
mkdir -p app/api
mkdir -p app/schemas

# Create all necessary __init__.py files
touch app/__init__.py
touch app/api/__init__.py
touch app/schemas/__init__.py

# Your folder structure should look like this:
# TEST/
# ├── app/
# │   ├── __init__.py
# │   ├── main.py
# │   ├── api/
# │   │   ├── __init__.py
# │   │   └── routes.py
# │   └── schemas/
# │       ├── __init__.py
# │       └── base.py
# ├── requirements.txt
# └── README.md