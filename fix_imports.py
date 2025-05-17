#!/usr/bin/env python3
"""
Quick script to fix imports in all Python files in the app directory
"""

import os

def fix_imports(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Fix direct imports
    replacements = [
        ('from config import', 'from app.config import'),
        ('from utils import', 'from app.utils import'),
        ('from models import', 'from app.models import'),
        ('from schemas import', 'from app.schemas import'),
        ('from services import', 'from app.services import'),
        ('from api import', 'from app.api import'),
        ('from dependencies import', 'from app.dependencies import'),
        ('from db import', 'from app.db import'),
        ('import config', 'import app.config'),
        ('import utils', 'import app.utils'),
        ('import models', 'import app.models'),
        ('import schemas', 'import app.schemas'),
        ('import services', 'import app.services'),
        ('import api', 'import app.api'),
        ('import dependencies', 'import app.dependencies'),
        ('import db', 'import app.db'),
    ]
    
    original_content = content
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    if content != original_content:
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"Fixed imports in {file_path}")
        return True
    else:
        print(f"No changes needed in {file_path}")
        return False

def process_directory(directory):
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_imports(file_path):
                    count += 1
    
    return count

if __name__ == "__main__":
    count = process_directory('app')
    print(f"Fixed imports in {count} files.")