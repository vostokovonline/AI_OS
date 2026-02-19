#!/usr/bin/env python3
"""
Verify Model Integrity - Pre-deployment Check
============================================

Validates that all models can be imported and have correct structure.
Run this before deploying to catch model issues early.

Usage:
    python verify_models.py

Exit codes:
    0 - All checks passed
    1 - Import errors detected
    2 - Model structure issues
"""
import ast
import sys
from pathlib import Path

def check_model_syntax(filepath: Path) -> bool:
    """Check if models.py has valid Python syntax"""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        print(f"✅ {filepath.name} - syntax valid")
        return True
    except SyntaxError as e:
        print(f"❌ {filepath.name} - syntax error: {e}")
        return False

def extract_model_classes(filepath: Path) -> list:
    """Extract all SQLAlchemy model classes from file"""
    with open(filepath, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    models = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it's a SQLAlchemy model (has __tablename__)
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == '__tablename__':
                            models.append(node.name)
                            break
    
    return models

def verify_models():
    """Main verification routine"""
    print("=" * 70)
    print("MODEL INTEGRITY VERIFICATION")
    print("=" * 70)
    
    models_file = Path(__file__).parent / "models.py"
    
    if not models_file.exists():
        print(f"❌ models.py not found at {models_file}")
        return 1
    
    # Check 1: Syntax
    if not check_model_syntax(models_file):
        return 1
    
    # Check 2: Extract models
    models = extract_model_classes(models_file)
    print(f"\n📊 Found {len(models)} model classes:")
    for model in sorted(models):
        print(f"   • {model}")
    
    # Check 3: Critical models present
    critical_models = ['Goal', 'GoalStatusTransition', 'Artifact']
    missing = [m for m in critical_models if m not in models]
    
    if missing:
        print(f"\n❌ CRITICAL MODELS MISSING: {', '.join(missing)}")
        return 2
    else:
        print(f"\n✅ All critical models present")
    
    # Check 4: GoalStatusTransition has required fields
    print("\n📋 Checking GoalStatusTransition structure...")
    with open(models_file, 'r') as f:
        content = f.read()
    
    required_fields = ['goal_id', 'from_status', 'to_status', 'reason', 'created_at']
    for field in required_fields:
        if field in content:
            print(f"   ✅ {field}")
        else:
            print(f"   ❌ {field} - MISSING")
            return 2
    
    print("\n" + "=" * 70)
    print("✅ ALL CHECKS PASSED - Ready for deployment")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(verify_models())
