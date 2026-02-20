from logging_config import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
Database Drift Detection
========================

Detects discrepancies between SQLAlchemy models and actual database schema.
Prevents architectural drift by alerting when models and DB are out of sync.

Usage:
    python detect_drift.py [--fix]

Exit codes:
    0 - No drift detected
    1 - Drift detected
    2 - Connection error
"""
import os
import sys
import argparse
from typing import List, Tuple
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

def get_expected_tables() -> List[str]:
    """Extract table names from models.py"""
    models_path = os.path.join(os.path.dirname(__file__), 'models.py')
    tables = []
    
    with open(models_path, 'r') as f:
        content = f.read()
    
    # Extract __tablename__ assignments
    import re
    pattern = r'__tablename__\s*=\s*["\']([^"\']+)["\']'
    tables = re.findall(pattern, content)
    
    return sorted(tables)

def get_actual_tables(db_url: str) -> List[str]:
    """Get actual table names from database"""
    engine = create_engine(db_url)
    inspector = inspect(engine)
    return sorted(inspector.get_table_names())

def detect_drift(db_url: str) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Detect drift between models and database
    
    Returns:
        (has_drift, missing_in_db, missing_in_models, all_tables)
    """
    try:
        expected = set(get_expected_tables())
        actual = set(get_actual_tables(db_url))
        
        missing_in_db = sorted(expected - actual)  # Models expect these but DB doesn't have them
        missing_in_models = sorted(actual - expected)  # DB has these but models don't know about them
        all_tables = sorted(expected | actual)
        
        has_drift = bool(missing_in_db or missing_in_models)
        
        return has_drift, missing_in_db, missing_in_models, all_tables
        
    except SQLAlchemyError as e:
        logger.info(f"❌ Database connection error: {e}")
        sys.exit(2)

def main():
    parser = argparse.ArgumentParser(description='Detect database drift')
    parser.add_argument('--fix', action='store_true', help='Show fix commands')
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("DATABASE DRIFT DETECTION")
    logger.info("=" * 70)
    
    # Get database URL
    db_url = os.getenv('DATABASE_URL', '').replace('+asyncpg', '')
    if not db_url:
        # Try to construct from components
        db_url = "postgresql://ns_admin:ns_password@ns_postgres:5432/ns_core_db"
    
    logger.info(f"\n🔗 Connecting to: {db_url.split('@')[-1] if '@' in db_url else 'localhost'}")
    
    # Detect drift
    has_drift, missing_in_db, missing_in_models, all_tables = detect_drift(db_url)
    
    # Print results
    logger.info(f"\n📊 Total tables: {len(all_tables)}")
    logger.info(f"   • Expected (from models): {len(get_expected_tables())}")
    logger.info(f"   • Actual (in database): {len(get_actual_tables(db_url))}")
    
    if missing_in_db:
        logger.info(f"\n❌ MISSING IN DATABASE ({len(missing_in_db)}):")
        for table in missing_in_db:
            logger.info(f"   • {table}")
        if args.fix:
            logger.info("\n🔧 To fix, run migrations:")
            logger.info("   docker exec -i ns_postgres psql -U ns_admin -d ns_core_db < migrations/<file>.sql")
    
    if missing_in_models:
        logger.info(f"\n⚠️  MISSING IN MODELS ({len(missing_in_models)}):")
        for table in missing_in_models:
            logger.info(f"   • {table}")
        if args.fix:
            logger.info("\n🔧 To fix, add model class to models.py")
    
    if not has_drift:
        logger.info("\n✅ NO DRIFT DETECTED - Models and database are in sync")
    
    logger.info("\n" + "=" * 70)
    
    return 1 if has_drift else 0

if __name__ == "__main__":
    sys.exit(main())
