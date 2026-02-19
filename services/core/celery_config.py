"""
Celery Configuration - Shared app instance
"""
import os
from celery import Celery

celery_app = Celery("ns", broker=os.getenv("CELERY_BROKER_URL"))
celery_app.conf.task_routes = {
    'tasks.*': {'queue': 'default'},
    'goal_executor.*': {'queue': 'default'}
}
