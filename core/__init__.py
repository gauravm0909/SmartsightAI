"""
SmartSight AI - Core Analytics & Intelligence Engine
Tagline: "Upload. Analyze. Visualize. Predict."
"""

from core.data_loader import DataLoader
from core.data_cleaner import DataCleaner
from core.data_profiler import DataProfiler
from core.data_analyzer import DataAnalyzer
from core.visualization import Visualizer
from core.ai_insights import AIInsightsEngine
from core.natural_language_query import NaturalLanguageQueryEngine
from core.ml_models import MLPredictiveEngine
from core.report_generator import ReportGenerator

__version__ = "1.0.0"
__all__ = [
    "DataLoader",
    "DataCleaner",
    "DataProfiler",
    "DataAnalyzer",
    "Visualizer",
    "AIInsightsEngine",
    "NaturalLanguageQueryEngine",
    "MLPredictiveEngine",
    "ReportGenerator",
]
