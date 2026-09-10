import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine import MiningEntityExtractor, MineSafetyRiskAnalyzer, MiningDocumentChunker

def test_mining_entity_extractor():
    extractor = MiningEntityExtractor()
    query = "What causes shuttle car crush injuries during underground pillar extraction under CMR 2017?"
    entities = extractor.extract_entities(query)
    
    assert entities["equipment"] == "shuttle car"
    assert entities["hazard"] == "crush injury"
    assert entities["regulation"] == "CMR 2017"
    assert entities["mine_type"] == "Underground Mining"

def test_risk_analyzer_calculation():
    analyzer = MineSafetyRiskAnalyzer(reports_dir="./data/msha_reports")
    risk = analyzer.calculate_hazard_risk("shuttle car crush injuries")
    
    assert "prob_accident_occurrence" in risk
    assert "prob_fatality_given_accident" in risk
    assert risk["prob_accident_occurrence"] > 0.0
    assert risk["prob_fatality_given_accident"] > 0.0

def test_document_chunker(tmp_path):
    chunker = MiningDocumentChunker(chunk_size=100, overlap=10)
    sample_file = tmp_path / "test_report.txt"
    sample_file.write_text("Coal Mines Regulations 2017 Regulation 143 requires mandatory CO telemetry monitoring. " * 20)
    
    chunks = chunker.process_file(str(sample_file))
    assert len(chunks) > 0
    assert "doc_title" in chunks[0]["metadata"]
    assert "tenant" in chunks[0]["metadata"]
