import pytest
from evaluation.eval_grounding import (
    extract_query_keywords,
    evaluate_passage_relevance,
    extract_claims,
    evaluate_offline_grounding,
)


def test_extract_query_keywords():
    query = "What caused the haul truck to overturn on the bench edge?"
    entities = {"equipment": "haul truck", "hazard": "bench edge overturn"}
    terms = extract_query_keywords(query, entities)
    assert "haul" in terms
    assert "truck" in terms
    assert "overturn" in terms
    assert "the" not in terms
    assert "what" not in terms


def test_evaluate_passage_relevance():
    query = "dumper slope overturn tipping bench"
    entities = {"equipment": "dumper", "hazard": "slope overturn"}
    chunks = [
        {"content": "Opencast dumper overturn at tipping bench caused by missing berm."},
        {"content": "Dumper slope stability failure at waste dump tipping point."},
        {"content": "Continuous miner cutter head maintenance in underground coal seam."},
    ]
    precision, flags = evaluate_passage_relevance(query, chunks, entities)
    assert flags[0] is True
    assert flags[1] is True
    assert flags[2] is False
    assert precision == pytest.approx(2 / 3)


def test_extract_claims():
    text = (
        "Analysis reveals high risk of spontaneous combustion in coal pillars. "
        "CMR 2017 Regulation 143 mandates carbon monoxide telemetry monitoring.\n\n"
        "Citations:\n• [Source 1: DGMS Safety Guidelines]"
    )
    claims = extract_claims(text)
    assert len(claims) == 2
    assert "spontaneous combustion" in claims[0]
    assert "Regulation 143" in claims[1]
    assert not any("Citations:" in c for c in claims)


def test_extract_claims_includes_statistics():
    text = (
        "Based on dynamic statistical hazard analysis across 38 incident reports, "
        "the calculated accident occurrence probability is 13.16% with a fatality probability of 60.0%.\n\n"
        "Citations:\n• [Source 1: MSHA Surface Rules]"
    )
    claims = extract_claims(text)
    assert len(claims) == 1
    assert "13.16%" in claims[0]


def test_evaluate_offline_grounding_supported():
    context_chunks = [
        {"content": "CMR 2017 Regulation 143 requires continuous telemetry monitoring of carbon monoxide in underground coal mines."}
    ]
    response = (
        "Underground coal mines require continuous telemetry monitoring of carbon monoxide. "
        "Compliance with CMR 2017 Regulation 143 is mandatory.\n\n"
        "Citations:\n• [Source 1: CMR 2017]"
    )
    res = evaluate_offline_grounding(response, [{"source_id": 1, "doc_title": "CMR 2017"}], context_chunks)
    assert res["citation_integrity"] == 1.0
    assert res["citation_grounding"] >= 0.90
    assert res["claims_grounded"] >= 1


def test_evaluate_offline_grounding_unsupported():
    context_chunks = [
        {"content": "Safe blasting practices require clearing the danger zone."}
    ]
    response = (
        "Space rocket avionics failed during atmospheric reentry over lunar orbit.\n\n"
        "Citations:\n• [Source 1: NASA Report]"
    )
    res = evaluate_offline_grounding(response, [{"source_id": 1, "doc_title": "NASA"}], context_chunks)
    assert res["claims_grounded"] == 0
    assert len(res["unsupported_claims"]) > 0
