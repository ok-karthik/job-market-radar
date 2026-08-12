import pytest
import pandas as pd
from scraper.filter_jobs import (
    clean_ui_artifacts,
    is_german_nlp,
    requires_german,
    is_contract_or_part_time,
    is_target_location,
    deduplicate_cross_posts,
)


def test_deduplicate_cross_posts_prefers_berlin():
    """Same (title, company) posted to multiple cities collapses to ONE row,
    keeping the Berlin listing regardless of its original position."""
    df = pd.DataFrame([
        {"id": "1", "title": "Site Reliability Engineer", "companyName": "Helsing",
         "location": "Munich, Bavaria, Germany"},
        {"id": "2", "title": "Site Reliability Engineer", "companyName": "Helsing",
         "location": "Berlin, Berlin, Germany"},
        {"id": "3", "title": "Site Reliability Engineer", "companyName": "Helsing",
         "location": "Frankfurt, Germany"},
    ])
    out = deduplicate_cross_posts(df)
    assert len(out) == 1
    assert out.iloc[0]["id"] == "2"  # the Berlin row


def test_deduplicate_cross_posts_deterministic_without_berlin():
    """With no Berlin listing, a German-located row is preferred over EU-remote,
    and the choice is deterministic (first-in-order among equals), not random."""
    df = pd.DataFrame([
        {"id": "1", "title": "Cloud Engineer", "companyName": "SAP",
         "location": "London, United Kingdom"},
        {"id": "2", "title": "Cloud Engineer", "companyName": "SAP",
         "location": "Munich, Germany"},
        {"id": "3", "title": "Cloud Engineer", "companyName": "SAP",
         "location": "Hamburg, Germany"},
    ])
    out = deduplicate_cross_posts(df)
    assert len(out) == 1
    assert out.iloc[0]["id"] == "2"  # first German row, EU-remote deprioritised


def test_deduplicate_cross_posts_similarity_guard_keeps_different_descriptions():
    """Same (title, company) but genuinely DIFFERENT descriptions must NOT be
    merged (the similarity guard); near-identical descriptions still collapse,
    keeping Berlin."""
    df = pd.DataFrame([
        # Two "Cloud Engineer @ BigCorp" that are actually different teams/roles.
        {"id": "1", "title": "Cloud Engineer", "companyName": "BigCorp",
         "location": "Munich, Germany",
         "descriptionText": "Manage AWS EKS clusters, Terraform IaC and GitOps pipelines for the platform team."},
        {"id": "2", "title": "Cloud Engineer", "companyName": "BigCorp",
         "location": "Berlin, Germany",
         "descriptionText": "Build data warehousing on Snowflake, dbt and Airflow for the analytics org."},
        # A true cross-post: identical description in two cities.
        {"id": "3", "title": "SRE", "companyName": "Helsing",
         "location": "Munich, Germany",
         "descriptionText": "Design on-premise Kubernetes, Prometheus and Grafana observability for defence AI."},
        {"id": "4", "title": "SRE", "companyName": "Helsing",
         "location": "Berlin, Germany",
         "descriptionText": "Design on-premise Kubernetes, Prometheus and Grafana observability for defence AI."},
    ])
    out = deduplicate_cross_posts(df)
    ids = set(out["id"])
    # Both distinct Cloud Engineer roles kept; the SRE cross-post collapses to Berlin.
    assert ids == {"1", "2", "4"}


def test_deduplicate_cross_posts_threshold_zero_ignores_descriptions():
    """threshold=0 falls back to pure title+company collapsing regardless of text."""
    df = pd.DataFrame([
        {"id": "1", "title": "Cloud Engineer", "companyName": "BigCorp",
         "location": "Munich, Germany", "descriptionText": "totally different text A"},
        {"id": "2", "title": "Cloud Engineer", "companyName": "BigCorp",
         "location": "Berlin, Germany", "descriptionText": "completely unrelated words B"},
    ])
    out = deduplicate_cross_posts(df, desc_similarity_threshold=0)
    assert len(out) == 1
    assert out.iloc[0]["id"] == "2"  # Berlin kept


def test_deduplicate_cross_posts_keeps_distinct_and_na():
    """Different roles are kept; rows with N/A title/company never collapse."""
    df = pd.DataFrame([
        {"id": "1", "title": "DevOps Engineer", "companyName": "Acme",
         "location": "Berlin, Germany"},
        {"id": "2", "title": "Platform Engineer", "companyName": "Acme",
         "location": "Berlin, Germany"},
        {"id": "3", "title": "N/A", "companyName": "N/A", "location": "Berlin, Germany"},
        {"id": "4", "title": "N/A", "companyName": "N/A", "location": "Munich, Germany"},
    ])
    out = deduplicate_cross_posts(df)
    # 2 distinct real roles + 2 non-collapsed N/A rows
    assert len(out) == 4
    assert set(out["id"]) == {"1", "2", "3", "4"}

def test_clean_ui_artifacts():
    # Test removal of "Show more"
    dirty_text_1 = "Some text here\n\nShow more\n\nOther text"
    assert clean_ui_artifacts(dirty_text_1) == "Some text here\nOther text"
    
    # Test removal of "Show less"
    dirty_text_2 = "Text\n  SHOW LESS  \nText"
    assert clean_ui_artifacts(dirty_text_2) == "Text\nText"
    
    # Test blank inputs
    assert clean_ui_artifacts("") == ""
    assert clean_ui_artifacts(None) is None


def test_is_german_nlp():
    german_text = "Wir suchen einen erfahrenen Softwareentwickler für unser Team in München."
    english_text = "We are looking for an experienced software developer for our team in Munich. This is an exciting opportunity."
    
    assert is_german_nlp(german_text) is True
    assert is_german_nlp(english_text) is False
    
    # Text too short should return False
    assert is_german_nlp("Zu kurz") is False
    
    # None should return False
    assert is_german_nlp(None) is False


def test_requires_german():
    # Test required patterns
    assert requires_german("You need fluent german for this role") is True
    assert requires_german("German level B2 is required") is True
    assert requires_german("Gute Deutschkenntnisse in Wort und Schrift") is True
    
    # Test optional patterns (should override required if both exist)
    assert requires_german("Basic German is a plus") is False
    assert requires_german("Fluent English required. German is nice to have") is False
    
    # Pure English
    assert requires_german("Looking for Python developers") is False


def test_is_contract_or_part_time():
    # Test employment type
    assert is_contract_or_part_time({"employmentType": "Contract"}) is True
    assert is_contract_or_part_time({"employmentType": "Part-time"}) is True
    assert is_contract_or_part_time({"employmentType": "Internship"}) is True
    
    # Test description text for contract indicators
    assert is_contract_or_part_time({"employmentType": "Full-time", "descriptionText": "Type: Contract, duration 6 months"}) is True
    assert is_contract_or_part_time({"employmentType": "Full-time", "descriptionText": "Hourly rate: $50/hour"}) is True
    
    # Test standard full time
    assert is_contract_or_part_time({"employmentType": "Full-time", "descriptionText": "Join our permanent team"}) is False


def test_is_target_location():
    # True cases
    assert is_target_location({"location": "Berlin, Germany"}) is True
    assert is_target_location({"location": "Munich"}) is True
    assert is_target_location({"location": "Frankfurt, DE"}) is True
    assert is_target_location({"location": "Deutschland"}) is True
    
    # False cases (US States)
    assert is_target_location({"location": "San Francisco, CA"}) is False
    assert is_target_location({"location": "New York, NY"}) is False
    assert is_target_location({"location": "Seattle, WA"}) is False
    
    # Edge case: 'de' state (Delaware) vs 'de' country (Germany)
    # The script uses ', de' to mean Germany, but if it has a US state marker it might fail
    # Unless it specifically says "germany"
    assert is_target_location({"location": "Wilmington, DE, United States"}) is False
    assert is_target_location({"location": "Berlin, de"}) is True

    # EU/UK Remote cases
    assert is_target_location({"location": "London, United Kingdom", "descriptionText": "Fully remote role"}) is True
    assert is_target_location({"location": "Europe", "descriptionText": "Work from anywhere in EU"}) is True
    assert is_target_location({"location": "Amsterdam, Netherlands", "descriptionText": "Remote position"}) is True
    
    # False cases for EU/UK (not remote)
    assert is_target_location({"location": "London, United Kingdom", "descriptionText": "On-site 5 days a week"}) is False
    assert is_target_location({"location": "Paris, France"}) is False


# ---------------------------------------------------------------------------
# requires_german — rewritten 2026-08-09 (17 of 163 jobs in one run leaked through)
# ---------------------------------------------------------------------------
# Every string below is lifted verbatim from a real posting in jobs_output/.

@pytest.mark.parametrize("text", [
    "German from C1 for confident client communication",
    "Note: German and English are both mandatory requirements for this role.",
    "Fluent business English and German is required",
    "Excellent oral and written communication skills in both German and English (C1/C2)",
    "very good English (min. C1 level) and German (min. B2 level) spoken and written",
    "Native-level German proficiency is required for this role.",
    "Languages:  German & English at C1",
    "Mandatory: German language skills at a minimum C1 level",
    "Proficient in German - B2",
    "Senior Azure Hybrid Infrastructure Engineer -(German-speaking)",
    "Gute Deutschkenntnisse in Wort und Schrift",
    "Verhandlungssichere Deutschkenntnisse",
])
def test_requires_german_catches_real_requirements(text):
    assert requires_german(text) is True, text


@pytest.mark.parametrize("text", [
    "German is a plus, but not required.",
    "German is nice to have but not mandatory",
    "German is beneficial but not mandatory.",
    "fluent in either German or Dutch is a nice to have",
    "Basic German is a plus",
    "Deutschkenntnisse von Vorteil",
    # 'Germany' the country is not a language requirement -- the trailing 'y'
    # defeats the \bgerman\b word boundary.
    "Right to work in Germany without employer visa sponsorship is required.",
    "This role is based in Berlin, Germany and English is the working language.",
    "Looking for Python developers",
])
def test_requires_german_ignores_optional_and_country_mentions(text):
    assert requires_german(text) is False, text


def test_optional_mention_cannot_cancel_a_requirement_elsewhere():
    """The original bug: the optional check was document-wide, so one
    'German is a plus' anywhere cancelled a hard C1 requirement in another
    bullet. Modality belongs to the NEAREST mention, not to the document."""
    jd = ("• Mandatory: German language skills at a minimum C1 level\n"
          "• French is a plus\n"
          "• Knowledge of Dutch is nice to have")
    assert requires_german(jd) is True
    # ...and the converse still holds: a genuinely optional-only posting passes.
    jd2 = ("• Fluent English required\n"
           "• German is a plus but not mandatory\n"
           "• Willingness to learn German")
    assert requires_german(jd2) is False
