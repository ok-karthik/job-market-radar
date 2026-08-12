import json
import os
import re
import sys
import time
import random
import hashlib
import argparse
from collections import Counter
from typing import Dict, List, Set, Tuple

import httpx
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
# ─────────────────────────────────────────────────────────────────────────────
# GLiNER: DISABLED BY DEFAULT since 2026-08-09. Set SEMANTIC_ENABLE_GLINER=1 to
# turn it back on; all the code below is intact and one env var away.
#
# WHY, measured over 1,500 real jobs from jobs_output/:
#   * GLiNER contributed **0 of 6,581** canonical skill tags that the regex
#     taxonomy would not have found on its own (0.00%). Re-running
#     TECH_SKILLS_PATTERNS over the same text actually finds MORE (6,781),
#     because the taxonomy has grown since those files were scored. So GLiNER's
#     contribution to TechSkills -- the field that actually drives FitScore,
#     CVSkillOverlap and every ROI table -- is entirely redundant.
#   * Its only unique output was EmergingSkills, and that is a radar whose
#     TOP entry appears in ~2% of postings, which the rendered table itself
#     labels "directional, not significance-tested". Against Kubernetes at
#     30.4% REQUIRED, nothing on that list can change a study decision.
#   * It is the dominant per-job cost of the whole pipeline (see _select_device:
#     the CPU-vs-MPS benchmark exists precisely because this workload is
#     GLiNER-bound).
#
# Turning it off keeps every skill number identical and makes the run
# dramatically faster. The suppression machinery it needed (SOFT_SKILL_BLOCKLIST,
# ESTABLISHED_NON_NOVEL, the relevance gate / self-mention drop / concentration
# guard in update_learning_plan.py) is retained -- it costs nothing when the
# emerging list is empty, and is required again the moment this is re-enabled.
ENABLE_GLINER = os.environ.get("SEMANTIC_ENABLE_GLINER", "0") not in ("0", "", "false", "False")

# Imported lazily inside __init__ so the dependency is not paid for when disabled.

try:
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from scraper.filter_jobs import is_german_nlp, requires_german
except ImportError as e:
    print(f"Warning: Failed to import filter_jobs ({e})")
    pass

# Remote source — always downloaded fresh at runtime so CV changes are picked up automatically.
# Personal values now live in a gitignored config.json — see user_config.py.
from . import user_config as _cfg  # noqa: E402
CV_URL = _cfg.CV_URL
# Local path where the downloaded PDF is written (gitignored).
CV_LOCAL_PATH = "CV.pdf"

# Per-job score cache (see SemanticJobAnalyzer.process_jobs). Lives under the
# gitignored jobs_output/ so recurring jobs across nightly runs skip the
# expensive GLiNER + embedding work. Bump SCORE_CACHE_VERSION to force a rebuild.
DEFAULT_SCORE_CACHE = os.path.join("jobs_output", ".score_cache.json")
SCORE_CACHE_MAX = 12000       # cap entries; oldest-by-touch pruned beyond this
SCORE_CACHE_VERSION = 1


def _download_cv(url: str = CV_URL, local_path: str = CV_LOCAL_PATH,
                 max_attempts: int = 4) -> str:
    """Download the CV PDF from *url* and write it to *local_path*.

    Always overwrites the local file so any edits made to the Google Docs
    source are reflected on every pipeline run without manual cleanup.
    Returns the local file path.

    Resilience: the download is retried with full-jitter capped exponential
    backoff (consistent with the scraper's retry philosophy) so a transient
    network blip doesn't take down the whole nightly pipeline. If every attempt
    fails but a previously-downloaded CV already exists on disk, we fall back to
    that stale copy with a loud warning — a day-old CV is far better than a
    crashed run (the semantic stage runs with check=True, so a hard failure here
    also kills the learning-plan + Notion steps downstream). Only when there is
    no cached copy to fall back to do we re-raise.
    """
    print(f"[*] Downloading CV from Google Docs...")
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                response = client.get(url)
                response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"[+] CV saved to '{local_path}' ({len(response.content):,} bytes).")
            return local_path
        except Exception as exc:  # network timeouts, DNS failures, HTTP errors
            last_exc = exc
            if attempt < max_attempts - 1:
                cap = min(60, 2 ** attempt * 10)
                backoff = random.uniform(cap / 2, cap)
                print(f"    [!] CV download failed ({exc!r}). "
                      f"Retrying in {backoff:.1f}s "
                      f"(attempt {attempt + 1}/{max_attempts})...")
                time.sleep(backoff)

    # All attempts exhausted — fall back to a previously cached CV if we have one.
    if os.path.exists(local_path):
        print(f"[!] CV download failed after {max_attempts} attempts "
              f"({last_exc!r}). Falling back to cached '{local_path}'.")
        return local_path
    raise RuntimeError(
        f"CV download failed after {max_attempts} attempts and no cached "
        f"'{local_path}' exists to fall back to."
    ) from last_exc


def _select_device() -> str:
    """Pick the torch device for the embedding + GLiNER models.

    Override with the ``SEMANTIC_DEVICE`` env var (``cpu``/``mps``/``cuda``).

    Default: prefer CUDA, then **CPU on Apple Silicon (mps)**, then CPU. The CPU
    preference over MPS is deliberate and benchmark-backed (2026-07-27): this
    workload is dominated by GLiNER running per-job on short, one-sentence-at-a-
    time inputs, which is dispatch-bound — MPS kernel-launch overhead makes it
    ~2.5x SLOWER than CPU here, while also pegging the GPU to ~100%. CPU is both
    faster and thermally quieter for these small models. (If a future change adds
    large-batch GPU work, re-benchmark and flip the default or set the env var.)
    """
    override = os.environ.get("SEMANTIC_DEVICE", "").strip().lower()
    if override in ("cpu", "mps", "cuda"):
        return override
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# Define categories with highly specific, expanded descriptions for zero-shot text classification
CATEGORY_DEFINITIONS = {
    # Infrastructure & Cloud
    "Platform Engineering": "Internal Developer Platform, Backstage, Kubernetes platform, developer experience, DevEx, self-service infrastructure, internal tooling",
    "Site Reliability Engineering (SRE)": "Observability, incident response, SLI, SLO, Datadog, Prometheus, high availability, performance tuning, on-call",
    "DevOps Engineering": "CI/CD pipelines, release automation, GitOps, Jenkins, GitLab CI, automation, continuous integration",
    "Cloud Engineering": "AWS engineer, Cloud administrator, Azure specialist, cloud operations, cloud migration, infrastructure engineer",
    "Security Engineering (DevSecOps)": "cybersecurity, penetration testing, IAM, network security, application security, cloud security, DevSecOps",
    
    # AI & Data
    "AI Infrastructure": "GPU provisioning, LLM serving, vLLM, Triton, compute clusters, Kubernetes for AI, hardware optimization",
    "MLOps": "Machine learning pipelines, MLflow, model registries, Kubeflow, training infrastructure, feature stores, data science platform",
    "Data Engineering": "data pipelines, data warehousing, big data, apache kafka, spark, sql, etl, dataops",
    "Data Science & ML Engineering": "data scientist, machine learning engineer, model training, NLP, deep learning, PyTorch, predictive modeling",
    
    # Architecture & Leadership
    "AI Solutions Architecture": "AI architect, generative AI solutions, AI sales engineer, ML solutions architect, customer AI architecture, designing AI pipelines for clients",
    "Solutions Architecture": "solutions architect, enterprise architect, technical pre-sales, cloud solutions, designing enterprise systems for clients",
    "Staff / Principal Engineering": "staff engineer, principal engineer, cross-team architecture, technical direction, senior individual contributor",
    # DELIBERATELY BROAD — do not add "software"/"technology" framing here.
    # It looks wrong (~21% of this category is civil/mechanical/HR/supply-chain
    # leadership: "Engineering Director - Pump Solutions", "Global Head of PMO",
    # "Senior Manager HR"), and a narrowed variant was A/B-tested on 2026-07-31
    # against real historical jobs. It made things WORSE on both sides:
    #   - Junk did not get rejected, it RELOCATED. categorize_job() is a
    #     single-label nearest-centroid pick over 17 categories with no "none of
    #     the above" option, so narrowing one definition just pushes those jobs to
    #     the next-nearest centroid — which for infra-flavoured vocabulary is
    #     Cloud/Platform Engineering, i.e. straight into the TARGET tracks.
    #     ("Construction Manager - Offshore Wind" and "Mechanical Engineer" both
    #     landed in Cloud Engineering; "Senior Manager HR" in Staff/Principal.)
    #   - 11 of 20 GENUINE software-leadership roles moved out, including
    #     "Engineering Manager - Platform Engineering / SRE" and "Senior Director
    #     of Engineering, Platform" — the exact rows the Manager Track section is
    #     built from.
    # The broad wording is therefore load-bearing: it acts as a SINK that keeps
    # non-software leadership OUT of NOW/NEXT/LATER. Off-ladder rejection is done
    # where it belongs — by title, at report time, via classify_role() in
    # jobs_analytics/update_learning_plan.py.
    "Engineering Leadership": "engineering manager, VP of Engineering, Director of Engineering, Head of Engineering, Engineering Team Lead, technical lead, managing developers, people management, tech lead, line manager",
    
    # Software Development
    "Backend Engineering": "Backend Engineer, Java, Python backend, Node.js backend, microservices, API development, server-side development, Spring Boot, Django, core backend applications",
    "Frontend & Fullstack Engineering": "Frontend Engineer, React, Angular, Vue, Javascript, CSS, HTML, full stack developer",
    
    # Other
    "Technical Sales & Pre-sales": "sales engineer, pre-sales engineer, technical account manager, vendor software demonstrations, customer engagement, technical workshops, proof-of-concept, rfp, rfi",
    "Product Management": "Product Owner, Product Manager, Non-technical Product Manager, agile product owner, backlog grooming, writing user stories, go-to-market strategy, product roadmap planning"
}

# Robust Tech Skills dictionary with regex patterns with word boundaries.
# This is the CANONICAL TAXONOMY — the reliable backbone of extraction.
# GLiNER adds on top of this, but this dictionary is the source of truth.
TECH_SKILLS_PATTERNS = {
    # ── Cloud Providers ──────────────────────────────────────────────────────────
    "AWS": r"\b(AWS|Amazon Web Services|S3|EC2|RDS|Lambda|ECS|EKS|CloudFormation|CloudWatch|SageMaker|Bedrock|Kinesis|DynamoDB|SNS|SQS|Step Functions|IAM|Route\s*53|CloudFront|VPC|Fargate|ECR|CodePipeline|CodeBuild|CDK|Athena|Glue|EventBridge|MSK|EMR)\b",
    "GCP": r"\b(GCP|Google Cloud|Google Cloud Platform|BigQuery|Cloud Run|GKE|Vertex AI|Pub/Sub|Cloud Functions|Spanner|Anthos|Cloud Composer)\b",
    "Azure": r"\b(Azure|AKS|Azure DevOps|Azure Functions|Cosmos DB|Microsoft Azure|Azure AD|Entra)\b",

    # ── Container & Orchestration ────────────────────────────────────────────────
    "Kubernetes": r"\b(Kubernetes|K8s|EKS|AKS|GKE|kubectl|Kustomize)\b",
    "Docker": r"\b(Docker|Containerization|Containers?|Podman)\b",
    "Helm": r"\b(Helm|Helm Charts?)\b",

    # ── Infrastructure as Code ───────────────────────────────────────────────────
    "Terraform": r"\b(Terraform|OpenTofu|HCL|Terragrunt)\b",
    "Ansible": r"\b(Ansible|Ansible Playbooks?)\b",
    "Puppet": r"\bPuppet\b",
    "Chef": r"\bChef\b",
    "Salt": r"\b(SaltStack|Salt Stack)\b",
    "Pulumi": r"\bPulumi\b",
    "CloudFormation": r"\bCloudFormation\b",

    # ── CI/CD & GitOps ───────────────────────────────────────────────────────────
    "CI/CD": r"\b(CI/CD|CICD|Continuous Integration|Continuous Deployment|Continuous Delivery)\b",
    "GitHub Actions": r"\b(GitHub Actions)\b",
    "GitLab CI": r"\b(GitLab CI|GitLab)\b",
    "Jenkins": r"\bJenkins\b",
    "ArgoCD": r"\b(ArgoCD|Argo\s*CD|Argo\s*Workflows?)\b",
    "GitOps": r"\bGitOps\b",
    "CircleCI": r"\bCircleCI\b",
    # Case-SENSITIVE: caps-only so the English 'flux' ('in flux') doesn't count
    # as the FluxCD GitOps tool.
    "Flux": r"\b(Flux|FluxCD)\b",

    # ── Observability & Monitoring ────────────────────────────────────────────────
    "Prometheus": r"\b(Prometheus|PromQL|Thanos)\b",
    "Grafana": r"\b(Grafana|Grafana Loki|Loki|Mimir|Tempo)\b",
    "Datadog": r"\b(Datadog|Data Dog)\b",
    "OpenTelemetry": r"\b(OpenTelemetry|OTel|OTLP)\b",
    "ELK Stack": r"\b(ELK|Elasticsearch|Logstash|Kibana|OpenSearch)\b",
    "PagerDuty": r"\bPagerDuty\b",
    "Opsgenie": r"\b(Opsgenie|Ops Genie)\b",
    "New Relic": r"\bNew Relic\b",
    "Splunk": r"\bSplunk\b",
    "Dynatrace": r"\bDynatrace\b",

    # ── AI/ML Tools ──────────────────────────────────────────────────────────────
    "LLMs": r"\b(LLMs?|Large Language Models?|GPT[-\s]?\d?|Claude|Anthropic|OpenAI|ChatGPT|Gemini|Mistral|Llama\s*\d?)\b",
    "GenAI": r"\b(GenAI|Generative AI|Gen AI)\b",
    "RAG": r"\b(RAG|Retrieval.Augmented.Generation)\b",
    "AI Agents": r"\b(AI Agents?|Agentic|Agent.?based|Multi.?Agent|Agent Orchestration|LangGraph|CrewAI|AutoGen|Agentic AI)\b",
    "AI Coding Tools": r"\b(Copilot|GitHub Copilot|Cursor|Claude Code|Codex|Cody|Tabnine)\b",
    "LangChain": r"\b(LangChain|LangGraph|LangSmith)\b",
    "MLflow": r"\bMLflow\b",
    "MLOps": r"\bMLOps\b",
    "Kubeflow": r"\bKubeflow\b",
    "vLLM": r"\bvLLM\b",
    "Triton": r"\b(Triton|Triton Inference)\b",
    "Haystack": r"\bHaystack\b",
    # NOTE: 'vector databases?' (optional plural) + 'vector db(s)' — the old
    # singular-only 'vector database' missed the common plural 'vector databases',
    # which then leaked into the emerging radar despite this canonical entry.
    "Vector DBs": r"\b(Qdrant|Pinecone|Weaviate|Milvus|ChromaDB|Chroma|FAISS|pgvector|vector databases?|vector dbs?|vector search|vector store)\b",
    "GPU/CUDA": r"\b(GPU|GPUs|CUDA|NVIDIA|TPU|A100|H100|H200)\b",
    # Case-SENSITIVE (see CASE_SENSITIVE_SKILLS): 'TF' caps-only so Terraform's
    # '.tf' files don't get mis-tagged as PyTorch. Full words keep all casings.
    "PyTorch": r"\b(PyTorch|Pytorch|pytorch|TensorFlow|Tensorflow|tensorflow|TF|JAX)\b",
    "Hugging Face": r"\b(Hugging\s*Face|HuggingFace|Transformers)\b",

    # ── Event Streaming ──────────────────────────────────────────────────────────
    "Kafka": r"\b(Kafka|Confluent|Kafka Streams)\b",
    "RabbitMQ": r"\bRabbitMQ\b",
    "Pulsar": r"\b(Apache Pulsar|Pulsar)\b",

    # ── Databases ────────────────────────────────────────────────────────────────
    "PostgreSQL": r"\b(PostgreSQL|Postgres)\b",
    "MongoDB": r"\b(MongoDB|Mongo)\b",
    "Redis": r"\bRedis\b",
    "DynamoDB": r"\bDynamoDB\b",
    "Snowflake": r"\bSnowflake\b",
    "Databricks": r"\bDatabricks\b",
    "MySQL": r"\bMySQL\b",
    # AWS Redshift — a real, established data warehouse. Promoted to canonical
    # (rather than blocklisted) so it counts as tracked demand instead of leaking
    # into the emerging radar as noise. DuckDB / ClickHouse likewise — concrete
    # analytical DBs that were surfacing as "emerging" noise (2026-07-27).
    "Redshift": r"\bRedshift\b",
    "DuckDB": r"\bDuckDB\b",
    "ClickHouse": r"\b(ClickHouse|Clickhouse)\b",

    # ── Languages ────────────────────────────────────────────────────────────────
    # Case-SENSITIVE: bare English 'go' (go live / on the go) must NOT count as
    # the language; all real casings of Go/Golang are enumerated explicitly.
    "Go": r"\b(Go|Golang|GoLang|GOLANG|golang)\b",
    "Python": r"\b(Python|Python3)\b",
    "Rust": r"\bRust\b",
    "Java": r"\bJava\b",
    # Case-SENSITIVE for the 'TS'/'JS' acronyms; full words keep all casings.
    "TypeScript": r"\b(TypeScript|Typescript|typescript|TS)\b",
    "JavaScript": r"\b(JavaScript|Javascript|javascript|JS)\b",
    "C++": r"\b(C\+\+|CPP)\b",
    "Bash": r"\b(Bash|Shell|Shell Scripting|Zsh)\b",
    "Kotlin": r"\bKotlin\b",
    "Scala": r"\bScala\b",

    # ── Infra & Networking ───────────────────────────────────────────────────────
    "Linux": r"\b(Linux|Ubuntu|Debian|RHEL|CentOS)\b",
    "Backstage": r"\bBackstage\b",
    "Vault": r"\b(Vault|HashiCorp Vault)\b",
    "Istio": r"\b(Istio|Service Mesh|Envoy|Linkerd)\b",
    "Nginx": r"\b(Nginx|NGINX)\b",
    "Consul": r"\bConsul\b",
    "VMware": r"\bVMware\b",
    "Okta": r"\bOkta\b",
    "OpenStack": r"\b(OpenStack|Open Stack)\b",

    # ── Data Engineering ─────────────────────────────────────────────────────────
    "Airflow": r"\b(Airflow|Apache Airflow)\b",
    # Case-SENSITIVE: caps-only so the English 'spark' ('spark innovation')
    # doesn't count as Apache Spark. Real mentions are always capitalized.
    "Spark": r"\b(Spark|Apache Spark|PySpark|pyspark)\b",
    "dbt": r"\bdbt\b",
    "Dagster": r"\bDagster\b",
    "Airbyte": r"\bAirbyte\b",
    "Fivetran": r"\bFivetran\b",

    # ── API & Frameworks ─────────────────────────────────────────────────────────
    "FastAPI": r"\bFastAPI\b",
    "gRPC": r"\bgRPC\b",
    "GraphQL": r"\bGraphQL\b",
    # Case-SENSITIVE: 'REST' caps-only so the English word 'rest' (the rest of
    # the team…) doesn't inflate it. 'RESTful' variants kept.
    "REST": r"\b(REST|RESTful|RESTFUL|Restful)\b",
    "Spring Boot": r"\bSpring Boot\b",
}

# Skills whose tokens collide with common English words (go, rest) or are
# casing-dependent acronyms (JS/TS/TF). These regexes are compiled WITHOUT the
# global IGNORECASE flag so prose doesn't inflate their demand counts. Their
# patterns above enumerate every legitimate casing explicitly, so real mentions
# (including lowercase 'golang'/'pytorch') are still captured.
CASE_SENSITIVE_SKILLS = {"Go", "REST", "JavaScript", "TypeScript", "PyTorch", "Spark", "Flux"}

# Skills that GLiNER frequently extracts but are NOT actionable tech skills.
# These are soft skills, values, generic concepts, or business terms.
SOFT_SKILL_BLOCKLIST = {
    # Soft skills / company values
    "accountability", "acceptance", "inclusion", "fairness", "innovation",
    "diversity", "flexibility", "growth", "ownership", "teamwork",
    "mission", "trust", "autonomy", "continuous learning", "quality",
    "best practices", "standards", "consistency", "robustness",
    "collaboration", "leadership", "mentorship", "transparency",
    "empathy", "integrity", "passion", "curiosity", "creativity",
    "communication", "empowerment", "humility", "resilience",
    "open culture", "flat hierarchies",

    # Generic concepts (not actionable tools)
    "scalability", "reliability", "performance", "automation",
    "architecture", "security", "compliance", "governance",
    "availability", "maintainability", "operational excellence",
    "technology", "concept", "tools", "tooling", "data",
    "hardware", "networking", "latency", "concurrency",
    "process optimization", "code quality", "system design",
    "capacity planning", "orchestration", "metrics", "monitor",
    "power", "accessibility", "compute", "storage", "scalable",
    "requirements", "dependencies", "complexity", "backup",
    "network", "disaster recovery", "robustness",

    # Business / process terms
    "digitalization", "digital transformation", "saas", "fintech",
    "customer success", "crm", "salesforce", "jira", "notion",
    "slack", "excel", "ms office", "mobility", "sustainability",
    "microsoft 365", "analytics", "sustainable mobility",
    "regulatory requirements", "gdpr", "iso 27001",

    # Category / meta labels (not tools)
    "computer science", "software engineering", "platform engineering",
    "data engineering", "data science", "devops", "r&d",
    "artificial intelligence", "machine learning", "ai",
    "robotics", "saas platform", "cloud services",

    # Spoken languages
    "english", "german", "french", "spanish", "dutch",
    "language", "languages", "german language",

    # Generic fillers
    "software", "tool", "platform", "system", "database", "api",
    "framework", "library", "infrastructure", "code", "cloud",
    "innovation", "diversity", "developer tools",

    # Generic architecture / practice concepts (NOT concrete tools) — these were
    # leaking into the "emerging skills" radar as if they were novel technologies.
    "observability", "distributed systems", "distributed system",
    "microservices", "microservice", "microservices architecture",
    "ai tools", "ai tooling", "ai coding tools", "rest api", "rest apis",
    "restful api", "restful apis", "web services", "web service", "apis",
    "cloud computing", "big data", "data pipelines", "data pipeline",
    "event-driven architecture", "high availability", "fault tolerance",
    "version control", "unit testing", "integration testing",
    "cost", "costs", "data quality", "system architecture",
    "software architecture", "data flows", "data flow", "databases",
    "privacy", "data privacy", "ci", "cd", "solution architecture",
    "cloud architecture", "data management", "data governance",
    "data modeling", "data modelling", "business intelligence",

    # More generic concepts / practices / domains + NER junk that were leaking
    # into the emerging radar (2026-07-27). These are practices or fields, not
    # concrete tools — the tools that implement them are already canonical
    # (monitoring→Prometheus/Grafana, IaC→Terraform, secrets management→Vault,
    # event-driven→Kafka, cloud infrastructure→AWS/GCP/Azure).
    "monitoring", "cloud infrastructure", "cloud platform", "cloud platforms",
    "infrastructure as code", "event-driven architectures", "secrets management",
    "performance optimization", "performance optimisation", "continuous improvement",
    "api design", "api development", "data platform", "data platforms",
    "cybersecurity", "cyber security", "computer vision", "efficiency",
    "mathematics", "maths", "statistics", "sdk", "sdks", "routing",
    "erp", "networking concepts", "operating systems", "software development",
    # Language-proficiency levels GLiNER mis-extracts as "tools" (A1/B2/C1 …)
    "a1", "a2", "b1", "b2", "c1", "c2",

    # Second-tier tail noise (2026-07-27): fields/methodologies/soft terms/job
    # titles/formats/hardware words GLiNER emits that are not concrete tools.
    "safety", "start-up mentality", "startup mentality", "design patterns",
    "energy transition", "operational efficiency", "ml", "emerging technologies",
    "iot", "agile", "scrum", "kanban", "data structures", "engineering manager",
    "technical concepts", "algorithms", "reinforcement learning", "deep learning",
    "machine learning", "nlp", "natural language processing", "adoption",
    "deployment", "agents", "usability", "guardrails", "failure modes",
    "engineering standards", "engineering excellence", "technical excellence",
    "auditability", "ai technologies", "ai technology", "cost efficiency",
    "cpu", "memory", "strategy", "testing", "identity", "throughput",
    "developer tooling", "developer experience", "access control", "risk",
    "json", "yaml", "xml", "sensors", "optimization", "optimisation",
    "macbook", "laptop", "slo", "slos", "sli", "slis", "sla", "slas",
    "uptime", "resilience engineering", "chaos engineering", "load balancing",
    "encryption", "authentication", "authorization", "documentation",
    # Third-tier tail (2026-07-27): networking/practice/domain words + generic
    # "X standards/tooling/analytics" phrases GLiNER emits as pseudo-tools.
    "firewalls", "firewall", "logging", "dns", "switches", "network switches",
    "relational databases", "relational database", "data analytics",
    "modern tooling", "technical standards", "coding standards", "secure",
    "stability", "autonomous driving", "ai governance", "load balancers",
    "vpn", "tcp/ip", "proxies", "load balancer", "networking protocols",
    "data warehousing", "data lakes", "data lake", "data warehouse",
    "iac", "siem", "devops practices", "cloud security", "system performance",
    "system reliability", "cloud-native architectures", "cloud-native",
    "security best practices", "integration patterns", "integrations",
    "simplicity", "segmentation", "evaluation", "cameras", "system design",
    "best practice", "microservices architectures",
}

# Established / ubiquitous technologies that are real tools but are the OPPOSITE
# of "emerging" — they should never appear in the novel-skills radar. Unlike the
# soft-skill blocklist (things that are never skills), these ARE legitimate tech;
# they're simply too mainstream to be a signal. Suppressed from the emerging bucket
# only. (If any becomes worth *tracking*, promote it into TECH_SKILLS_PATTERNS.)
ESTABLISHED_NON_NOVEL = {
    # Ubiquitous dev tooling / VCS
    "git", "github", "gitlab", "bitbucket", "svn", "jira", "confluence",
    # Ubiquitous OS / shell
    "linux", "unix", "windows", "macos", "bash", "shell", "powershell",
    # General-purpose languages not on the infra track (frontend/BI/enterprise)
    "javascript", "typescript", "html", "css", "php", "ruby", "c#", ".net",
    "c++", "c", "sql", "nosql", "perl", "matlab", "r",
    # Frontend frameworks (out of scope for platform/SRE)
    "react", "reactjs", "react.js", "angular", "vue", "vue.js", "node.js",
    "nodejs", "next.js", "svelte", "jquery", "bootstrap",
    # BI / analytics / enterprise apps (out of scope)
    "sap", "tableau", "looker", "power bi", "powerbi", "qlik", "excel",
    "salesforce", "servicenow", "sharepoint", "microsoft office",
    # SAP product family — huge in the German market, so its sub-products reach
    # the radar individually even though the bare "sap" entry is blocked.
    "sap s/4hana", "s/4hana", "sap btp", "sap hana", "hana", "abap",
    "sap fiori", "sap ariba", "sap successfactors", "ipaas",
    # JVM/JS build & test tooling — established, and application-layer
    "gradle", "maven", "ant", "jest", "junit", "cypress", "selenium",
    "teamcity", "bamboo", "webpack", "vite", "babel", "eslint",
    # Ambiguous single words / company names GLiNER over-emits
    "google", "microsoft", "amazon", "apple", "meta", "oracle", "ibm",
    "jobgether", "wellhub", "linkedin", "workday", "sap successfactors",
    "facebook", "agoda", "openup",
    # Off-track JS/mobile stacks (out of scope for platform/SRE, like react/node)
    "nestjs", "nest.js", "android", "ios", "flutter", "react native",
    "swift", "swiftui", "objective-c", "xcode", "kotlin multiplatform",
    # Ubiquitous office/IDE suites (not infra tools)
    "google workspace", "g suite", "jetbrains", "intellij", "vs code",
    "visual studio", "eclipse", "outlook",
    # Design / product-craft disciplines, not tools
    "ux", "ui", "ux/ui", "ui/ux", "figma", "sketch", "miro",
    # Ubiquitous data-science libraries (real, but off the platform/infra track)
    "pandas", "numpy", "scipy", "scikit-learn", "sklearn", "matplotlib",
    "sql server", "mssql", "sqlite",
    # Backend web frameworks — real and long-established, but application-layer,
    # not platform/infra. They rode the radar purely on Python/JVM job volume.
    "flask", "django", "fastapi", "rails", "ruby on rails", "laravel",
    "symfony", "express", "express.js", "spring", "spring boot", "asp.net",
    # Employee-benefit / HR / CRM / support SaaS. These appear in the perks and
    # "our stack" sections of German JDs across many employers, so the
    # concentration guard can't catch them, but none is an engineering tool.
    "jobrad", "urban sports club", "wellpass", "hansefit", "qualitrain",
    "givve", "sodexo", "corporate benefits", "nilo health", "likeminded",
    "egym", "personio", "hubspot", "zendesk", "intercom", "freshdesk",
    "mailchimp", "docusign", "greenhouse", "lever", "ashby", "bamboohr",
    # Off-track hardware / robotics / automotive-vertical vocabulary. Reaches
    # the radar via defense and autonomous-driving employers, which are
    # numerous enough to clear the multi-employer bar.
    "lidar", "radar", "ros", "ros2", "can bus", "autosar", "verilog", "vhdl",
    "plc", "scada", "cad", "solidworks", "heat pump", "heat pumps",
    # Wire protocols and data formats — universal plumbing, never a differentiator
    "http", "https", "tcp", "udp", "tcp/ip", "ip", "ssl", "tls", "ssh",
    "json", "yaml", "xml", "csv", "html5", "rest", "soap", "ftp", "smtp",
}

# Salary extraction patterns for EUR-denominated salaries
SALARY_PATTERNS = [
    # 1. Full number range (e.g. 80,000 - 100,000)
    re.compile(r'(?:€|EUR|euro)\s*(?P<min_k>\d{2,3})[,.]?(?P<min_zeros>\d{3})\s*[-–to]+\s*(?:€|EUR|euro)?\s*(?P<max_k>\d{2,3})[,.]?(?P<max_zeros>\d{3})', re.IGNORECASE),
    re.compile(r'(?P<min_k>\d{2,3})[,.]?(?P<min_zeros>\d{3})\s*[-–to]+\s*(?:€|EUR|euro)?\s*(?P<max_k>\d{2,3})[,.]?(?P<max_zeros>\d{3})\s*(?:€|EUR|euro)', re.IGNORECASE),
    
    # 2. k-format range (e.g. 80k - 100k)
    re.compile(r'(?:€|EUR|euro)\s*(?P<min_k>\d{2,3})k\s*[-–to]+\s*(?:€|EUR|euro)?\s*(?P<max_k>\d{2,3})k', re.IGNORECASE),
    re.compile(r'(?P<min_k>\d{2,3})k\s*[-–to]+\s*(?:€|EUR|euro)?\s*(?P<max_k>\d{2,3})k\s*(?:€|EUR|euro)', re.IGNORECASE),
    
    # 3. Hybrid range (e.g. 80 - 100k)
    re.compile(r'(?:€|EUR|euro)\s*(?P<min_k>\d{2,3})\s*[-–to]+\s*(?:€|EUR|euro)?\s*(?P<max_k>\d{2,3})k', re.IGNORECASE),
    re.compile(r'(?P<min_k>\d{2,3})\s*[-–to]+\s*(?:€|EUR|euro)?\s*(?P<max_k>\d{2,3})k\s*(?:€|EUR|euro)', re.IGNORECASE),
    
    # 4. Single full number (e.g. 95,000)
    re.compile(r'(?:€|EUR|euro)\s*(?P<min_k>\d{2,3})[,.]?(?P<min_zeros>\d{3})', re.IGNORECASE),
    re.compile(r'(?P<min_k>\d{2,3})[,.]?(?P<min_zeros>\d{3})\s*(?:€|EUR|euro)', re.IGNORECASE),
    
    # 5. Single k-format (e.g. 95k)
    re.compile(r'(?:€|EUR|euro)\s*(?P<min_k>\d{2,3})k', re.IGNORECASE),
    re.compile(r'(?P<min_k>\d{2,3})k\s*(?:€|EUR|euro)', re.IGNORECASE),
]


# ─────────────────────────────────────────────────────────────────────────────
# Role-family rejection  (added 2026-08-09)
# ─────────────────────────────────────────────────────────────────────────────
# WHY THIS EXISTS, AND WHY IT IS NOT PART OF categorize_job():
#
# `categorize_job()` is a forced-choice nearest-centroid pick over 17 categories
# with no "none of the above". It therefore CANNOT reject — every job gets a
# label, and off-target jobs land in whichever centroid is nearest, which for
# anything infra-flavoured is Platform or Cloud Engineering. Measured 2026-08-09
# over the whole corpus: of the 1,426 jobs in the four NOW categories, **834
# (58%) had no infrastructure term in the title at all** — `Field-Programmable
# Gate Arrays Engineer`, `Senior QA Engineer`, `Chief Technology Officer - B2B
# Software`, `Graduate Software Engineer`, `Senior Value Engineer` (customer
# success), `.NET Software Engineer`. Every theme percentage computed on that
# population was diluted roughly 2x.
#
# Narrowing the category definitions is the obvious fix and it is the WRONG one
# (see .agents/AGENTS.md, do-NOT-do #2): narrowing relocates junk to the
# next-nearest centroid instead of rejecting it, and evicts correct matches.
# Rejection has to be a SEPARATE, explicit signal. That is this function.
#
# DESIGN CONSTRAINTS:
#   * FLAG, NEVER DROP. This writes an additive `RoleFamily` field; no row is
#     removed. The stage-1 title pre-filter drops before hydration and is
#     therefore invisible to every later stage — a false positive there loses a
#     job permanently. Here a mistake is inspectable and reversible, and the raw
#     data is preserved (standing guardrail: never delete jobs_output data).
#   * CONSERVATIVE DEFAULT. Rejection requires positive evidence. Anything
#     genuinely ambiguous returns 'unclear', which downstream treats as keep.
#   * TITLE FIRST, STACK SECOND. Titles are short and high-precision. The
#     description is only consulted when the title is uninformative
#     ("Software Engineer"), because JD boilerplate mentions everything.
#   * CHEAP AND UNCACHED. Pure regex, computed fresh on every run and
#     deliberately kept OUT of the score cache, so improving these patterns
#     takes effect immediately without invalidating expensive GLiNER work.

# Disciplines that are never an infra/platform role, whatever else the title
# says. Matched as ROLE NOUNS rather than as qualifiers, following the lesson
# recorded for `Marketing` in the stage-1 filter: blanket-matching a domain word
# drops good jobs ("Senior Data Engineer - Marketing Platform").
_ROLE_HARD_OFF = {
    # `field[- ]programmable` is spelled out in real titles ("Field-Programmable
    # Gate Arrays Engineer"), so the acronym alone is not enough.
    "hardware/EE": r"\b(fpga|field[- ]programmable|asic|verilog|vhdl|pcb|rtl design|"
                   r"semiconductor|silicon|firmware|analog|photonics|"
                   r"rf ?/? ?microwave|microwave engineer|\brf engineer\b|"
                   r"radio frequency|\brfic\b|integrated circuit)\b",
    "mechanical/civil": r"\b(mechanical|civil|structural|chemical|hvac|acoustic|turbine|hydraulic|welding|piping|surveyor)\s+(engineer|designer|technician)",
    "QA/test": r"\b(qa|quality assurance|quality|test(ing)?|sdet)\s+(engineer|analyst|lead|manager|specialist)\b|\bsdet\b"
               r"|\bengineer in test\b|\btest automation\b",
    "sales/CS": r"\b(sales|account|pre[- ]?sales|presales|value|solutions?\s+consultant|customer success|business development|partner(ships)?|partner development)\s+(engineer|executive|manager|director|lead|representative)\b"
                r"|\bcustomer engineer\b|\bcommercial solutions engineer\b|\bcustomer cloud engineer\b|\bseo\b"
                r"|\bpre[- ]?sales\s+architect\b|\bdesignated support engineer\b"
                r"|\baccount executive\b|\bgo[- ]to[- ]market\b|\bgtm engineer\b|\brevops\b|\brevenue operations\b"
                r"|\bbusiness consultant\b|\bcustomer solutions engineer\b",
    "science/research": r"\b(data scientist|research scientist|bioinformatic|computational biolog|"
                        r"clinical|laborator|chemist|physicist|researcher|quantitative research\w*)\b",
    # Academia. Real titles from the corpus: `Postdoctoral Fellow in Lunar
    # Positioning and Timing`, `Lecturer/Senior Lecturer - Databases`.
    "academia": r"\b(post[- ]?doc\w*|postdoctoral|lecturer|professor|research fellow|"
                r"phd (student|candidate|position)|doctoral|habilitation|teaching|"
                r"(master|bachelor)[' ]?s? thesis|principal investigator|curriculum developer)\b",
    # Hospitality / retail / media. `Bar & Lounge Service Staff` and `Senior
    # Video Producer` were both classified as Platform Engineering.
    "hospitality/media": r"\b(bar|lounge|service staff|barista|chef|waiter|waitress|kitchen|"
                         r"housekeep\w*|receptionist|retail assistant|"
                         r"video producer|content creator|copywriter|videographer|photographer|"
                         r"social media|community manager)\b",
    # Manufacturing / production floor. Matched as a ROLE noun -- bare
    # `manufacturing` would block "Platform Engineer, Manufacturing Systems".
    # NOT "production engineer": that is Meta's name for an SRE and it stays in
    # the infra keep list.
    "manufacturing": r"\b(manufacturing|assembly|foundry|machining|tooling)\s+"
                     r"(engineer|technician|manager|specialist|lead)\b"
                     r"|\b(manufacturing|assembly)\s+\w+\s+(engineer|technician)\b"
                     r"|\b(plating|shop floor|maintenance engineer|quality (control|inspector))\b",
    # Aerospace / power / heavy engineering — DISCIPLINE nouns only.
    #
    # `aviation`, `aircraft` and `airbus` were deliberately REMOVED from this
    # list: they are INDUSTRY verticals, and blocking them rejected
    # `Senior DevOps & Engineering Platform Engineer - Aviation` at FitScore
    # 92.1. Same lesson the stage-1 filter records for `Automotive`. An aircraft
    # role with no infra term in its title still falls through to stack scoring.
    "aerospace/power": r"\b(propulsion|hvdc|substation|switchgear|grid code|transmission line|"
                       r"photovoltaic|perovskite|nuclear|turbine blade|wind farm)\b",
    # PLM / CAD / engineering-application administration. `3DX Platform
    # Administrator` scored on-target purely on the word "Platform".
    "PLM/CAD": r"\b(3dx|3dexperience|catia|dassault|solidworks|autocad|plm|teamcenter|"
               r"siemens nx|ansys|matlab)\b",
    # Analytics / BI. dbt+SQL reporting work, not platform work. `Data Engineer`
    # is deliberately NOT here -- it is genuinely adjacent and some data-platform
    # roles are infra roles.
    "analytics/BI": r"\banalytics engineer\b|\bbusiness intelligence\b|\bbi (developer|analyst|engineer)\b"
                    r"|\bdata analyst\b|\b(tableau|power ?bi|looker) (developer|analyst|consultant)\b",
    # Compiler / toolchain — a distinct discipline that shares no vocabulary with
    # platform work despite both being "systems" jobs.
    "compiler/toolchain": r"\bcompiler (engineer|developer)\b|\b(llvm|mlir)\b|\btoolchain engineer\b",
    # Domain-specific engineering that shares no stack with platform work.
    "specialist domain": r"\bcfd\b|\bcomputational fluid\b|\bcomputer vision engineer\b"
                         r"|\bnavigation engineer\b|\bposition, navigation\b|\bpnt engineer\b"
                         r"|\bcomputational patholog\w*\b|\bbiomarker\b",
    # Quantum computing.
    "quantum": r"\bquantum\b",
    # Functional safety / requirements engineering (systems-engineering
    # disciplines). NOTE `safety engineer`, not bare `safety` -- and
    # `reliability engineer` stays in the infra keep-list, so
    # "Reliability and Safety Engineer" is caught here rather than kept.
    "safety/requirements": r"\b(safety engineer|functional safety|requirements engineer|"
                           r"verification (and validation|engineer)|homologation)\b",
    # Vehicle / embedded software specifics. Matched on the DISCIPLINE
    # (microcontroller, infotainment, RTOS), never on the industry word
    # `automotive` -- blocking that cost a genuine GenAI Solutions Architect.
    "embedded/vehicle sw": r"\b(qnx|rtos|autosar|infotainment|\bivi\b|microcontroller|"
                           r"perception engineer|sensor integration|advanced driving|"
                           r"autonomous driving|in-?vehicle)\b",
    # Chip / silicon architecture.
    "silicon/chip": r"\b(computer architect|chip (architect|design)|neural graphics|"
                    r"soc (design|architect)|\brtl\b)\b",
    # Physical-industry engineering and medical science.
    "energy/medical": r"\b(in-?situ recovery|power and utilities|launch operations|drilling|"
                      r"mining|mri|radiolog\w*|medical (device|imaging))\b",
    # Product engineering = application development at a product company.
    "product eng": r"\bproduct engineer\b|\bmartech\b|\bmulesoft\b",
    # ML research / modelling, as opposed to ML *infrastructure*. Mirrors the
    # role_fit_flag idea in publish_to_notion.py: these score high on vocabulary
    # overlap and sit outside the infra track.
    "ML research": r"\b(applied scientist|algorithm (developer|engineer)|"
                   r"research engineer|deep learning researcher)\b",
    # ITSM / enterprise-tool administration.
    "ITSM/enterprise tools": r"\bbmc helix\b|\bservicenow\b|\bjira admin\w*\b|\batlassian\b|\bconfluence admin\w*\b",
    # Business/marketing-side AI roles -- "AI" in the title, no engineering in
    # the job. Matched as ROLE nouns so "AI Platform Engineer" is untouched.
    "business AI": r"\bai\s+(growth|implementation|search|adoption|transformation|solution)\s+"
                   r"(manager|lead|strategist|specialist|consultant)\b"
                   r"|\bai (specialist|strategist)\b|\bdigital marketing\b"
                   r"|\bcommunications manager\b|\bfounders? associate\b",
    # Industrial plant operations. German titles appear verbatim in these ads.
    "industrial ops": r"\bbetriebsingenieur\w*\b|\bplant engineer\b|\bo&m engineer\b"
                      r"|\bindustrial (it )?engineer\b|\bproject engineer\b|\bprocess engineer\b",
    # Desk-side application support (distinct from the IT-support family below).
    "app support": r"\bapplication support\b|\bit service (engineer|desk)\b|"
                   r"\bbusiness intelligence analyst\b|"
                   r"\btechnical (support|service|consultant)\b|\bsupport (coordinator|specialist)\b|"
                   r"\bauthentication specialist\b",
    # Non-software industries. These arrive because `categorize_job()` is
    # forced-choice: a footwear developer or a bakery R&D manager has no correct
    # label among the 17, so it lands in Product Management or Engineering
    # Leadership. All real titles from the corpus.
    "non-software industry": r"\b(footwear|apparel|textile|bakery|food|beverage|retail|"
                             r"fleet manager|sorting hub|vehicle readiness|"
                             r"new product introduction|product industrialization|"
                             r"transaction assurance|merchandis\w*|fashion)\b",
    # Marketing / content strategy.
    "marketing/content": r"\b(marketing (strategy|engineer|manager)|content strateg\w*|"
                         r"online marketing|product marketing|brand\b|copywriting)\b",
    # Commercial / operations management, not engineering.
    "ops/commercial mgmt": r"\b(betriebsleiter|operations manager|inventory control|fulfillment|"
                           r"delivery station|vendor management|it sourcing|procurement|"
                           r"engagement manager|due diligence|events specialist|content lead|"
                           r"tracking manager|decision scientist|career accelerator|"
                           r"business developer|founder'?s associate|revenue management|"
                           r"project management|application management)\b",
    "product/design": r"\bhead of product\b|\bproduct (manage(r|ment)|owner|designer|analyst|lead(er)?|strategy)\b"
                      r"|\b(ux|ui)\s*(designer|researcher)?\b|\b(scrum master|agile coach|business analyst|project manager|program manager)\b|\bgraphic designer\b",
    "mobile": r"\b(ios|android|mobile|flutter|react native|swift(ui)?)\s+(developer|engineer)\b"
              r"|\b(developer|engineer)\s+(ios|android|flutter)\b",
    # Allows intervening words: `Lead Unity Software Engineer (Gameplay)` and
    # `Lead C++ Software Engineer (Gameplay)` both slipped an adjacency-only
    # version of this pattern.
    "game": r"\b(game|gameplay|unity|unreal|godot)\b[\w\s,&/()+-]{0,24}\b(developer|engineer|designer|programmer|artist)\b"
            r"|\b(developer|engineer|designer|programmer)\b[\w\s,&/()+-]{0,24}\b(gameplay|game studio)\b",
    # `marketing` matches only as a ROLE noun, never as a domain qualifier --
    # the stage-1 filter records that blanket-matching it dropped a genuine
    # "Senior Data Engineer - Marketing Platform".
    "non-tech function": r"\b(recruit(er|ment)|talent|technical writer|hr\b|people|finance|"
                         r"accountant|controlling|legal|procurement|office manager)\b"
                         r"|\bmarketing\s+(manager|specialist|lead|analyst|assistant|intern|director)\b"
                         r"|\bperformance marketing\b|\bstrategy consulting\b|\bmanagement consultant\b",
    # Vehicle/aero/robotics control domains. Matched on DOMAIN nouns, never on
    # the industry word "automotive" -- the stage-1 filter records that blocking
    # the industry cost a genuine "Sr Solutions Architect GenAI, Automotive".
    "embedded/robotics": r"\b(bring[- ]?up engineer|avionics|autopilot|chassis|powertrain|adas|"
                         r"mechatronik|mechatronic|plc|scada|opto[- ]?mechanical|photonic\w*)\b",
    # Desk-side / 1st-2nd level IT. NOT "administrator" on its own -- Kubernetes
    # Administrator is a deliberate Group C scrape keyword and a real target.
    # The tier words MUST be followed by level/line. An earlier version listed
    # `first` as a bare alternative and rejected `Senior Infrastructure Engineer
    # (AI-First)` at FitScore 74.4 -- the single worst false positive found
    # during validation.
    "IT support": r"\bit[- ]?support\b|\bservice desk\b|\bhelp ?desk\b|\bdesktop support\b"
                  r"|\b(1st|2nd|first|second)[- ](level|line)\b",
    # Physical / on-site engineering work.
    "field ops": r"\b(field service|site engineer|commissioning|installation|maintenance technician|"
                 r"warehouse|logistics|localization specialist|"
                 r"facilit(y|ies)|data cent(er|re) (engineering )?operations|\bdceo\b)\b",
    # DBA is its own career track, not platform/SRE.
    "DBA": r"\bdatabase administrator\b|\b(oracle|mysql|postgres\w*|mssql|sql server) (dba|administrator)\b",
}

# A real infrastructure/platform signal anywhere in the title. Overrides the
# SOFT list below outright — the stage-1 filter learned this the hard way:
# an inline negative lookahead only looks FORWARD from the match, so
# "Full-Stack Engineer & Infrastructure Co-Builder" was kept while
# "Infrastructure Engineer / Full-Stack" would have been dropped. A separate
# keep pattern applies to the whole title.
#
# NOTE: bare `systems engineer` and `automation engineer` were REMOVED from this
# list on 2026-08-09 after measurement. They looked like infra terms and are not:
# they matched `Senior Robotic Systems Engineer`, `Senior Avionics System
# Engineer - Electrical Power System`, `Electronic Systems Engineer`, `Embedded
# Systems Engineer (Aerospace)`, `Senior Power Systems Engineer` and `Senior
# Intelligent Automation Engineer - UiPath`. Those titles now fall through to
# stack scoring, which judges them on the description instead.
_ROLE_INFRA_KEEP = re.compile(
    r"\b(platform|infrastructure|infra|sre|site reliability|devops|dev ?sec ?ops|"
    r"cloud|kubernetes|k8s|container|mlops|ml ?ops|gitops|terraform|observability|"
    r"reliability engineer|developer experience|devex|finops|production engineer|"
    # AI *infrastructure* terms -- the user NEXT track. Deliberately the
    # serving/inference side, not application-AI words: `model serving` and
    # `inference` are infra, `AI assistant` and `chatbot` are not. Without these
    # `Member of Technical Staff - Model Serving / API Backend Engineer` was
    # rejected as a backend title.
    r"model serving|inference|ml platform|gpu|llm infrastructure|ai infrastructure)\b",
    re.IGNORECASE,
)

# INDUSTRY verticals, checked AFTER the infra keep-pattern so an infra role in
# that industry survives. `Robotics Software Engineer` is a robotics job;
# `SRE - Robotics Platform` is an SRE job that happens to serve robots, and the
# only thing separating them is which word is the ROLE. Same reason `aviation`
# and `automotive` are absent from the hard list entirely.
_ROLE_INDUSTRY_OFF = {
    "robotics/aerospace industry": re.compile(
        r"\b(robotic\w*|drone|uav|satellite|spacecraft|autonomous vehicle|"
        r"aircraft|aviation|airbus|avionic\w*|space situational|space systems)\b", re.IGNORECASE),
    # Executive / people-management ladder. the user targets the IC Staff track,
    # so VP/Director/Head-of/C-level are off -- but this sits in the INDUSTRY
    # tier (checked after the infra keep-pattern) on purpose, so
    # `Head of AI & Agentic Platform Engineering` still reads as on-target while
    # `Director of Data (BI & Analytics)` and `VP Engineering` do not.
    # Engineering Manager / Team Lead are deliberately ABSENT: those are
    # genuinely borderline and stay 'unclear' rather than being rejected.
    "executive": re.compile(
        r"\b(vp|vice president|director of|head of|chief \w+ officer|\bcto\b|\bceo\b|\bcio\b|"
        r"entrepreneur in residence|founding (cto|ceo))\b", re.IGNORECASE),
    # Graduate / trainee schemes. The stage-1 filter hard-skips Junior/Intern but
    # not "Graduate Program", and these are entry-level by definition.
    "graduate/trainee": re.compile(
        r"\b(graduate (program|programme|scheme)|trainee|apprentice|working student|"
        r"entry[- ]level|new grad)\b", re.IGNORECASE),
}

# Every real job title contains a role noun. Titles that contain none are
# scraper artefacts, not postings -- the corpus holds `Command Palette`,
# `Context.dev`, `Prem Kumar` and `Craft 2026: How to plug AI into your team?`,
# which are page furniture, a person's name and an event, parsed as jobs.
# Checked only AFTER the infra keep-pattern, so an infra title never reaches it.
_ROLE_NOUN = re.compile(
    r"\b(engineer\w*|developer|architect|administrator|admin|manager|lead|"
    r"specialist|consultant|scientist|analyst|director|head|officer|principal|"
    r"staff|technician|expert|designer|programmer|researcher|intern|"
    r"apprentice|associate|coordinator|owner|master|strategist|advisor|"
    r"practitioner|professional|executive|representative|technologist|"
    r"sre|devops|ops|cto|ceo|cio|vp)\b",
    re.IGNORECASE,
)

# Application-development role nouns. NOT rejected on the title alone — a
# backend or full-stack title with a genuinely infra-heavy description is a real
# adjacent match. These go to stack scoring with a raised bar instead.
# NOTE the structure: tokens that START or END with a non-word character cannot
# live inside a `\b(...)\b` wrapper. `\b\.net` needs a word char before the dot
# (there is a space in "Lead .NET Engineer"), and `c\+\+\b` needs a word char
# after the plusses. Both silently never matched until this was split out.
_ROLE_SOFT_OFF = re.compile(
    r"\b(?:front[- ]?end|frontend|full[- ]?stack|fullstack|web developer|"
    r"react|angular|vue|ui engineer|"
    r"back[- ]?end|java|dotnet|php|ruby|rails|node\.?js|"
    r"salesforce|sap|abap|erp|crm|sharepoint|wordpress)\b"
    r"|(?<![\w.])\.net(?![\w])"
    r"|(?<![\w+])c\+\+"
    r"|(?<![\w#])c\#(?![\w#])",
    re.IGNORECASE,
)

# Stack evidence, used ONLY when the title is uninformative. Deliberately narrow
# on the web side: TypeScript/JavaScript are excluded because CDK and Pulumi put
# them in genuine infra JDs, so they do not discriminate.
#
# WIDENED 2026-08-09. The first version omitted the cloud provider names and all
# traditional sysadmin vocabulary, so `Senior Azure Hybrid Infrastructure
# Engineer` scored ZERO infra signal -- a false negative created purely by a gap
# in this list, which then looked like evidence that the scoring rule was unsafe.
# If a term belongs in an infrastructure job ad, it belongs here.
_STACK_INFRA = re.compile(
    r"\b(kubernetes|k8s|eks|aks|gke|openshift|rancher|terraform|opentofu|terragrunt|"
    r"ansible|helm|argo ?cd|flux ?cd|docker|containerd|podman|prometheus|grafana|"
    r"opentelemetry|otel|datadog|dynatrace|splunk|elastic ?search|karpenter|istio|"
    r"linkerd|cilium|vault|packer|puppet|chef|saltstack|nomad|openstack|"
    r"infrastructure as code|\biac\b|ci/?cd|jenkins|gitlab ci|github actions|"
    r"\bvpc\b|\biam\b|\bsre\b|observability|on[- ]call|incident|\bslo\b|\bsli\b|"
    r"load balanc|ingress|service mesh|bare[- ]metal|site reliability|"
    r"platform engineering|cloud infrastructure|"
    # cloud providers -- omitting these was the original bug
    r"\baws\b|amazon web services|\bazure\b|\bgcp\b|google cloud|\bec2\b|\bs3\b|"
    r"lambda|cloudformation|\bcdk\b|bicep|\barm templates?\b|"
    # traditional ops / sysadmin, which is still infrastructure work
    r"linux|unix|windows server|active directory|powershell|bash script|shell script|"
    r"vmware|hyper-?v|virtualiz|virtualis|\bdns\b|\bdhcp\b|firewall|\bvpn\b|"
    r"networking|\btcp/?ip\b|storage|backup|disaster recovery|high availability|"
    r"monitoring|alerting|capacity planning|patch(ing| management)|"
    r"provisioning|configuration management|infrastructure automation)\b",
    re.IGNORECASE,
)
_STACK_WEB = re.compile(
    r"\b(react|angular|vue\.?js|svelte|next\.?js|nuxt|redux|tailwind|"
    r"\bcss\b|\bscss\b|\bsass\b|\bhtml\b|jquery|webpack|vite|storybook|"
    r"responsive design|browser|\bdom\b|front[- ]?end|user interface|"
    r"figma|wordpress|shopify|laravel|symfony|spring boot|asp\.net|xamarin)\b",
    re.IGNORECASE,
)

ROLE_FAMILY_ON = "on-target"
ROLE_FAMILY_OFF = "off-target"
ROLE_FAMILY_UNCLEAR = "unclear"


def classify_role_family(title: str, description: str = "") -> tuple:
    """Return ``(family, reason)`` describing whether a posting is an
    infrastructure/platform role at all.

    ``family`` is one of ``'on-target'`` | ``'off-target'`` | ``'unclear'``.
    ``reason`` is a short machine-readable trace (``'hard:QA/test'``,
    ``'title:infra'``, ``'stack:web 6 vs infra 1'``) so a wrong verdict can be
    audited without re-running anything.

    Precedence, and why:

    1. **Hard off-discipline in the title wins outright.** An FPGA or QA or
       sales role is not an infra role however much cloud vocabulary its
       description carries.
    2. **Then an explicit infra term in the title.** This overrides the soft
       application-development list, so ``Platform Engineer (Full-Stack)``
       survives.
    3. **Then stack evidence from the description**, with the bar raised when
       the title looks like application development.
    4. **Default is 'unclear', which downstream treats as keep.** Rejection
       requires positive evidence; a thin JD is never rejected for being thin.
    """
    t = str(title or "")
    for family, pat in _ROLE_HARD_OFF.items():
        if re.search(pat, t, re.IGNORECASE):
            return ROLE_FAMILY_OFF, f"hard:{family}"

    d = str(description or "")
    infra = len(set(m.group(0).lower() for m in _STACK_INFRA.finditer(d)))
    web = len(set(m.group(0).lower() for m in _STACK_WEB.finditer(d)))

    if _ROLE_INFRA_KEEP.search(t):
        # A description too short to state requirements cannot support an
        # on-target verdict. Six rows in the 2026-08-09 corpus are under 300
        # chars, and they include a plainly fake posting ("About The Role: Some
        # description / Benefits & perks: Fruits! Fruits!") that was sitting in
        # the on-target list at FitScore 0. Downgrade rather than drop — three
        # of the six are real jobs, just terse.
        if len(d) < 300:
            return ROLE_FAMILY_UNCLEAR, f"title:infra but description too short ({len(d)} chars)"
        return ROLE_FAMILY_ON, f"title:infra (stack {infra})"

    for family, rx in _ROLE_INDUSTRY_OFF.items():
        if rx.search(t):
            return ROLE_FAMILY_OFF, f"industry:{family}"

    if not _ROLE_NOUN.search(t):
        return ROLE_FAMILY_OFF, "artefact:no role noun in title"

    if _ROLE_SOFT_OFF.search(t):
        # No stack rescue. An `infra >= 5` escape hatch was tried and REMOVED on
        # 2026-08-09: fintech backend ads name AWS, Kubernetes and CI/CD in
        # passing, so it readmitted eight of them -- `Backend Engineer - Money
        # Transfers`, `Backend Engineer (Golang) - Balance Management`,
        # `Backend Software Engineer - Golang or Java` (infra 13!). A backend JD
        # mentioning Kubernetes is still a backend job. The only escape hatch is
        # an explicit infra term in the TITLE, checked above, which is what
        # keeps `Backend / SRE Engineer` and `Full-Stack Engineer &
        # Infrastructure Co-Builder`.
        return ROLE_FAMILY_OFF, f"soft:app-dev title (infra {infra} / web {web})"

    if web >= 3 and web > infra:
        return ROLE_FAMILY_OFF, f"stack:web {web} vs infra {infra}"
    if infra >= 4 and infra > web:
        return ROLE_FAMILY_ON, f"stack:infra {infra} vs web {web}"
    # NOTE: "zero infra vocabulary => off-target" was implemented on 2026-08-09
    # and REMOVED the same day after validation. It rejected 928 jobs and read as
    # a clean rule, but three genuine targets were among them -- MOIA's
    # `(Senior) Platform Engineer` (DevEx team), GetYourGuide's `Senior ML Ops
    # Engineer, AI Platform Team`, and Cohere's `Forward Deployed Engineer,
    # Agentic Platform`. All three describe the role in prose and never name a
    # tool, so absence of stack vocabulary is NOT evidence of absence of infra
    # work. Those jobs stay 'unclear'; use RoleFamily == 'on-target' when a
    # strict population is needed, rather than trying to reject your way there.
    return ROLE_FAMILY_UNCLEAR, f"stack:infra {infra} / web {web}"


def load_cv_text(cv_source: str) -> str:
    """
    Load CV text from a file path (.txt, .pdf, .docx) or an HTTPS URL.

    Supported sources:
    - Plain text file:  path/to/cv.txt
    - PDF file:         path/to/cv.pdf
    - DOCX file:        path/to/cv.docx
    - Google Docs URL:  https://docs.google.com/document/.../export?format=pdf
    - Any HTTPS URL:    treated as a PDF download
    """
    import io

    if cv_source.startswith("http://") or cv_source.startswith("https://"):
        # --- URL path: fetch and treat response as PDF bytes ---
        print(f"[*] Fetching CV from URL: {cv_source}")
        import httpx as _httpx
        with _httpx.Client(follow_redirects=True, timeout=30.0) as client:
            response = client.get(cv_source)
            response.raise_for_status()
        pdf_bytes = response.content
        import fitz  # pymupdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc).strip()

    ext = cv_source.lower().rsplit(".", 1)[-1] if "." in cv_source else "txt"

    if ext == "pdf":
        import fitz  # pymupdf
        doc = fitz.open(cv_source)
        return "\n".join(page.get_text() for page in doc).strip()

    if ext == "docx":
        from docx import Document
        doc = Document(cv_source)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()

    # Default: plain text
    with open(cv_source, "r", encoding="utf-8") as f:
        return f.read()




class SemanticJobAnalyzer:
    def __init__(self, cv_path: str):
        # BAAI/bge-small-en-v1.5: same ~33M params as MiniLM-L6 but 512-token context
        # (MiniLM-L6-v2 truncates at 256 tokens, silently losing most of a job description)
        MODEL_NAME = "BAAI/bge-small-en-v1.5"
        print(f"[*] Loading embedding model ({MODEL_NAME})...")
        device = _select_device()
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        print(f"[*] Model loaded on {device}.")

        if ENABLE_GLINER:
            print("[*] Loading GLiNER NER model (urchade/gliner_small-v2.1)...")
            import warnings
            from gliner import GLiNER
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*resume_download.*")
                self.gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
            self.gliner_model.to(device)
            print(f"[*] GLiNER loaded on {device}.")
        else:
            self.gliner_model = None
            print("[*] GLiNER disabled (ENABLE_GLINER=0) — regex taxonomy only. "
                  "EmergingSkills will be empty. See the note by ENABLE_GLINER.")

        # Prepare Category embeddings
        self.category_names = list(CATEGORY_DEFINITIONS.keys())
        category_texts = [f"{name} - {desc}" for name, desc in CATEGORY_DEFINITIONS.items()]
        print("[*] Encoding categories...")
        self.category_embeddings = self.model.encode(category_texts, convert_to_tensor=True)

        # Prepare CV embedding — supports .txt / .pdf / .docx / https:// URL
        print(f"[*] Loading CV from: {cv_path}")
        cv_text = load_cv_text(cv_path)
        print(f"[*] CV loaded ({len(cv_text)} chars). Encoding...")
        self.cv_embedding = self.model.encode(cv_text, convert_to_tensor=True)

        # Compile regex patterns. Colliding/ambiguous tokens (Go, REST, JS/TS/TF)
        # are compiled case-SENSITIVELY so English prose doesn't inflate them.
        self.compiled_patterns = {
            skill: re.compile(
                pattern, 0 if skill in CASE_SENSITIVE_SKILLS else re.IGNORECASE
            )
            for skill, pattern in TECH_SKILLS_PATTERNS.items()
        }

        # Fingerprints for the per-job score cache (process_jobs). An entry is
        # reused only when job text + CV + extraction code all match, so a CV
        # edit or taxonomy change transparently invalidates stale entries.
        self._cv_fp = hashlib.sha256(cv_text.encode("utf-8", "ignore")).hexdigest()[:16]
        # ENABLE_GLINER is part of the fingerprint: without it, entries cached
        # while GLiNER was on kept returning their old EmergingSkills lists
        # after it was switched off (132 of 2,064 rows in the 2026-08-09
        # corpus), so the flag appeared not to have taken effect.
        self._code_fp = hashlib.sha256(repr((
            SCORE_CACHE_VERSION, ENABLE_GLINER, sorted(TECH_SKILLS_PATTERNS),
            sorted(SOFT_SKILL_BLOCKLIST), sorted(ESTABLISHED_NON_NOVEL),
        )).encode()).hexdigest()[:16]

        # The CV's own canonical tools (regex-only, model-free) — powers the
        # transparent skill-overlap component of the fit score (see _apply_fit).
        self.cv_skills = set(self.extract_tech_skills(cv_text, fast_mode=True))

    @staticmethod
    def _job_fp(title: str, desc: str) -> str:
        return hashlib.sha256(f"{title}\x00{desc}".encode("utf-8", "ignore")).hexdigest()[:16]

    def _canonicalize_skill(self, text: str) -> str | None:
        """Return the canonical taxonomy key if *text* matches a known skill.

        Lets a GLiNER free-text hit (e.g. 'k8s', 'kubernetes cluster') fold onto
        its canonical name ('Kubernetes') instead of fragmenting the counts.
        """
        for name, pattern in self.compiled_patterns.items():
            if pattern.search(text):
                return name
        return None

    def extract_tech_skills(
        self, text: str, fast_mode: bool = False, with_emerging: bool = False
    ):
        """Extract tech skills using robust regex word boundaries and zero-shot NER.

        Returns a canonical ``List[str]`` by default. GLiNER hits that match the
        canonical taxonomy are folded onto their canonical name; genuinely novel
        hits (beyond the taxonomy) are kept separately and only returned when
        ``with_emerging=True`` (then the return is ``(canonical, emerging)``), so
        NER noise never pollutes the canonical demand counts.
        """
        if not text:
            return ([], []) if with_emerging else []

        skills = []

        # 1. Regex Extractions (Core predefined tools)
        for name, pattern in self.compiled_patterns.items():
            if pattern.search(text):
                skills.append(name)

        if fast_mode or not ENABLE_GLINER:
            return (skills, []) if with_emerging else skills

        # 2. GLiNER dynamic extractions
        # Labels are deliberately restricted to CONCRETE PRODUCT categories. The
        # old "technology concept" label invited generic practice/concept spans
        # ("monitoring", "infrastructure as code", "cloud infrastructure") that
        # then leaked into the emerging radar as if they were novel tools. Dropping
        # it also makes each inference ~15% cheaper (one fewer label to match).
        labels = ["software tool", "programming language", "cloud platform", "database", "hardware"]

        # GLiNER truncates at 384 tokens, so we split by paragraph to ensure no skills are lost at the bottom of the JD
        chunks = [c.strip() for c in text.split("\n") if len(c.strip()) > 10]
        entities = []
        for chunk in chunks:
            chunk_entities = self.gliner_model.predict_entities(chunk, labels)
            entities.extend(chunk_entities)

        # Deduplicate and clean up
        existing_skills_lower = set(s.lower() for s in skills)
        emerging: List[str] = []
        emerging_lower: Set[str] = set()

        for ent in entities:
            # Only keep higher confidence matches
            if ent.get("score", 0) < 0.6:
                continue

            tool_name = ent["text"].strip()

            # Basic cleaning to ignore messy extractions
            if len(tool_name) < 2 or len(tool_name) > 30:
                continue

            # Filter out blocklisted terms (soft skills, generic concepts, business terms)
            if tool_name.lower() in SOFT_SKILL_BLOCKLIST:
                continue

            # Fold onto the canonical taxonomy when possible so 'k8s' etc. don't
            # count separately from 'Kubernetes'.
            canonical = self._canonicalize_skill(tool_name)
            if canonical:
                if canonical.lower() not in existing_skills_lower:
                    skills.append(canonical)
                    existing_skills_lower.add(canonical.lower())
                continue

            # Established/ubiquitous tech is real but NOT novel — keep it out of
            # the emerging radar (its whole point is early-warning on new tools).
            if tool_name.lower() in ESTABLISHED_NON_NOVEL:
                continue

            # Genuinely novel skill (beyond the taxonomy) -> emerging bucket.
            if tool_name.islower():
                tool_name = tool_name.capitalize()
            if (
                tool_name.lower() not in existing_skills_lower
                and tool_name.lower() not in emerging_lower
            ):
                emerging.append(tool_name)
                emerging_lower.add(tool_name.lower())

        return (skills, emerging) if with_emerging else skills

    def extract_salary(self, text: str) -> dict:
        """Extract salary information from job description text.
        
        Returns a dict with 'min', 'max', and 'raw' fields.
        If no salary is found, returns an empty dict.
        """
        if not text:
            return {}
        
        for pattern in SALARY_PATTERNS:
            match = pattern.search(text)
            if match:
                d = match.groupdict()
                try:
                    # Minimum salary calculation
                    if d.get("min_zeros"):
                        min_val = int(d["min_k"]) * 1000 + int(d["min_zeros"]) if int(d["min_k"]) < 1000 else int(d["min_k"] + d["min_zeros"])
                    else:
                        min_val = int(d["min_k"]) * 1000
                    
                    # Maximum salary calculation (if present)
                    if d.get("max_k"):
                        if d.get("max_zeros"):
                            max_val = int(d["max_k"]) * 1000 + int(d["max_zeros"]) if int(d["max_k"]) < 1000 else int(d["max_k"] + d["max_zeros"])
                        else:
                            max_val = int(d["max_k"]) * 1000
                    else:
                        max_val = min_val
                        
                    # Sanity check: salaries should be between 30k-500k
                    if 30000 <= min_val <= 500000 and 30000 <= max_val <= 500000:
                        return {"min": min_val, "max": max_val, "raw": match.group(0).strip()}
                except (ValueError, TypeError):
                    continue
        
        return {}

    def categorize_job(self, title: str, description: str) -> str:
        """Semantically categorize the job description."""
        title_lower = str(title).lower()
        
        # 1. Rule-based overrides for high-confidence titles
        if "product manager" not in title_lower and "product owner" not in title_lower:
            if any(kw in title_lower for kw in ["engineering manager", "head of engineering", "vp of engineering", "director of engineering", "engineering team lead", "teamlead engineering"]):
                return "Engineering Leadership"
            if "backend" in title_lower or "back-end" in title_lower:
                return "Backend Engineering"
            if "frontend" in title_lower or "front-end" in title_lower or "fullstack" in title_lower or "full-stack" in title_lower:
                return "Frontend & Fullstack Engineering"
            if "data engineer" in title_lower:
                return "Data Engineering"
            if "data scientist" in title_lower or "machine learning engineer" in title_lower:
                return "Data Science & ML Engineering"
            if "sre" in title_lower.split() or "site reliability" in title_lower:
                return "Site Reliability Engineering (SRE)"
            if "devops" in title_lower:
                return "DevOps Engineering"
            if "platform engineer" in title_lower:
                return "Platform Engineering"
            if "cloud engineer" in title_lower:
                return "Cloud Engineering"
            if "staff engineer" in title_lower or "principal engineer" in title_lower:
                return "Staff / Principal Engineering"
            if "solution architect" in title_lower or "solutions architect" in title_lower:
                return "Solutions Architecture"

        # 2. We combine title and a snippet of the description for categorization
        # Using the first 1000 characters is usually enough for the gist.
        # Repeating the title increases its semantic weight against long descriptions.
        text = f"Title: {title}\nTitle: {title}\nDescription: {str(description)[:1000]}"
        job_emb = self.model.encode(text, convert_to_tensor=True)
        
        # Calculate cosine similarities
        cosine_scores = util.cos_sim(job_emb, self.category_embeddings)[0]
        
        # Find the category with the highest score
        best_idx = torch.argmax(cosine_scores).item()
        best_category = self.category_names[best_idx]
        
        # Safeguard: Product Management shouldn't win if the title explicitly mentions engineering and not product
        if best_category == "Product Management" and "engineer" in title_lower and "product" not in title_lower:
            # Re-evaluate without Product Management
            pm_idx = self.category_names.index("Product Management")
            cosine_scores[pm_idx] = -1.0
            best_idx = torch.argmax(cosine_scores).item()
            best_category = self.category_names[best_idx]

        return best_category

    def compute_match_score(self, description: str) -> float:
        """Compute raw cosine similarity (0-1) against CV.

        Returns a raw cosine similarity value. Callers should apply
        min-max normalisation across the full batch before displaying
        scores, so that the best match always reads ~100 and the
        weakest ~0 (see process_jobs).
        """
        if not description:
            return 0.0

        job_emb = self.model.encode(description, convert_to_tensor=True)
        cosine_score = util.cos_sim(job_emb, self.cv_embedding)[0].item()
        # Clamp to [0, 1] — text cosine similarity is rarely negative
        return max(0.0, min(1.0, cosine_score))

    def process_jobs(self, jobs: List[Dict], cache_path: str = DEFAULT_SCORE_CACHE) -> List[Dict]:
        """Process a list of jobs, adding SemanticCategory, TechSkills, and MatchScore.

        The expensive per-job work (GLiNER extraction, category + CV embeddings,
        salary regex) is memoised in a JSON cache keyed by job id, so jobs that
        recur across nightly runs are not re-scored. An entry is reused only when
        the job text, the CV, and the extraction code all match its fingerprints;
        the *raw* cosine score is cached, but the final SemanticMatchScore is
        always re-normalised across the current batch (see _normalize_scores),
        since normalisation is inherently batch-relative. Pass cache_path=None to
        disable caching (e.g. in tests).
        """
        processed_jobs = []
        cache = self._load_cache(cache_path)
        hits = 0

        print(f"[*] Processing {len(jobs)} jobs...")

        for job in tqdm(jobs, desc="Analyzing Jobs"):
            title = job.get('title', '')
            desc = job.get('descriptionText', '')
            full_text = f"{title}\n{desc}"

            jid = str(job.get('id', ''))
            fp = self._job_fp(title, desc)
            ent = cache.get(jid) if jid else None
            if ent and ent.get('fp') == fp and ent.get('cv') == self._cv_fp and ent.get('code') == self._code_fp:
                # Cache hit — reuse everything deterministic for this job.
                job['TechSkills'] = ent['skills']
                job['EmergingSkills'] = ent['emerging']
                job['SemanticCategory'] = ent['category']
                job['SemanticMatchScore'] = ent['raw']           # raw; normalised below
                job['SalaryMin'], job['SalaryMax'], job['SalaryRaw'] = ent['salary']
                ent['ts'] = time.time()
                hits += 1
            else:
                job['TechSkills'], job['EmergingSkills'] = self.extract_tech_skills(
                    full_text, with_emerging=True
                )
                job['SemanticCategory'] = self.categorize_job(title, desc)
                job['SemanticMatchScore'] = self.compute_match_score(full_text)
                salary_info = self.extract_salary(desc)
                job['SalaryMin'] = salary_info.get('min') if salary_info else None
                job['SalaryMax'] = salary_info.get('max') if salary_info else None
                job['SalaryRaw'] = salary_info.get('raw', '') if salary_info else ''
                if jid:
                    cache[jid] = {
                        'fp': fp, 'cv': self._cv_fp, 'code': self._code_fp,
                        'skills': job['TechSkills'], 'emerging': job['EmergingSkills'],
                        'category': job['SemanticCategory'], 'raw': job['SemanticMatchScore'],
                        'salary': [job['SalaryMin'], job['SalaryMax'], job['SalaryRaw']],
                        'ts': time.time(),
                    }

            # Role-family rejection is deliberately OUTSIDE the cache: it is pure
            # regex (microseconds) and keeping it uncached means a pattern fix
            # takes effect on the next run without invalidating expensive GLiNER
            # extraction for the whole corpus.
            job['RoleFamily'], job['RoleFamilyReason'] = classify_role_family(title, desc)

            processed_jobs.append(job)

        if cache_path:
            print(f"[*] Score cache: {hits}/{len(jobs)} hits ({len(jobs) - hits} freshly scored).")
            self._save_cache(cache_path, cache)

        fam = Counter(j['RoleFamily'] for j in processed_jobs)
        print(f"[*] Role family: {fam.get(ROLE_FAMILY_ON, 0)} on-target · "
              f"{fam.get(ROLE_FAMILY_UNCLEAR, 0)} unclear · "
              f"{fam.get(ROLE_FAMILY_OFF, 0)} off-target")

        # ── Min-max normalise scores across the full batch ────────────────────────
        # Raw cosine values cluster tightly (e.g. 0.51-0.81) making the ranking
        # hard to read.  Normalising spreads them to 0-100 while preserving order.
        raw_scores = [j['SemanticMatchScore'] for j in processed_jobs]
        for j, norm in zip(processed_jobs, self._normalize_scores(raw_scores)):
            j['SemanticMatchScore'] = norm

        # ── Transparent fit signal (additive; needs the normalised score) ─────
        for j in processed_jobs:
            self._apply_fit(j)

        return processed_jobs

    def _apply_fit(self, job: Dict) -> None:
        """Attach a transparent, explainable fit signal on top of the (already
        batch-normalised) SemanticMatchScore. Cosine similarity rewards vocabulary
        overlap; this blends in how many of the job's tools the CV actually lists,
        and records a human-readable WhyMatched string. Adds: CVSkillOverlap (int),
        FitScore (0-100, 0.7 semantic / 0.3 skill-overlap), WhyMatched (str)."""
        sem = float(job.get('SemanticMatchScore', 0.0))
        tools = job.get('TechSkills', []) or []
        shared = [t for t in tools if t in self.cv_skills]
        overlap = len(shared)
        skill_component = min(overlap, 10) / 10.0 * 100.0   # saturates at 10 tools
        job['CVSkillOverlap'] = overlap
        job['FitScore'] = round(0.7 * sem + 0.3 * skill_component, 1)
        shared_str = ", ".join(shared[:5]) if shared else "no shared tools"
        job['WhyMatched'] = (
            f"{sem:.0f}/100 semantic · {overlap} CV tools ({shared_str}) · "
            f"{job.get('SemanticCategory', '')}"
        )

    @staticmethod
    def _load_cache(cache_path) -> Dict:
        if not cache_path or not os.path.exists(cache_path):
            return {}
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}   # corrupt/unreadable cache is never fatal — just rebuild

    @staticmethod
    def _save_cache(cache_path, cache: Dict) -> None:
        # Prune to the most-recently-touched SCORE_CACHE_MAX entries so the file
        # can't grow without bound across months of runs.
        if len(cache) > SCORE_CACHE_MAX:
            keep = sorted(cache.items(), key=lambda kv: kv[1].get('ts', 0), reverse=True)[:SCORE_CACHE_MAX]
            cache = dict(keep)
        try:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            tmp = f"{cache_path}.tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(cache, f)
            os.replace(tmp, cache_path)
        except Exception:
            pass   # cache is an optimisation; never fail the run over it

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """Min-max normalise raw cosine scores to the 0-100 range (2dp), order
        preserved. When there is no spread (all-equal, or a single job) every
        score collapses to 50.0 — avoids divide-by-zero and honestly signals
        'nothing to rank against'. Empty input returns []."""
        if not scores:
            return []
        min_s, max_s = min(scores), max(scores)
        if max_s <= min_s:
            return [50.0 for _ in scores]
        spread = max_s - min_s
        return [round((s - min_s) / spread * 100.0, 2) for s in scores]

    def get_annotated_raw_data(self, raw_file_path: str):
        """Annotates the raw dataset with language, category, and skills. Returns a DataFrame."""
        import os
        if not os.path.exists(raw_file_path):
            return None
            
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            jobs = json.load(f)
            
        df = pd.DataFrame(jobs)
        
        # Calculate language requirements
        if 'is_german_nlp' in globals() and 'requires_german' in globals():
            df["is_german"] = df["descriptionText"].apply(is_german_nlp)
            df["req_german"] = df["descriptionText"].apply(requires_german)
            df["Needs_German"] = df["is_german"] | df["req_german"]
        else:
            df["Needs_German"] = False
            
        # Fast categorization
        cat_names = self.category_names
        cat_embeddings = self.category_embeddings
        
        texts_to_encode = [
            f"Title: {row.get('title', '')}\nDescription: {str(row.get('descriptionText', ''))[:1000]}"
            for _, row in df.iterrows()
        ]
        
        job_embeddings = self.model.encode(texts_to_encode, convert_to_tensor=True)
        cos_scores = util.cos_sim(job_embeddings, cat_embeddings)
        
        categories = []
        for i in range(len(df)):
            best_cat_idx = cos_scores[i].argmax().item()
            categories.append(cat_names[best_cat_idx])
            
        df["Category"] = categories
        tqdm.pandas(desc="Extracting Skills (Raw)")
        df['TechSkills'] = df.progress_apply(lambda row: self.extract_tech_skills(str(row.get('title', '')) + " " + str(row.get('descriptionText', '')), fast_mode=True), axis=1)
        
        return df

def is_germany_location(loc_str):
    import re
    loc = str(loc_str).lower()
    german_markers = ["germany", "deutschland", "berlin", "munich", "münchen", "hamburg", "cologne", "köln", "frankfurt", "stuttgart", "düsseldorf", "dusseldorf", "dortmund", "essen", "hannover", "nuremberg", "nürnberg", "augsburg", "wiesbaden", "heidelberg", "mannheim", "karlsruhe", "freiburg", "leipzig", "dresden", "potsdam", "münster", "bonn", "mainz", "bochum", "aachen", "constance"]
    if any(marker in loc for marker in german_markers) or loc.endswith(", de") or loc == "de":
        return True
    german_region_patterns = [r"rhine.ruhr", r"cologne.?bonn", r"neckar", r"bavarian", r"rhineland", r"westphalia", r"palatinate", r"hesse", r"saxony", r"thuringia", r"mecklenburg", r"saarland", r"swabia", r"lower saxony", r"north rhine", r"baden.w\w+temberg"]
    return any(re.search(p, loc) for p in german_region_patterns)

def _generate_single_report(df_subset: pd.DataFrame, dataset_name: str, output_file_handle, console, is_raw: bool):
    from collections import Counter
    if df_subset.empty:
        return

    df_copy = df_subset.copy()
    if "SemanticCategory" in df_copy.columns and "Category" not in df_copy.columns:
        df_copy["Category"] = df_copy["SemanticCategory"]
        
    if "Needs_German" not in df_copy.columns:
        df_copy["Needs_German"] = False
        
    # 1. Crosstab Language
    crosstab = pd.crosstab(df_copy["Category"], df_copy["Needs_German"], margins=True, margins_name="Total")
    if True in crosstab.columns and False in crosstab.columns:
        crosstab = crosstab[[False, True, "Total"]]
        crosstab.columns = ["English Only", "Requires German", "Total"]
    elif False in crosstab.columns:
        crosstab.columns = ["English Only", "Total"]
        crosstab["Requires German"] = 0
        crosstab = crosstab[["English Only", "Requires German", "Total"]]
    elif True in crosstab.columns:
        crosstab.columns = ["Requires German", "Total"]
        crosstab["English Only"] = 0
        crosstab = crosstab[["English Only", "Requires German", "Total"]]
        
    # After stage 2 every surviving job is English-only, so "Requires German" is
    # a column of zeros in the FILTERED reports. Keep it only for RAW, where it
    # is the whole point of the crosstab.
    if not is_raw and "Requires German" in crosstab.columns:
        crosstab = crosstab[["Total"]] if crosstab["Requires German"].sum() == 0 \
            else crosstab
        if list(crosstab.columns) == ["Total"]:
            crosstab.columns = ["Jobs"]

    formatted_crosstab = crosstab.copy().astype(str)
    for idx in crosstab.index:
        if "Total" not in crosstab.columns:
            continue
        total = crosstab.loc[idx, "Total"]
        if total > 0:
            eng = crosstab.loc[idx, "English Only"]
            ger = crosstab.loc[idx, "Requires German"]
            formatted_crosstab.loc[idx, "English Only"] = f"{eng} ({int(round(eng/total*100))}%)"
            formatted_crosstab.loc[idx, "Requires German"] = f"{ger} ({int(round(ger/total*100))}%)"
            formatted_crosstab.loc[idx, "Total"] = str(total)
            
    # 2. Skills breakdown
    df_skills = pd.DataFrame()
    if not is_raw:
        skills_data = {}
        for cat in df_copy["Category"].unique():
            cat_df = df_copy[df_copy["Category"] == cat]
            cat_total = len(cat_df)
            all_skills = []
            for skills_list in cat_df["TechSkills"]:
                if isinstance(skills_list, list):
                    all_skills.extend(skills_list)
            
            skill_counts = Counter(all_skills)
            top_skills = skill_counts.most_common(10)
            
            skills_str = ", ".join([f"{skill} ({count/cat_total*100:.0f}%)" for skill, count in top_skills]) if top_skills else "None"
            skills_data[cat] = {"Total Jobs": cat_total, "Top Skills": skills_str}
            
        df_skills = pd.DataFrame.from_dict(skills_data, orient="index")
        if not df_skills.empty:
            df_skills.index.name = "Category"
    
    # Render the report. We build the rich tables ONCE and print them to both the
    # live terminal `console` and a recording console whose ANSI export is written
    # to the file — so `cat jobs_..._insights.txt` shows the same colourised tables
    # you see during the run. Falls back to plain pandas text if rich is missing.
    try:
        from rich.table import Table
        from rich.console import Console
        from rich import box

        t1 = Table(title=None, box=box.ROUNDED)
        t1.add_column("Category", justify="left", style="cyan", no_wrap=True)
        for col in formatted_crosstab.columns:
            t1.add_column(str(col), justify="right", style="green")
        for index, row in formatted_crosstab.iterrows():
            style = "bold magenta" if index == "Total" else None
            t1.add_row(str(index), *[str(val) for val in row], style=style)

        t2 = None
        if not is_raw and not df_skills.empty:
            t2 = Table(box=box.ROUNDED, show_lines=True)
            t2.add_column("Category", justify="left", style="cyan", no_wrap=True)
            t2.add_column("Total Jobs", justify="right", style="magenta")
            t2.add_column("Top Skills (% of jobs)", justify="left", style="yellow")
            for index, row in df_skills.iterrows():
                t2.add_row(str(index), str(row["Total Jobs"]), str(row["Top Skills"]))

        # Fixed-width recording console → deterministic tables independent of the
        # real terminal size; force_terminal so it emits ANSI even to a pipe/file.
        #
        # `file=io.StringIO()` is load-bearing: a Console with record=True still
        # WRITES to its file (stdout by default) as well as capturing, so every
        # table was rendered twice in the run log — once by `rec`, once by the
        # live `console`. Recording is unaffected by redirecting the sink.
        import io as _io
        rec = Console(record=True, force_terminal=True, width=100, file=_io.StringIO())
        targets = [rec] + ([console] if console else [])
        for c in targets:
            c.print(f"\n[bold green]MARKET INSIGHTS: {dataset_name}[/bold green]")
            # The filtered report no longer carries a language column (every
            # surviving row is English-only), so the heading would misdescribe it.
            c.print("[bold]1. Category vs Language Requirement[/bold]" if is_raw
                    else "[bold]1. Jobs by Category[/bold]")
            c.print(t1)
            if t2 is not None:
                c.print("\n[bold]2. Top Skills by Category[/bold]")
                c.print(t2)
        output_file_handle.write(rec.export_text(styles=True))
        output_file_handle.write("\n\n")
    except ImportError:
        # Plain-text fallback (no rich available).
        output_file_handle.write(f"MARKET INSIGHTS: {dataset_name}\n")
        output_file_handle.write("=" * 80 + "\n\n")
        output_file_handle.write(("1. Category vs Language Requirement\n" if is_raw
                                  else "1. Jobs by Category\n"))
        output_file_handle.write("-" * 50 + "\n")
        output_file_handle.write(formatted_crosstab.to_string())
        if not is_raw:
            output_file_handle.write("\n\n\n2. Top Skills by Category\n")
            output_file_handle.write("-" * 50 + "\n")
            output_file_handle.write(df_skills.to_string() if not df_skills.empty else "No data")
        output_file_handle.write("\n\n")

def generate_insight_reports(df: pd.DataFrame, output_path: str, is_raw: bool,
                             mode: str = "w"):
    """Write market-insight sections to *output_path*.

    *mode* is passed straight to ``open`` so callers can build a single combined
    report: write the RAW section with mode="w", then append the FILTERED
    section(s) with mode="a" into the same file.
    """
    console = None
    try:
        from rich.console import Console
        console = Console()
    except ImportError:
        pass

    with open(output_path, mode, encoding='utf-8') as f:
        if is_raw:
            _generate_single_report(df, "RAW DATA (All Scraped Jobs, incl. German)", f, console, True)
        else:
            # Filtered data — split into Germany and EU/UK Remote
            df["Region"] = df["location"].apply(lambda loc: "Germany" if is_germany_location(loc) else "EU/UK Remote")
            
            df_germany = df[df["Region"] == "Germany"]
            
            # The EU/UK-remote split was dropped 2026-08-10: it is a small tail
            # (those rows only survive stage 2 when explicitly remote) and it
            # duplicated the Germany section's shape without changing any
            # decision. `Region` is still set, so the CSV retains the breakdown.
            _generate_single_report(df_germany, "FILTERED DATA (Germany)", f, console, False)
                
    print(f"\n[+] Market Insights text file saved to: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Semantic Job Analyzer")
    parser.add_argument("input_file", help="Path to the input JSON file containing jobs")
    parser.add_argument("--keep-off-target", action="store_true",
                        help="Keep roles classified off-target by classify_role_family() "
                             "(FPGA/QA/sales/product/frontend/backend/academia/... ). "
                             "By default they are dropped from the output; the input "
                             "_filtered.json always retains every row.")
    parser.add_argument("--cv-file", default=None,
                        help="Path to CV (.txt, .pdf, .docx). Defaults to cached 'CV.pdf'.")

    args = parser.parse_args()

    # Always download a fresh copy of the CV so Google Docs edits are picked up immediately.
    cv_path = args.cv_file if args.cv_file else _download_cv()
    
    print("==========================================")
    print("        Semantic Job Analyzer             ")
    print("==========================================")
    
    # Load input data
    print(f"[*] Loading jobs from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        jobs = json.load(f)
        
    # Deduplicate jobs by title and companyName
    seen = set()
    unique_jobs = []
    for job in jobs:
        identifier = job.get('id')
        if identifier not in seen:
            seen.add(identifier)
            unique_jobs.append(job)
            
    if len(jobs) != len(unique_jobs):
        print(f"[*] Removed {len(jobs) - len(unique_jobs)} duplicate job postings. {len(unique_jobs)} unique jobs remaining.")
        
    # Initialize analyzer
    analyzer = SemanticJobAnalyzer(cv_path=cv_path)
    
    # Process jobs
    processed_jobs = analyzer.process_jobs(unique_jobs)

    # ── Drop off-target roles from the OUTPUT ────────────────────────────────
    # `classify_role_family` only labels; this is where the label is acted on.
    # Dropping is safe HERE in a way it would not be at stage 1: the input
    # `_filtered.json` still holds every row, so a wrong verdict costs a re-run
    # rather than the job itself. Stage 1's title pre-filter drops before
    # hydration, which is why that one only ever gets tightened with evidence.
    #
    # Only `off-target` is dropped. `unclear` is KEPT on purpose -- it is where
    # the genuinely adjacent roles live (Forward Deployed, AI/ML Engineer, Data
    # Engineering, bare "Software Engineer"), and rejecting on absence of
    # evidence is the mistake documented against the removed zero-stack rule.
    if not args.keep_off_target:
        off = [j for j in processed_jobs if j.get('RoleFamily') == ROLE_FAMILY_OFF]
        processed_jobs = [j for j in processed_jobs if j.get('RoleFamily') != ROLE_FAMILY_OFF]
        if off:
            reasons = Counter(j.get('RoleFamilyReason', '').split(' (')[0] for j in off)
            print(f"[*] Dropped {len(off)} off-target roles from the output "
                  f"({len(processed_jobs)} remain). Top reasons: "
                  + ", ".join(f"{r}={n}" for r, n in reasons.most_common(6)))
            print("    (re-run with --keep-off-target to inspect them)")
    
    # Sort by the composite FitScore (semantic + CV-skill overlap), falling back
    # to the raw semantic score.
    processed_jobs.sort(
        key=lambda x: (x.get('FitScore', 0), x.get('SemanticMatchScore', 0)),
        reverse=True,
    )
    
    # Create DF for stats before flattening TechSkills
    df_filtered = pd.DataFrame(processed_jobs)
    
    # Combined Market Insights: RAW section first, FILTERED section(s) appended,
    # all in a single '{base}_insights.txt' file. (The base drops the '_filtered'
    # suffix so the name is clean — previously the filtered report was written to
    # a doubled '..._filtered_filtered_insights.txt'.)
    raw_file = args.input_file.replace("_Ranked_Filtered.json", ".json").replace("_filtered.json", ".json")
    insights_out = raw_file.replace(".json", "_insights.txt")
    wrote_raw = False
    if ("_filtered" in args.input_file or "_Ranked_Filtered" in args.input_file) and raw_file != args.input_file:
        print("\n[*] Generating Raw Market Insights from full scrape...")
        df_raw = analyzer.get_annotated_raw_data(raw_file)
        if df_raw is not None:
            generate_insight_reports(df_raw, insights_out, is_raw=True, mode="w")
            wrote_raw = True

    # Append (or start, if there was no raw section) the Filtered insights.
    print("\n[*] Generating Filtered Market Insights...")
    generate_insight_reports(df_filtered, insights_out, is_raw=False,
                             mode="a" if wrote_raw else "w")
    
    # Save results
    output_json = args.input_file.replace(".json", "_semantic.json")
    output_csv = args.input_file.replace(".json", "_semantic.csv")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_jobs, f, indent=2, ensure_ascii=False)
        
    # Convert list in 'TechSkills' to a comma-separated string for CSV readability
    if 'TechSkills' in df_filtered.columns:
        df_filtered['TechSkills'] = df_filtered['TechSkills'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    df_filtered.to_csv(output_csv, index=False)
    
    print("\n[+] Success! Output saved to:")
    print(f"    - {output_json}")
    print(f"    - {output_csv}")
    
    print("\n[+] Top 5 Matches (Semantic):")
    for i, job in enumerate(processed_jobs[:5]):
        print(f"\n{i+1}. {job.get('title')} @ {job.get('companyName')}")
        print(f"   Category: {job.get('SemanticCategory')}")
        print(f"   Match Score: {job.get('SemanticMatchScore')}")
        print(f"   Skills: {', '.join(job.get('TechSkills', []))}")

if __name__ == "__main__":
    main()
