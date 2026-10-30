"""Proposal generator for the Cynbit AI proposal assistant."""

import os

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI


# Load environment variables from the project root.
load_dotenv()


def _format_references(references):
    """Convert retrieved project references into readable text."""

    if not references:
        return "No matching reference projects were retrieved."

    if isinstance(references, str):
        return references

    if isinstance(references, dict):
        references = [references]

    formatted = []

    for index, ref in enumerate(references, start=1):
        if not isinstance(ref, dict):
            formatted.append(f"Reference {index}: {ref}")
            continue

        technologies = ref.get("technologies", [])
        deliverables = ref.get("deliverables", [])

        if not isinstance(technologies, (list, tuple)):
            technologies = [technologies]

        if not isinstance(deliverables, (list, tuple)):
            deliverables = [deliverables]

        formatted.append(
            f"""
### Reference {index}: {ref.get('title', 'Untitled project')}

Project ID: {ref.get('project_id', 'Not provided')}
Domain: {ref.get('domain', 'Not provided')}
Summary: {ref.get('summary', 'Not provided')}
Solution: {ref.get('solution', 'Not provided')}
Technologies: {', '.join(map(str, technologies))}
Deliverables: {', '.join(map(str, deliverables))}

Provenance: Bundled reference records are synthetic demonstration data.
They are not verified Cynbit customer engagements.
""".strip()
        )

    return "\n\n---\n\n".join(formatted)


def _local_fallback(rfp_text, context):
    """Return a useful draft when no model API is configured."""

    return f"""# Draft Technical Proposal

**Generation mode:** Local template; no LLM API was called.

## 1. Executive Summary

This preliminary proposal outlines an approach to the client's requirements.
Architecture, estimates, and delivery commitments require further discovery
and human review.

## 2. Client Requirements

{rfp_text}

## 3. Retrieved Reference Projects

{context}

## 4. Recommended Technical Approach

- Confirm functional and non-functional requirements.
- Design the architecture, integrations, and data flows.
- Implement the backend APIs and user interface.
- Select storage and processing services based on workload.
- Implement authentication, authorization, encryption, and audit logging.
- Add monitoring, backups, recovery, and deployment automation.
- Validate reliability, performance, and security before release.

## 5. Implementation Milestones

1. Discovery and requirements validation.
2. Architecture and technical design.
3. Infrastructure and CI/CD foundation.
4. Backend, frontend, and integration development.
5. Testing, security validation, and performance testing.
6. User acceptance, deployment, and handover.

Final durations depend on scope, dependencies, and staffing.

## 6. Team and Effort

Potential roles include solution architect, backend engineer, frontend
engineer, data/AI engineer where applicable, DevOps engineer, and QA engineer.
Person-weeks and calendar duration require a confirmed delivery plan.

## 7. Budget

A reliable estimate requires agreed scope, staffing rates, cloud consumption,
third-party costs, and contingency. This template is not a price quote.

## 8. Security and Compliance

Apply least-privilege access, encryption, secure secret management, audit
logging, retention policies, backup and recovery, and incident response.
Healthcare compliance requires appropriate organizational, contractual,
technical, and operational controls.

## 9. AI/ML Strategy

Where relevant, define a baseline, representative evaluation data, measurable
metrics, false-positive and false-negative analysis, drift monitoring, and
human oversight. Do not invent model performance results.

## 10. Risks and Mitigations

- Unclear requirements: discovery workshops and a decision log.
- Integration failures: validate interfaces early.
- Security issues: threat modeling and security testing.
- Performance issues: representative workload and load testing.
- AI errors: evaluation, monitoring, and human review.

## 11. Assumptions and Open Questions

Confirm requirements, integrations, throughput, latency, availability,
retention, recovery objectives, compliance obligations, budget, and timeline.

## 12. Evidence and Limitations

Retrieved references are contextual examples, not proof that the proposed
solution has been implemented. Bundled reference data is synthetic and must
not be represented as verified Cynbit customer work.
"""


def generate_proposal(
    rfp_text: str,
    references=None,
    model=None,
    api_key=None,
    base_url=None,
):
    """
    Generate a proposal.

    Returns:
        (proposal_markdown, generation_mode)
    """

    if not isinstance(rfp_text, str) or not rfp_text.strip():
        raise ValueError("Please provide a non-empty RFP or project brief.")

    context = _format_references(references)

    system_prompt = """
You are a senior solution architect preparing a client-ready technical
proposal for a software engineering consultancy.

Produce a specific, coherent, technically justified proposal in Markdown.

Required sections:
1. Executive Summary
2. Requirements and Scope
3. Recommended Technical Architecture
4. Security, Privacy, and Compliance
5. AI/ML Strategy, if applicable
6. Implementation Milestones
7. Team and Effort
8. Budget Breakdown
9. Risks and Mitigations
10. Testable Acceptance Criteria
11. Assumptions, Exclusions, and Open Questions
12. Evidence and Limitations

Rules:
- Treat the client RFP as the source of requirements.
- Separate explicit requirements, recommendations, and assumptions.
- Never invent customer engagements, certifications, measured results,
  or contractual commitments.
- Never represent synthetic reference projects as actual Cynbit projects.
- Explain the rationale and trade-offs of the chosen technology stack.
- Keep estimates, budget totals, staffing, and schedule consistent.
- Label budgets and schedules as preliminary estimates.
- Identify missing information instead of inventing precise commitments.
- Do not claim cloud services alone guarantee regulatory compliance.
- For healthcare, address privacy, access controls, auditability, and
  clinical validation. High-impact AI decisions require human oversight.
- Distinguish proposed performance targets from measured results.
- Include testable acceptance criteria and relevant failure handling.
- Clearly identify exclusions and dependencies.
"""

    user_prompt = f"""
Prepare a complete technical proposal based on the following RFP.

================ CLIENT RFP ================

{rfp_text}

============= RETRIEVED REFERENCES =============

{context}

============= ARCHITECTURE GUIDANCE =============

For a cloud-native healthcare dashboard, evaluate AWS where appropriate.
Potential candidates include AWS IoT Core, Kinesis, FastAPI, a suitable
relational database, time-series storage where needed, IAM, KMS, CloudWatch,
Terraform, and CI/CD automation.

These are candidates, not mandatory services. Explain why each selected
technology is appropriate.

For continuous telemetry, identify message frequency, payload size, peak
throughput, latency, availability, retention, and recovery objectives as
discovery questions.

Respect any budget and deadline in the RFP. If estimates are necessary,
label them illustrative, state assumptions and exclusions, and ensure budget
line items add up to the stated total.

For AI/ML, describe a baseline, evaluation data, metrics, monitoring,
limitations, and human oversight. Never fabricate measured results.

Explain how retrieved references inform the recommendations without treating
them as verified customer engagements.

Write all 12 required sections in clear, professional Markdown.
"""

    # Azure OpenAI configuration takes precedence when complete.
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment = os.getenv("CHAT_DEPLOYMENT_NAME")

    azure_configured = all(
        value and value.strip()
        for value in (
            azure_endpoint,
            azure_key,
            azure_version,
            azure_deployment,
        )
    )

    if azure_configured:
        mode = "Azure OpenAI"

        try:
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint.strip(),
                api_key=azure_key.strip(),
                api_version=azure_version.strip(),
            )
            selected_model = azure_deployment.strip()

        except Exception as exc:
            raise RuntimeError(
                f"Azure OpenAI initialization failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    elif api_key and api_key.strip():
        mode = "OpenAI-compatible API"

        client_args = {"api_key": api_key.strip()}

        if base_url and base_url.strip():
            client_args["base_url"] = base_url.strip()

        try:
            client = OpenAI(**client_args)
            selected_model = (
                (model or "").strip()
                or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            )

        except Exception as exc:
            raise RuntimeError(
                f"API client initialization failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    else:
        return (
            _local_fallback(rfp_text, context),
            "Local template (no API configured)",
        )

    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        proposal = response.choices[0].message.content

        if not proposal or not proposal.strip():
            raise RuntimeError("The model returned an empty proposal.")

        return proposal.strip(), mode

    except Exception as exc:
        raise RuntimeError(
            f"Proposal generation failed using {mode}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
