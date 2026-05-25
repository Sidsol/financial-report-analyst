# Project Vision: LivingADR

<!-- CRISPY Project Phase: CLARIFY → produces vision.md -->
<!-- This document captures the project-level vision for a greenfield, multi-feature build. -->
<!-- A "project" is a CONTAINER of features. Individual features get their own spec.md later. -->

| Field        | Value                                                 |
|--------------|-------------------------------------------------------|
| **Project**  | LivingADR                                             |
| **Folder**   | `crispy-docs/projects/004-living-adr`                 |
| **Date**     | 2026-05-24                                            |
| **Status**   | Draft                                                 |

---

## 1. Problem & Opportunity

Documentation rot makes AI coding assistants less trustworthy: stale architectural context causes confident outdated recommendations, hallucinated patterns, and wasted engineering time. Teams also lose the rationale behind structural changes when those decisions live only in pull requests, chats, or individual memory.

LivingADR addresses this by treating architecture documentation as a living context layer, not a one-time writing task. The opportunity is to detect meaningful codebase changes, preserve the human-approved rationale as Architecture Decision Records (ADRs), and make that rationale queryable for both engineers and AI assistants.

---

## 2. Vision Statement

LivingADR is an automated architecture archivist for software teams: it observes meaningful repository changes, proposes ADRs for human review, stores approved rationale as connected architecture knowledge, and serves trustworthy answers to developers and AI coding assistants so they can understand not just what the code does, but why it exists that way.

---

## 3. Target Users & Stakeholders

| Role | Description | Primary Concerns |
|------|-------------|------------------|
| PR authors / engineers | Developers whose structural code changes may create or update architecture rationale. | Low-friction capture, accurate drafts, minimal workflow interruption, clear edit/reject path. |
| Tech leads / architects | Human approvers responsible for architectural quality and decision consistency. | Review control, decision accuracy, avoiding noisy false positives, durable rationale. |
| PoC role note | During the PoC, a single user fulfills both the PR-author and tech-lead-approver roles. | Keep the role distinction explicit for post-PoC expansion. |
| AI coding-assistant users | Engineers using IDE assistants or coding agents that need current architectural context. | Reliable "why" answers, citations to approved decisions, reduced hallucination from stale docs. |
| Platform / DevEx team | Operators responsible for repository integration, service reliability, and developer adoption. | Maintainability, observability, onboarding flow, source-control extensibility, operational cost. |
| Repository owners / maintainers | Teams accountable for codebase health and documentation practices. | Governance, ownership mapping, auditability, avoiding unwanted documentation churn. |

---

## 4. High-Level Capabilities (Feature Themes)

| ID | Theme | Priority | Notes |
|----|-------|----------|-------|
| TH-01 | Change Ingestion & Event Triggering | P1 | Capture enough GitHub repository-change context to decide whether architecture rationale may need to be recorded. |
| TH-02 | Structural-Change & Intent Understanding | P1 | Identify architecture-significant changes and summarize the likely decision intent without treating every code edit as an ADR-worthy event. |
| TH-03 | ADR Drafting & Human Review Workflow | P1 | Produce human-readable ADR candidates and route them through approve / edit / reject review before they become authoritative context. |
| TH-04 | Connected Architecture Knowledge | P1 | Preserve approved ADRs and their relationships to components, code areas, and prior decisions so rationale remains navigable over time. |
| TH-05 | Queryable Context Delivery for IDEs & AI Assistants | P1 | Serve approved architecture rationale to developers and AI tools, especially for "why" questions that require decision context. |
| TH-06 | Observability & Quality Monitoring | P2 | Measure extraction, drafting, approval, and retrieval quality so the system can be tuned and trusted. |
| TH-07 | Governance & Repository Onboarding | P2 | Define ownership, onboarding, and extensibility patterns for adding repositories and future source-control providers. |

---

## 5. MVP Definition

**MVP includes:** A **merged PR** in a single personal GitHub repository (user-provided, TBD) whose diff introduces **a new dependency, a database/schema change, or an API contract change** triggers ingestion; the system identifies the structural-change class; it proposes a single ADR draft; a human approver can approve, edit, or reject it; approved rationale is stored as connected architecture knowledge; and an IDE/assistant-facing query path can return the relevant ADR/rationale for a developer's "why" question.

**MVP excludes:** Azure DevOps support, multi-repo federation, advanced retrieval ranking, broad repository-governance automation, self-hosted graph-database operations, non-ADR documentation generation, drafts triggered during PR review (pre-merge), automated multi-repo rollout (deferred until after PoC validates with 1 repo / 1 user), and fully autonomous authoritative updates without human approval.

**MVP success looks like:** For the **1 repository / 1 user PoC**, the single user-provided personal GitHub repository produces at least one approvable, approved, and queryable ADR through the end-to-end flow, with measurable signal on SM-01 through SM-04.

---

## 6. Success Metrics

All targets are PoC baselines for a **1 repository / 1 user** pilot and should be re-baselined after the PoC.

| ID | Metric | Target | Measured by |
|----|--------|--------|-------------|
| SM-01 | ADR draft acceptance rate | ≥ 50% approved or approved-after-edit during PoC (validate and re-baseline after PoC) | Percentage of generated ADR drafts that are approved or approved-after-edit rather than rejected during the MVP pilot. |
| SM-02 | PR-to-draft latency | ≤ 5 minutes from PR merge to draft available for review (validate and re-baseline after PoC) | Timestamp delta from qualifying merged GitHub PR to ADR draft surfaced for human review. |
| SM-03 | Retrieval relevance for architecture "why" questions | [TBD — needs validation] | Human-rated relevance score across a held-out query set with expected ADR/rationale references. |
| SM-04 | Architecture question coverage | [TBD — needs validation] | Percentage of representative repository "why" questions answerable with cited approved ADR/context from the connected knowledge layer. |
| SM-05 | Unapproved authoritative mutations | 0 in MVP | Audit log comparison of approved review decisions against all authoritative context mutations. |

- SM-03 depends on Open Question: held-out query set definition.
- SM-04 depends on Open Question: held-out query set definition.

---

## 7. Constraints

- **Timeline:** Not stated; PoC scope first, no fixed external deadline.
- **Budget / team size:** 15-person team owns 5 repositories; MVP targets 1 user-owned personal GitHub repository with 1 active reviewer. PoC operates as a single-user loop: the same human submits the PR and reviews the ADR draft. Multi-person review patterns and multi-repo rollout are post-PoC.
- **Compliance / regulatory:** **None — internal tool**; standard internal-only data handling applies, no external regulatory regime.
- **Tech preferences (pre-locked):**
  - LangGraph + LlamaIndex orchestration stack is locked in.
  - GitHub-first integration; Azure DevOps must be a future extensibility seam (architecture should not preclude it).
  - **Claude** is the LLM for V1 (powers structural intent extraction and ADR drafting); other models may be swapped in post-PoC.
  - **Graph storage MUST be a swappable abstraction**, with **LlamaIndex Property Graph** as the default implementation (Neo4j or other graph stores must remain pluggable for post-PoC).

---

## 8. Assumptions

| ID | Assumption | Validated? | Risk if wrong |
|----|------------|------------|---------------|
| A-01 | LlamaIndex Property Graph backing is the default graph/retrieval approach behind a swappable abstraction. | Yes (with swap-seam requirement — see §7) | If swap seam is mis-shaped, swapping out LlamaIndex later forces a larger refactor. |
| A-02 | A dual-MCP topology is the right boundary: one integration/input side and one IDE/assistant-facing context side. | No | The system could be over-split or under-specified, affecting deployment, auth, and developer experience. |
| A-03 | Observability can use either LangSmith or Arize Phoenix. | No | Telemetry, evaluation, and tracing requirements may force a different observability abstraction. |
| A-04 | A human-in-the-loop approval gate before graph mutation is appropriate for MVP. | No | Review friction may slow adoption, or weaker governance may allow stale/incorrect authoritative context. |
| A-05 | Merged-PR diff plus PR description contains enough signal to detect the three structural change classes (new dependency, schema change, API contract change) with acceptable precision. | No | The system may generate noisy drafts or miss important decisions without additional design/context inputs. |
| A-06 | One ADR representation can remain readable for humans while also being effective for LLM retrieval. | No | Optimizing for one audience could degrade usefulness for the other. |
| A-07 | A 1-repo / 1-user PoC produces enough signal to validate the system before broader rollout. | No | PoC may not surface multi-repo issues such as cross-repo decision references, scaling, or repo-onboarding friction; additionally, a single-user PoC where the author and approver are the same person cannot validate inter-person HITL friction, reviewer-availability handling, or approval-conflict dynamics that emerge with separate roles. |
| A-08 | A user-owned personal GitHub repository is a representative PoC environment for validating end-to-end flow. | No | A personal repo may lack the change frequency, dependency churn, or schema/API-contract surface area needed to exercise all three structural-change triggers; PoC may finish without producing enough ADR candidates to meaningfully assess SM-01 and SM-02. |

---

## 9. Out of Scope (Project Level)

- Azure DevOps support in the MVP; it remains a future extensibility seam.
- Multi-repository federation or cross-organization architecture graph rollout in the MVP.
- Multi-repository rollout during the PoC phase (the team owns 5 repos for future context; PoC uses one user-owned personal GitHub repository).
- Self-hosted graph-database operations as an MVP requirement.
- Generating non-ADR documentation such as API reference docs, onboarding guides, runbooks, or changelogs.
- Replacing human architecture governance or making authoritative architecture decisions without accountable review.
- Comprehensive IDE extension UX beyond the minimum query path needed to consume approved context.
- Advanced retrieval ranking, personalization, or broad benchmark suites beyond MVP validation needs.
- Historical backfill of every prior architecture decision unless selected as a deliberate later feature.

---

## 10. Open Questions

- What held-out query set should be used to evaluate retrieval quality and "why" question coverage?
- What final target thresholds should be set for SM-03 retrieval relevance and SM-04 architecture question coverage once the query set is defined?
- Does the IDE/assistant-facing MCP server need an auth model beyond an IDE token or existing developer identity?
- How should repository-to-approver mapping and approval-conflict handling work after PoC, when multi-repo rollout re-opens this problem? (The self-review PoC means inter-person approval dynamics are also post-PoC concerns.)
- Is an audit trail required for rejected drafts, edited drafts, and graph/context mutations during PoC, and what audit detail is required before multi-repo rollout?
- What does the next step after PoC look like — rollout to all 5 repos, expansion to other teams, or both? (Roadmap-time decision.)

---

<!-- NOTE FOR AGENT: -->
<!-- The next CRISPY project phase is DOMAIN RESEARCH. -->
<!-- Domain research must NOT read this vision document — it researches the problem domain blind. -->
