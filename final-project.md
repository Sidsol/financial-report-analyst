# Final Project: LivingADR

## Introduction

LivingADR is an automated architecture archivist for software teams. It addresses the pervasive problem of documentation rot — stale architectural context that causes AI coding assistants to produce confident but outdated recommendations, hallucinated patterns, and wasted engineering time.

The system observes meaningful repository changes (merged PRs introducing new dependencies, schema changes, or API contract changes), proposes Architecture Decision Records (ADRs) for human review, stores approved rationale as connected architecture knowledge, and serves trustworthy answers to developers and AI coding assistants. The goal is to help teams understand not just *what* the code does, but *why* it exists that way.

The MVP targets a **1 repository / 1 user proof-of-concept** using a personal GitHub repository, with a hybrid architecture consisting of two deployable processes: a GitHub-first workflow service (LangGraph + LlamaIndex + Claude pipeline with a human-in-the-loop review gate) and a read-only MCP context server for IDE/assistant integration.

## Living ADR Documents

- **Vision (Spec):** [final_living_adr/vision.md](final_living_adr/vision.md) — Defines the problem, opportunity, target users, high-level capabilities, MVP scope, success metrics, constraints, and assumptions for the LivingADR project.
- **Architecture:** [final_living_adr/architecture.md](final_living_adr/architecture.md) — Documents the system architecture, tech stack (Python 3.12, FastAPI, LangGraph, LlamaIndex, Claude, MCP SDK), service boundaries, data model, deployment strategy, and anti-patterns to avoid.