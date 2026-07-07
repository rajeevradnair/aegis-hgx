# Dataset Pivot Decision

## Purpose

This document records the AEGIS-HGX decision to preserve CIC as the completed tabular baseline dataset and use LANL as the primary dataset for graph construction.

## Phase

This is training/evaluation/test planning and graph data modeling.

It is not inference.

No serving behavior changes are introduced by this decision.

## Previous Assumption

The earlier graph plan assumed that CIC flow data would be used directly for graph construction.

That is possible in a limited form, but it risks creating a weak graph if the available CIC artifact is mainly a cleaned numeric feature matrix.

## Problem

Graph construction requires stable entity identities.

Examples:

- users
- hosts
- source entities
- destination entities
- processes
- ports
- domains
- files
- relationship types

A cleaned tabular feature matrix often removes identity-like columns because tabular models can overfit to identifiers.

That creates a mismatch:

```text
tabular modeling wants clean numeric features
graph modeling needs entity identity and relationships