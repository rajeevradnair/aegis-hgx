# LANL Event Type Reference

## Purpose

This document explains the `event_type` values in:

```text
data/processed/lanl/clean_all_events.parquet
```

It is intended to support event-table validation, graph construction, feature engineering, and later model evaluation.

This is **data preparation documentation**. It is not model training and it is not inference.

---

## Core Event Columns

| Column | Meaning |
|---|---|
| `event_family` | Broad source family, such as authentication, DNS, network flow, process, or red-team evidence |
| `event_type` | Specific kind of activity within the event family |
| `source_entity` | Entity that initiated or owns the event |
| `destination_entity` | Entity that received, resolved, or was acted on by the event |
| `source_entity_type` | Type of the source entity, such as `user` or `host` |
| `destination_entity_type` | Type of the destination entity, such as `host`, `process`, or `host_or_domain` |
| `event_result` | Result or status of the event |
| `label` | Ground-truth label; `1` means confirmed red-team evidence, `0` means default non-red-team evidence |

---

## Event Type Summary

| Event Type | Meaning | Graph Shape | Label Behavior |
|---|---|---|---|
| `network_flow` | Host-to-host network communication | `host -> host` | Usually `0` unless later matched to red-team evidence |
| `auth_authmap` | Authentication mapping event | `user -> host` | Usually `0` |
| `process_start` | Process start event | `user -> process` | Usually `0` |
| `auth_tgs` | Kerberos Ticket Granting Service event | `user -> host` | Usually `0` |
| `dns_resolution` | Host resolves another host or domain | `host -> host_or_domain` | Usually `0` |
| `auth_tgt` | Kerberos Ticket Granting Ticket event | `user -> host` | Usually `0` |
| `redteam_activity` | Confirmed adversarial activity | `user -> host` | Always `1` |
| `process_end` | Process end event | `user -> process` | Usually `0` |
| `auth_logon` | User login event | `user -> host` | Usually `0` |
| `auth_logoff` | User logout event | `user -> host` | Usually `0` |

---

## Event Type Details

### `network_flow`

Represents observed network communication from one host to another.

- **Source entity:** `source_host`
- **Destination entity:** `destination_host`
- **Graph interpretation:** `host -> host`
- **Why it matters:** Captures communication patterns, lateral movement signals, unusual peer connections, and traffic relationships.

### `auth_authmap`

Represents an authentication mapping event from the LANL authentication logs.

- **Source entity:** `source_user`
- **Destination entity:** `destination_host`
- **Graph interpretation:** `user -> host`
- **Why it matters:** Helps connect identities to machines and supports user-host relationship modeling.

### `process_start`

Represents the start of a process.

- **Source entity:** `source_user`
- **Destination entity:** `process_name`
- **Graph interpretation:** `user -> process`
- **Why it matters:** Useful for detecting unusual execution behavior, such as unexpected shells, scripts, or administrative tools.

### `auth_tgs`

Represents a Kerberos Ticket Granting Service event.

- **Source entity:** `source_user`
- **Destination entity:** `destination_host`
- **Graph interpretation:** `user -> host`
- **Why it matters:** Captures service-access behavior across hosts and can help detect abnormal access paths.

### `dns_resolution`

Represents a host resolving another host or domain name.

- **Source entity:** `source_host`
- **Destination entity:** `resolved_host`
- **Graph interpretation:** `host -> host_or_domain`
- **Why it matters:** Can reveal discovery, staging, unusual lookup behavior, or preparation for lateral movement.

### `auth_tgt`

Represents a Kerberos Ticket Granting Ticket event.

- **Source entity:** `source_user`
- **Destination entity:** `destination_host`
- **Graph interpretation:** `user -> host`
- **Why it matters:** Captures identity authentication behavior and helps establish normal user login patterns.

### `redteam_activity`

Represents confirmed red-team ground-truth activity from the LANL red-team file.

- **Source entity:** `source_user`
- **Destination entity:** `destination_host`
- **Graph interpretation:** `user -> host`
- **Label:** `1`
- **Why it matters:** Provides confirmed adversarial evidence for evaluation and later label propagation experiments.

### `process_end`

Represents the end of a process.

- **Source entity:** `source_user`
- **Destination entity:** `process_name`
- **Graph interpretation:** `user -> process`
- **Why it matters:** Helps reconstruct process lifetimes and execution sequences.

### `auth_logon`

Represents a login authentication event.

- **Source entity:** `source_user`
- **Destination entity:** `destination_host`
- **Graph interpretation:** `user -> host`
- **Why it matters:** Central to user-host behavior modeling, lateral movement analysis, and temporal authentication patterns.

### `auth_logoff`

Represents a logout authentication event.

- **Source entity:** `source_user`
- **Destination entity:** `destination_host`
- **Graph interpretation:** `user -> host`
- **Why it matters:** Helps identify session boundaries and user-host interaction duration.

---

## Important Modeling Notes

### `event_type` is not the label

`event_type` describes what happened.

`label` describes whether the event is known malicious or not.

| `event_type` | `label` | Meaning |
|---|---:|---|
| `auth_logon` | `0` | Normal login evidence |
| `network_flow` | `0` | Normal network-flow evidence |
| `redteam_activity` | `1` | Confirmed adversarial evidence |

### Preserve `event_type` as an edge attribute

The graph builder should preserve:

```text
event_type
timestamp
event_family
event_result
label
source_file
row_number
```

These fields are needed for traceability, error analysis, temporal modeling, and future explainability.

---

## Graph Construction Rules

| Event Type | Recommended Edge Type |
|---|---|
| `auth_logon` | `user_authenticates_to_host` |
| `auth_logoff` | `user_logs_off_host` |
| `auth_tgt` | `user_requests_tgt_for_host` |
| `auth_tgs` | `user_requests_service_ticket_for_host` |
| `auth_authmap` | `user_maps_to_host` |
| `dns_resolution` | `host_resolves_host_or_domain` |
| `network_flow` | `host_communicates_with_host` |
| `process_start` | `user_starts_process` |
| `process_end` | `user_ends_process` |
| `redteam_activity` | `redteam_user_targets_host` |

---

## Quality Checks

Before graph construction, validate that:

```text
clean_all_events.parquet exists
event_type has no nulls
source_entity has no nulls
destination_entity has no nulls
label contains expected values
redteam_activity rows have label = 1
non-redteam rows default to label = 0
event_type values match this reference
```

Useful command:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_parquet("data/processed/lanl/clean_all_events.parquet")

print(df["event_type"].value_counts())
print(df["label"].value_counts())
print(df[["event_type", "source_entity_type", "destination_entity_type", "label"]].drop_duplicates())
PY
```

---

## Implementation Note

If accidental values such as `9auth_logon` or `10auth_logoff` appear, treat them as normalization bugs. The canonical values should be:

```text
auth_logon
auth_logoff
```