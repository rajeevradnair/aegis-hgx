# LANL Graph Inspection Profile

Generated at UTC: `2026-07-09T23:40:39.884913+00:00`

This report validates the LANL graph before PyTorch Geometric conversion and graph model training.

This is graph validation and evaluation preparation, not inference.

## Graph Summary

| Metric | Value |
|---|---:|
| graph_type | MultiDiGraph |
| is_directed | True |
| is_multigraph | True |
| node_count | 24102 |
| edge_count | 4000749 |

## Table Counts

| Metric | Value |
|---|---:|
| node_rows | 24102 |
| edge_rows | 4000749 |
| valid_edge_rows | 4000749 |
| invalid_edge_rows | 0 |
| unique_node_ids | 24102 |
| unique_edge_ids | 4000749 |

## Node Type Counts

| Value | Count |
|---|---:|
| host | 9446 |
| host_or_domain | 1775 |
| process | 1895 |
| user | 10986 |

## Node Label Counts

| Value | Count |
|---|---:|
| 0 | 23697 |
| 1 | 405 |

## Edge Type Counts

| Value | Count |
|---|---:|
| authentication:auth_authmap | 8560 |
| authentication:auth_logoff | 430915 |
| authentication:auth_logon | 416219 |
| authentication:auth_tgs | 98266 |
| authentication:auth_tgt | 46040 |
| dns:dns_resolution | 1000000 |
| network_flow:network_flow | 1000000 |
| process:process_end | 287401 |
| process:process_start | 712599 |
| redteam_ground_truth:redteam_activity | 749 |

## Event Family Counts

| Value | Count |
|---|---:|
| authentication | 1000000 |
| dns | 1000000 |
| network_flow | 1000000 |
| process | 1000000 |
| redteam_ground_truth | 749 |

## Edge Label Counts

| Value | Count |
|---|---:|
| 0 | 4000000 |
| 1 | 749 |

## Total Degree Summary

| Metric | Value |
|---|---:|
| min | 1 |
| max | 200239 |
| mean | 331.9848145382126 |
| median | 35.0 |
| p95 | 765.0 |

## In Degree Summary

| Metric | Value |
|---|---:|
| min | 0 |
| max | 200239 |
| mean | 165.9924072691063 |
| median | 0.0 |
| p95 | 218.95000000000073 |

## Out Degree Summary

| Metric | Value |
|---|---:|
| min | 0 |
| max | 60117 |
| mean | 165.9924072691063 |
| median | 13.0 |
| p95 | 473.0 |

## Top Total-Degree Nodes

| Node ID | Node Key | Entity Type | Label | Total Degree | In Degree | Out Degree |
|---:|---|---|---:|---:|---:|---:|
| 11765 | process::P16 | process | 0 | 200239 | 200239 | 0 |
| 7208 | host::C586 | host | 1 | 199851 | 167662 | 32189 |
| 11092 | host_or_domain::C706 | host_or_domain | 0 | 161338 | 161338 | 0 |
| 12693 | process::P5 | process | 0 | 137478 | 137478 | 0 |
| 9925 | host_or_domain::C1685 | host_or_domain | 0 | 130981 | 130981 | 0 |
| 6400 | host::C457 | host | 1 | 129951 | 114603 | 15348 |
| 9940 | host_or_domain::C1707 | host_or_domain | 0 | 125828 | 125828 | 0 |
| 6465 | host::C467 | host | 1 | 124991 | 95406 | 29585 |
| 6864 | host::C529 | host | 1 | 112920 | 94522 | 18398 |
| 7362 | host::C612 | host | 1 | 109490 | 91846 | 17644 |
| 6858 | host::C528 | host | 1 | 98569 | 71780 | 26789 |
| 10902 | host_or_domain::C5030 | host_or_domain | 0 | 96381 | 96381 | 0 |
| 7434 | host::C625 | host | 1 | 93401 | 82242 | 11159 |
| 193 | host::C1065 | host | 1 | 92534 | 75748 | 16786 |
| 2325 | host::C1685 | host | 0 | 80805 | 39922 | 40883 |
| 7135 | host::C5721 | host | 0 | 69224 | 43994 | 25230 |
| 2281 | host::C16712 | host | 0 | 63225 | 3108 | 60117 |
| 2391 | host::C1707 | host | 0 | 61274 | 30362 | 30912 |
| 12422 | process::P25 | process | 0 | 56657 | 56657 | 0 |
| 7134 | host::C5720 | host | 0 | 52140 | 37491 | 14649 |
| 5572 | host::C3380 | host | 1 | 47961 | 20 | 47941 |
| 10366 | host_or_domain::C22841 | host_or_domain | 0 | 44135 | 44135 | 0 |
| 10843 | host_or_domain::C457 | host_or_domain | 0 | 42144 | 42144 | 0 |
| 7895 | host::C706 | host | 1 | 38020 | 21536 | 16484 |
| 11008 | host_or_domain::C586 | host_or_domain | 0 | 37071 | 37071 | 0 |

## Top In-Degree Nodes

| Node ID | Node Key | Entity Type | Label | Total Degree | In Degree | Out Degree |
|---:|---|---|---:|---:|---:|---:|
| 11765 | process::P16 | process | 0 | 200239 | 200239 | 0 |
| 7208 | host::C586 | host | 1 | 199851 | 167662 | 32189 |
| 11092 | host_or_domain::C706 | host_or_domain | 0 | 161338 | 161338 | 0 |
| 12693 | process::P5 | process | 0 | 137478 | 137478 | 0 |
| 9925 | host_or_domain::C1685 | host_or_domain | 0 | 130981 | 130981 | 0 |
| 9940 | host_or_domain::C1707 | host_or_domain | 0 | 125828 | 125828 | 0 |
| 6400 | host::C457 | host | 1 | 129951 | 114603 | 15348 |
| 10902 | host_or_domain::C5030 | host_or_domain | 0 | 96381 | 96381 | 0 |
| 6465 | host::C467 | host | 1 | 124991 | 95406 | 29585 |
| 6864 | host::C529 | host | 1 | 112920 | 94522 | 18398 |
| 7362 | host::C612 | host | 1 | 109490 | 91846 | 17644 |
| 7434 | host::C625 | host | 1 | 93401 | 82242 | 11159 |
| 193 | host::C1065 | host | 1 | 92534 | 75748 | 16786 |
| 6858 | host::C528 | host | 1 | 98569 | 71780 | 26789 |
| 12422 | process::P25 | process | 0 | 56657 | 56657 | 0 |
| 10366 | host_or_domain::C22841 | host_or_domain | 0 | 44135 | 44135 | 0 |
| 7135 | host::C5721 | host | 0 | 69224 | 43994 | 25230 |
| 10843 | host_or_domain::C457 | host_or_domain | 0 | 42144 | 42144 | 0 |
| 2325 | host::C1685 | host | 0 | 80805 | 39922 | 40883 |
| 7134 | host::C5720 | host | 0 | 52140 | 37491 | 14649 |
| 11008 | host_or_domain::C586 | host_or_domain | 0 | 37071 | 37071 | 0 |
| 12321 | process::P21 | process | 0 | 36265 | 36265 | 0 |
| 13040 | process::P9 | process | 0 | 34691 | 34691 | 0 |
| 10922 | host_or_domain::C528 | host_or_domain | 0 | 34124 | 34124 | 0 |
| 2391 | host::C1707 | host | 0 | 61274 | 30362 | 30912 |

## Top Out-Degree Nodes

| Node ID | Node Key | Entity Type | Label | Total Degree | In Degree | Out Degree |
|---:|---|---|---:|---:|---:|---:|
| 2281 | host::C16712 | host | 0 | 63225 | 3108 | 60117 |
| 5572 | host::C3380 | host | 1 | 47961 | 20 | 47941 |
| 2325 | host::C1685 | host | 0 | 80805 | 39922 | 40883 |
| 7208 | host::C586 | host | 1 | 199851 | 167662 | 32189 |
| 2391 | host::C1707 | host | 0 | 61274 | 30362 | 30912 |
| 6465 | host::C467 | host | 1 | 124991 | 95406 | 29585 |
| 22668 | user::U22@DOM1 | user | 0 | 28749 | 0 | 28749 |
| 4583 | host::C2492 | host | 0 | 27680 | 316 | 27364 |
| 6858 | host::C528 | host | 1 | 98569 | 71780 | 26789 |
| 13948 | user::C1685$@DOM1 | user | 0 | 26392 | 0 | 26392 |
| 7135 | host::C5721 | host | 0 | 69224 | 43994 | 25230 |
| 23678 | user::U66@DOM1 | user | 1 | 20590 | 0 | 20590 |
| 5422 | host::C3173 | host | 1 | 20106 | 97 | 20009 |
| 7180 | host::C5802 | host | 0 | 32495 | 12608 | 19887 |
| 16878 | user::C599$@DOM1 | user | 0 | 19376 | 0 | 19376 |
| 6864 | host::C529 | host | 1 | 112920 | 94522 | 18398 |
| 13201 | user::ANONYMOUS LOGON@C586 | user | 0 | 17934 | 0 | 17934 |
| 7362 | host::C612 | host | 1 | 109490 | 91846 | 17644 |
| 13889 | user::C1624$@DOM1 | user | 0 | 16987 | 0 | 16987 |
| 193 | host::C1065 | host | 1 | 92534 | 75748 | 16786 |
| 7895 | host::C706 | host | 1 | 38020 | 21536 | 16484 |
| 6400 | host::C457 | host | 1 | 129951 | 114603 | 15348 |
| 13337 | user::C1114$@DOM1 | user | 0 | 15141 | 0 | 15141 |
| 7134 | host::C5720 | host | 0 | 52140 | 37491 | 14649 |
| 7164 | host::C5778 | host | 0 | 15145 | 781 | 14364 |

## Connected Components

### Weak Components

## Weak Component Summary

| Metric | Value |
|---|---:|
| component_count | 36 |
| largest_component_size | 24024 |
| smallest_component_size | 2 |
| component_size_sample | [24024, 3, 3, 3, 3, 3, 3, 3, 3, 2] |
| largest_component_ratio | 0.9967637540453075 |

### Strong Components

## Strong Component Summary

| Metric | Value |
|---|---:|
| component_count | 21403 |
| largest_component_size | 2694 |
| smallest_component_size | 1 |
| component_size_sample | [2694, 5, 2, 2, 1, 1, 1, 1, 1, 1] |
| largest_component_ratio | 0.1117749564351506 |

## Isolated Nodes

| Metric | Value |
|---|---:|
| isolated_node_count | 0 |

## Self-Loops

| Metric | Value |
|---|---:|
| self_loop_count | 0 |

## Red-Team Summary

| Metric | Value |
|---|---:|
| redteam_edge_count | 749 |
| redteam_participating_node_count | 405 |
| redteam_labeled_node_count | 405 |

## Red-Team Edge Type Counts

| Value | Count |
|---|---:|
| redteam_ground_truth:redteam_activity | 749 |

## Red-Team Node Type Counts

| Value | Count |
|---|---:|
| host | 301 |
| user | 104 |
