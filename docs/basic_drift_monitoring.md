# Drift Monitoring

Drift monitoring answers:

Is the current data distribution different from the reference data distribution that the model was trained or validated against?

In AEGIS-HGX, this is especially important because cyber environments change over time:
- new users
- new hosts
- new processes
- new network destinations
- new logging behavior
- new attacker behavior
- new business workflows

Drift does not automatically mean attack. It means the model may be seeing unfamiliar input distributions and should be investigated.

