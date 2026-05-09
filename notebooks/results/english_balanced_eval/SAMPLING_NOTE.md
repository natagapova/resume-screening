# Sampling

Soft rebalance: same normalized-text dedupe. Classes with deduped count <= median keep all rows. Classes above median get target = median + soft_pull*(deduped-median). Sampling uses source×has_city_mention strata. If a prior english_balanced_8class_eval.csv existed with exactly 90 rows/class (720 total), it is archived and its rows are preferred when still valid.

## Counts

class,rows_before_full_map,rows_after_soft_rebalance,deduped_pool_full,soft_target_n
backend_general_dev,1174,635,1164,635
generic_it_ops,118,118,118,118
it_governance_leadership,90,90,90,90
project_product,168,168,168,168
sysadmin_devops_network,525,392,525,392
tech_support_helpdesk,209,209,209,209
technical_specialized,678,451,678,451
web_frontend,415,351,415,351