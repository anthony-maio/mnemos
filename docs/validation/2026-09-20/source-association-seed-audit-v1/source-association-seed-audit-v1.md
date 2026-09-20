Source-association seed audit v1

Read-only audit of repetition 0 in the terminal 90-row role-aware report. No inference, criteria, or source artifacts were changed.

Source report SHA256: 7dae4c375c65c3b8f82b85375f60fb87d30d641a91dc348c30dcd0f9a31d0f3f
Measurement relevance floor: 0.2

Evidence boundaries: the source report remains complete, with 18 lifecycle control failures and zero target-top-five rows. This audit makes no quality claim.

Observed cases:
- nomic_task / revision_forget_replay:
  - semantic: seed=revision_forget_replay-distractor-31; seed_available_relation=False; source_edges_active=False; target_direct=True; target_status=scored; semantic=0.7255944013595581; graph_energy=0.0; total=0.7255944013595581; rank=7; structural_path=False; activation_path=False; top5_fifth=0.7309426665306091
  - source_association: seed=revision_forget_replay-distractor-31; seed_available_relation=False; source_edges_active=True; target_direct=True; target_status=scored; semantic=0.7255944013595581; graph_energy=0.0; total=0.7255944013595581; rank=7; structural_path=False; activation_path=False; top5_fifth=0.7309426665306091
  - combined: seed=revision_forget_replay-distractor-31; seed_available_relation=False; source_edges_active=True; target_direct=True; target_status=scored; semantic=0.7255944013595581; graph_energy=0.0; total=0.7255944013595581; rank=7; structural_path=False; activation_path=False; top5_fifth=0.7309426665306091
  - combined vs semantic: same_returned_ids=True; target_rank_delta=0; target_graph_energy_delta=0.0; target_total_delta=0.0; path_details_changed=False; newly_reachable=False
  - source_association vs semantic: same_returned_ids=True; target_rank_delta=0; target_graph_energy_delta=0.0; target_total_delta=0.0; path_details_changed=False; newly_reachable=False
- nomic_task / scope_isolation_pair:
  - semantic: seed=scope_isolation_pair-pump; seed_available_relation=True; source_edges_active=False; target_direct=False; target_status=scored; semantic=0.6441748142242432; graph_energy=0.5769215106964112; total=0.7018669652938843; rank=31; structural_path=True; activation_path=True; top5_fifth=0.7634403109550476
  - source_association: seed=scope_isolation_pair-pump; seed_available_relation=True; source_edges_active=True; target_direct=False; target_status=scored; semantic=0.6441748142242432; graph_energy=0.8; total=0.7241748142242432; rank=20; structural_path=True; activation_path=True; top5_fifth=0.7634403109550476
  - combined: seed=scope_isolation_pair-pump; seed_available_relation=True; source_edges_active=True; target_direct=False; target_status=scored; semantic=0.6441748142242432; graph_energy=0.8; total=0.7241748142242432; rank=20; structural_path=True; activation_path=True; top5_fifth=0.7634403109550476
  - combined vs semantic: same_returned_ids=True; target_rank_delta=-11; target_graph_energy_delta=0.22307848930358887; target_total_delta=0.022307848930358976; path_details_changed=True; newly_reachable=False
  - source_association vs semantic: same_returned_ids=True; target_rank_delta=-11; target_graph_energy_delta=0.22307848930358887; target_total_delta=0.022307848930358976; path_details_changed=True; newly_reachable=False
- nomic_task / semantic_only_chain:
  - semantic: seed=semantic_only_chain-quiet_room; seed_available_relation=False; source_edges_active=False; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.6467072367668152
  - source_association: seed=semantic_only_chain-quiet_room; seed_available_relation=False; source_edges_active=True; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.6467072367668152
  - combined: seed=semantic_only_chain-quiet_room; seed_available_relation=False; source_edges_active=True; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.6467072367668152
  - combined vs semantic: same_returned_ids=True; target_rank_delta=None; target_graph_energy_delta=0.0; target_total_delta=None; path_details_changed=False; newly_reachable=False
  - source_association vs semantic: same_returned_ids=True; target_rank_delta=None; target_graph_energy_delta=0.0; target_total_delta=None; path_details_changed=False; newly_reachable=False
- nomic_task / shared_capture_triplet:
  - semantic: seed=shared_capture_triplet-distractor-11; seed_available_relation=False; source_edges_active=False; target_direct=True; target_status=scored; semantic=0.6115994453430176; graph_energy=0.0; total=0.6115994453430176; rank=32; structural_path=False; activation_path=False; top5_fifth=0.7080897021293641
  - source_association: seed=shared_capture_triplet-distractor-11; seed_available_relation=False; source_edges_active=True; target_direct=True; target_status=scored; semantic=0.6115994453430176; graph_energy=0.0; total=0.6115994453430176; rank=28; structural_path=False; activation_path=False; top5_fifth=0.6766791939735413
  - combined: seed=shared_capture_triplet-distractor-11; seed_available_relation=False; source_edges_active=True; target_direct=True; target_status=scored; semantic=0.6115994453430176; graph_energy=0.0; total=0.6115994453430176; rank=32; structural_path=False; activation_path=False; top5_fifth=0.7080897021293641
  - combined vs semantic: same_returned_ids=True; target_rank_delta=0; target_graph_energy_delta=0.0; target_total_delta=0.0; path_details_changed=False; newly_reachable=False
  - source_association vs semantic: same_returned_ids=False; target_rank_delta=-4; target_graph_energy_delta=0.0; target_total_delta=0.0; path_details_changed=False; newly_reachable=False
- nomic_task / shared_extraction_batch:
  - semantic: seed=shared_extraction_batch-distractor-19; seed_available_relation=False; source_edges_active=False; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.7746619251268912
  - source_association: seed=shared_extraction_batch-distractor-19; seed_available_relation=False; source_edges_active=True; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.725310742855072
  - combined: seed=shared_extraction_batch-distractor-19; seed_available_relation=False; source_edges_active=True; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.7746619251268912
  - combined vs semantic: same_returned_ids=True; target_rank_delta=None; target_graph_energy_delta=0.0; target_total_delta=None; path_details_changed=False; newly_reachable=False
  - source_association vs semantic: same_returned_ids=False; target_rank_delta=None; target_graph_energy_delta=0.0; target_total_delta=None; path_details_changed=False; newly_reachable=False
- unprefixed / revision_forget_replay:
  - semantic: seed=revision_forget_replay-distractor-31; seed_available_relation=False; source_edges_active=False; target_direct=True; target_status=scored; semantic=0.718430757522583; graph_energy=0.0; total=0.718430757522583; rank=15; structural_path=False; activation_path=False; top5_fifth=0.7791932990209346
  - source_association: seed=revision_forget_replay-distractor-31; seed_available_relation=False; source_edges_active=True; target_direct=True; target_status=scored; semantic=0.718430757522583; graph_energy=0.0; total=0.718430757522583; rank=14; structural_path=False; activation_path=False; top5_fifth=0.7410467863082886
  - combined: seed=revision_forget_replay-distractor-31; seed_available_relation=False; source_edges_active=True; target_direct=True; target_status=scored; semantic=0.718430757522583; graph_energy=0.0; total=0.718430757522583; rank=15; structural_path=False; activation_path=False; top5_fifth=0.7791932990209346
  - combined vs semantic: same_returned_ids=True; target_rank_delta=0; target_graph_energy_delta=0.0; target_total_delta=0.0; path_details_changed=False; newly_reachable=False
  - source_association vs semantic: same_returned_ids=False; target_rank_delta=-1; target_graph_energy_delta=0.0; target_total_delta=0.0; path_details_changed=False; newly_reachable=False
- unprefixed / scope_isolation_pair:
  - semantic: seed=scope_isolation_pair-distractor-07; seed_available_relation=False; source_edges_active=False; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.8110563501886973
  - source_association: seed=scope_isolation_pair-distractor-07; seed_available_relation=False; source_edges_active=True; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.7708336114883423
  - combined: seed=scope_isolation_pair-distractor-07; seed_available_relation=False; source_edges_active=True; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.8110563501886973
  - combined vs semantic: same_returned_ids=True; target_rank_delta=None; target_graph_energy_delta=0.0; target_total_delta=None; path_details_changed=False; newly_reachable=False
  - source_association vs semantic: same_returned_ids=False; target_rank_delta=None; target_graph_energy_delta=0.0; target_total_delta=None; path_details_changed=False; newly_reachable=False
- unprefixed / semantic_only_chain:
  - semantic: seed=semantic_only_chain-quiet_room; seed_available_relation=False; source_edges_active=False; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.6431610584259033
  - source_association: seed=semantic_only_chain-quiet_room; seed_available_relation=False; source_edges_active=True; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.6431610584259033
  - combined: seed=semantic_only_chain-quiet_room; seed_available_relation=False; source_edges_active=True; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.6431610584259033
  - combined vs semantic: same_returned_ids=True; target_rank_delta=None; target_graph_energy_delta=0.0; target_total_delta=None; path_details_changed=False; newly_reachable=False
  - source_association vs semantic: same_returned_ids=True; target_rank_delta=None; target_graph_energy_delta=0.0; target_total_delta=None; path_details_changed=False; newly_reachable=False
- unprefixed / shared_capture_triplet:
  - semantic: seed=shared_capture_triplet-distractor-11; seed_available_relation=False; source_edges_active=False; target_direct=True; target_status=scored; semantic=0.5700389742851257; graph_energy=0.0; total=0.5700389742851257; rank=33; structural_path=False; activation_path=False; top5_fifth=0.729273750782013
  - source_association: seed=shared_capture_triplet-distractor-11; seed_available_relation=False; source_edges_active=True; target_direct=True; target_status=scored; semantic=0.5700389742851257; graph_energy=0.0; total=0.5700389742851257; rank=33; structural_path=False; activation_path=False; top5_fifth=0.6716575026512146
  - combined: seed=shared_capture_triplet-distractor-11; seed_available_relation=False; source_edges_active=True; target_direct=True; target_status=scored; semantic=0.5700389742851257; graph_energy=0.0; total=0.5700389742851257; rank=33; structural_path=False; activation_path=False; top5_fifth=0.729273750782013
  - combined vs semantic: same_returned_ids=True; target_rank_delta=0; target_graph_energy_delta=0.0; target_total_delta=0.0; path_details_changed=False; newly_reachable=False
  - source_association vs semantic: same_returned_ids=False; target_rank_delta=0; target_graph_energy_delta=0.0; target_total_delta=0.0; path_details_changed=False; newly_reachable=False
- unprefixed / shared_extraction_batch:
  - semantic: seed=shared_extraction_batch-distractor-09; seed_available_relation=False; source_edges_active=False; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.7904244218888705
  - source_association: seed=shared_extraction_batch-distractor-09; seed_available_relation=False; source_edges_active=True; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.7580910921096802
  - combined: seed=shared_extraction_batch-distractor-09; seed_available_relation=False; source_edges_active=True; target_direct=False; target_status=not_in_candidate_pool; semantic=None; graph_energy=0.0; total=None; rank=None; structural_path=False; activation_path=False; top5_fifth=0.7904244218888705
  - combined vs semantic: same_returned_ids=True; target_rank_delta=None; target_graph_energy_delta=0.0; target_total_delta=None; path_details_changed=False; newly_reachable=False
  - source_association vs semantic: same_returned_ids=False; target_rank_delta=None; target_graph_energy_delta=0.0; target_total_delta=None; path_details_changed=False; newly_reachable=False

Conclusions:
- The saved traces establish two observable bottlenecks: distractor seed selection prevents source relations from reaching the authored target in four cases, and even the one reachable prefixed target remains below the top-five cutoff.
- Source-association edges change target path details only in the prefixed scope-isolation case; they do not create new target reachability there because the target is already reachable in the semantic arm.
- The traces do not establish why the embedding semantically prefers distractors. That remains unresolved without a separate embedding-space or query/document semantic investigation.
