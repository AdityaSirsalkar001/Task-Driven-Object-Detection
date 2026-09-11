# Knowledge Graph Expansion Report

## Scope

Only `task_affordance_kg.yaml` was expanded. Qwen parsing, deterministic graph traversal, Florence detection, clustering, filtering, and ranking logic were not changed. SentenceTransformer and attribute-vector similarity are not used.

## Before and After

| Measure | Before | After |
| --- | ---: | ---: |
| Graph nodes | 255 | 481 |
| Graph edges | 366 | 841 |
| ObjectClasses | 41 | 140 |
| Affordances | 13 | 34 |
| ObjectCategories | 13 | 33 |
| Detector aliases | 92 | 215 |
| Validation errors | 0 | 0 |
| Validation warnings | 0 | 0 |

## Task Coverage

Counts include all roles, including non-output context roles.

| Task | Candidates | Detector aliases | Active nodes | Active edges | Active affordances |
| --- | ---: | ---: | ---: | ---: | --- |
| `dig_hole` | 18 | 29 | 59 | 106 | digging_soil, breaking_hard_ground, scooping_soil, boring_hole, trenching |
| `extinguish_fire` | 20 | 32 | 68 | 123 | extinguishing_directly, carrying_suppressant, delivering_water, delivering_sand, smothering_fire, automatic_fire_suppression |
| `open_parcel` | 13 | 21 | 46 | 70 | cutting_packaging, slicing_tape, opening_package |
| `place_flowers` | 20 | 31 | 73 | 118 | holding_flowers, containing_flower_stems, supporting_plant, temporary_flower_container |
| `pour_sugar` | 17 | 26 | 60 | 102 | dispensing_granules, scooping_granules, holding_sugar, funneling_granules |
| `sit_comfortably` | 15 | 20 | 45 | 88 | supporting_seated_person, cushioned_seating |
| `smear_butter` | 14 | 22 | 48 | 73 | spreading_soft_food, brushing_soft_food |
| `step_on` | 16 | 23 | 49 | 95 | supporting_body_weight, elevated_step, climbing_steps |
| `wine` | 16 | 23 | 52 | 96 | holding_wine, opening_wine, pouring_wine, serving_wine |

## Requested Query Checks

`extinguish_fire` now includes direct tools, suppressant delivery, sand/smothering tools, and fixed systems. Important additions include `fire_hose`, `water_hose`, `bucket`, `water_bucket`, `pail`, `sand_bucket`, `bucket_of_sand`, `fire_bucket`, `fire_sprinkler`, `sprinkler_head`, and `fire_hose_reel`. Water-based candidates contain the condition note: `water-based suppression is fire-type dependent`.

`dig_hole` now includes shovels, spades, trowels, mattocks, pickaxes, post-hole diggers, hand/earth augers, trenching tools, a garden fork, and a soil scoop.

`place_flowers` now keeps preferred destinations above alternatives and weak fallbacks. It includes vases, pots, planters, urns, jars, pitchers, flower buckets, baskets, bottle-based containers, a watering can, and a cup.

## Commands

```bash
./run_three_stage.sh validate-kg
./run_three_stage.sh query-kg --task extinguish_fire
./run_three_stage.sh query-kg --task dig_hole
./run_three_stage.sh query-kg --task place_flowers
```
