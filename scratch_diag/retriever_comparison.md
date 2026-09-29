# Retriever comparison — ASEM-THG vs FastASEM (conv-26)

- config: `configs\presets\sota_benchmark.yaml`  k1=30 k2=8 delta=0.25 lambda=0.35 rrf_k=60 max_hops=2

- **ASEM-THG** bank: 337 notes, 2991 edges  (`static\memory_banks\locomo10\ds_thg\ASEM-THG\locomo_0000\asem_thg.sqlite`)
- **FastASEM** bank: 282 notes, 1877 edges  (`static\memory_banks\locomo10\ds_fixed\FastASEM\locomo_0000\fast_asem.sqlite`)


---

## [Temporal] `When did Caroline go to the LGBTQ support group?`  (gold≈'7 May 2023')

- gold in top-8: **ASEM-THG rank 1** | **FastASEM rank 1**

| rank | ASEM-THG (enhanced) | FastASEM (RRF) |
|--:|---|---|
| 1 | On 7 May 2023, Caroline attended an LGBTQ support group. | On 7 May 2023, Caroline went to an LGBTQ support group. |
| 2 | As of 8 May 2023, the LGBTQ support group has made Caroline feel accepted and given her courage | Caroline joined a new LGBTQ activist group on 11 July 2023. |
| 3 | Caroline is a member of the group 'Connected LGBTQ Activists'. | Caroline is a member of the group 'Connected LGBTQ Activists'. |
| 4 | On 18 July 2023, Caroline joined a new LGBTQ activist group. | On 10 July 2023, Caroline went to an LGBTQ conference. |

---

## [Single-hop] `What did Caroline research?`  (gold≈'adoption')

- gold in top-8: **ASEM-THG rank 2** | **FastASEM rank 2**

| rank | ASEM-THG (enhanced) | FastASEM (RRF) |
|--:|---|---|
| 1 | On 8 May 2023, Caroline planned to go do some research. | Caroline planned to go do some research after the conversation on 8 May 2023. |
| 2 | Caroline is researching adoption agencies. | Caroline is researching adoption agencies. |
| 3 | Caroline has a dream to have a family and give a loving home to kids who need it. | Caroline attended an advocacy event. |
| 4 | Caroline is thrilled to make a family for kids who need one. | Caroline transitioned. |

---

## [Single-hop (term)] `Where did Caroline move from 4 years ago?`  (gold≈'Sweden')

- gold in top-8: **ASEM-THG rank None** | **FastASEM rank 6**

| rank | ASEM-THG (enhanced) | FastASEM (RRF) |
|--:|---|---|
| 1 | Caroline moved from her home country around 2019. | Caroline has known her friends for 4 years, since she moved from her home country. |
| 2 | Caroline has known her friends for 4 years, since she moved from her home country. | Caroline started transitioning three years before 9 June 2023, around June 2020. |
| 3 | Caroline's friends and family have been instrumental in her transition. | Caroline transitioned. |
| 4 | Caroline shared her personal journey, struggles, and development since coming out during her ta | Caroline started playing acoustic guitar about five years ago, around 2018. |

---

## [Multi-hop] `Would Melanie be considered an ally to the transgender community?`  (gold≈'support')

- gold in top-8: **ASEM-THG rank 1** | **FastASEM rank 8**

| rank | ASEM-THG (enhanced) | FastASEM (RRF) |
|--:|---|---|
| 1 | Melanie's family helped out and showed lots of love and support during her move. | Caroline decided to transition and join the transgender community. |
| 2 | Caroline is thinking of working with trans people to help them accept themselves and support th | Caroline mentors a transgender teen who is like her. |
| 3 | As of 8 May 2023, the LGBTQ support group has made Caroline feel accepted and given her courage | Caroline is a transgender woman. |
| 4 | On 7 May 2023, Caroline found the transgender stories at the LGBTQ support group inspiring. | Caroline is going to a transgender conference in July 2023. |

---

## [Conversational] `What are Caroline's plans for the summer?`  (gold≈'adoption')

- gold in top-8: **ASEM-THG rank None** | **FastASEM rank None**

| rank | ASEM-THG (enhanced) | FastASEM (RRF) |
|--:|---|---|
| 1 | Caroline proposed planning something special for the summer of 2023, just the two of them, to c | Caroline and Melanie plan to do something special for the summer of 2023, just the two of them, |
| 2 | Melanie agreed to plan something special with Caroline for the summer of 2023. | Caroline and Melanie plan to do a family outing in the summer of 2023. |
| 3 | Caroline's friends, family, and mentors motivate her and give her strength. | Caroline plans to continue her education. |
| 4 | Caroline is thrilled to make a family for kids who need one. | Caroline plans to check out career options. |

---

## [Bare term] `pottery`  (gold≈'pottery')

- gold in top-8: **ASEM-THG rank 1** | **FastASEM rank 1**

| rank | ASEM-THG (enhanced) | FastASEM (RRF) |
|--:|---|---|
| 1 | Melanie is a big fan of pottery, finding the creativity and skill awesome and calming. | Caroline has not tried pottery. |
| 2 | Pottery is a huge part of Melanie's life, not just a hobby, and helps her express her emotions. | Melanie finds pottery relaxing and creative. |
| 3 | Melanie loves pottery and finds it relaxing and creative. | Melanie made a plate in pottery class on 24 August 2023. |
| 4 | Caroline has not yet done pottery but is open to trying it sometime. | Melanie uses pottery for self-expression and peace. |

---

## Summary — gold rank (top-8, lower is better; `-` = miss)

| query | ASEM-THG | FastASEM |
|---|--:|--:|
| Temporal: When did Caroline go to the LGBTQ support group? | 1 | 1 |
| Single-hop: What did Caroline research? | 2 | 2 |
| Single-hop (term): Where did Caroline move from 4 years ago? | – | 6 |
| Multi-hop: Would Melanie be considered an ally to the transgend | 1 | 8 |
| Conversational: What are Caroline's plans for the summer? | – | – |
| Bare term: pottery | 1 | 1 |
