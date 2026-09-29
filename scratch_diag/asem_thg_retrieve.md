# ASEM-THG retrieval test

- bank: `static\memory_banks\locomo10\ds_thg\ASEM-THG\locomo_0000\asem_thg.sqlite`  (337 notes)
- retriever: `EnhancedHybridRetriever`  k1=20 k2=5 delta=0.3 lambda=0.4 max_hops=2 alpha/beta/gamma=0.35/0.25/0.40
- model config: `configs\models\deepseek_api.yaml`  (embedder only; no LLM calls)

## Raw retrieval channels (what the RRF fuse consumes)

**BM25 channel** — query `Sweden` (top 5):
1. (8.44) Caroline owns a necklace with a cross and a heart that was a gift from her grandmother in Sweden.
2. (4.46) Caroline's grandmother gave her the necklace when Caroline was young, and it symbolizes love, faith and streng

**Entity channel** — entities `['Melanie','pottery']` (top 5):
1. Melanie's kids were stoked for the dinosaur exhibit at the museum.  (ents: Melanie)
2. Melanie read a book in 2022 that reminds her to always pursue her dreams.  (ents: Melanie)
3. Melanie has been running longer distances since her last chat with Caroline to de-stress and clear her mind.  (ents: Melanie)
4. Melanie finds running great for her headspace and mental health and plans to keep it up.  (ents: Melanie)
5. Caroline advises Melanie to prioritize mental health and take care of herself.  (ents: Caroline, Melanie)

---

## [Temporal (when)] `When did Caroline go to the LGBTQ support group?`

- base RRF (Phase A+B): **8** notes (phase_a_hits=53)
- after multi-hop + global re-rank: **13** notes (multi_hop_added=5, max_hops=2)

**Top retrieved (enhanced):**

| # | sim(e_q,e) | intent(e_q,z) | q | session_date | fact | entities |
|--:|--:|--:|--:|---|---|---|
| 1 | 0.841 | 0.876 | 0.50 | 1:56 pm on 8 May, 2023 | On 7 May 2023, Caroline attended an LGBTQ support group. | Caroline |
| 2 | 0.835 | 0.814 | 0.50 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, the LGBTQ support group has made Caroline feel accepted and given her courage to embrace herself. | Caroline |
| 3 | 0.858 | 0.898 | 0.50 | 8:56 pm on 20 July, 2023 | Caroline is a member of the group 'Connected LGBTQ Activists'. | Caroline, Connected LGBTQ Activists |
| 4 | 0.814 | 0.814 | 0.50 | 8:56 pm on 20 July, 2023 | On 18 July 2023, Caroline joined a new LGBTQ activist group. | Caroline |
| 5 | 0.744 | 0.724 | 0.50 | 1:56 pm on 8 May, 2023 | On 7 May 2023, Caroline found the transgender stories at the LGBTQ support group inspiring. | Caroline |
| 6 | 0.773 | 0.762 | 0.50 | 1:56 pm on 8 May, 2023 | On 7 May 2023, Caroline felt happy and thankful for the support at the LGBTQ support group. | Caroline |

---

## [Bare term] `Sweden`

- base RRF (Phase A+B): **5** notes (phase_a_hits=2)
- after multi-hop + global re-rank: **10** notes (multi_hop_added=5, max_hops=2)

**Top retrieved (enhanced):**

| # | sim(e_q,e) | intent(e_q,z) | q | session_date | fact | entities |
|--:|--:|--:|--:|---|---|---|
| 1 | 0.313 | 0.238 | 0.50 | 10:37 am on 27 June, 2023 | Caroline owns a necklace with a cross and a heart that was a gift from her grandmother in Sweden. | Caroline, Sweden |
| 2 | 0.271 | 0.247 | 0.50 | 7:55 pm on 9 June, 2023 | Caroline moved from her home country around 2019. | Caroline |
| 3 | 0.077 | 0.087 | 0.50 | 1:14 pm on 25 May, 2023 | Caroline has a dream to have a family and give a loving home to kids who need it. | Caroline |
| 4 | 0.066 | 0.029 | 0.50 | 7:55 pm on 9 June, 2023 | Caroline's friends, family, and mentors motivate her and give her strength. | Caroline |
| 5 | 0.065 | 0.025 | 0.50 | 1:14 pm on 25 May, 2023 | Caroline is grateful for all the support she has got from friends and mentors. | Caroline |
| 6 | 0.152 | 0.004 | 0.50 | 10:37 am on 27 June, 2023 | Caroline's grandmother gave her the necklace when Caroline was young, and it symbolizes love, faith and strength. | Caroline, Sweden |

---

## [Bare term] `pottery`

- base RRF (Phase A+B): **8** notes (phase_a_hits=20)
- after multi-hop + global re-rank: **13** notes (multi_hop_added=5, max_hops=2)

**Top retrieved (enhanced):**

| # | sim(e_q,e) | intent(e_q,z) | q | session_date | fact | entities |
|--:|--:|--:|--:|---|---|---|
| 1 | 0.548 | 0.565 | 0.50 | 1:36 pm on 3 July, 2023 | Melanie is a big fan of pottery, finding the creativity and skill awesome and calming. | Melanie |
| 2 | 0.568 | 0.600 | 0.50 | 1:36 pm on 3 July, 2023 | Pottery is a huge part of Melanie's life, not just a hobby, and helps her express her emotions. | Melanie |
| 3 | 0.628 | 0.646 | 0.50 | 1:33 pm on 25 August, 2023 | Melanie loves pottery and finds it relaxing and creative. | Melanie |
| 4 | 0.634 | 0.651 | 0.50 | 12:09 am on 13 September, 2023 | Caroline has not yet done pottery but is open to trying it sometime. | Caroline |
| 5 | 0.545 | 0.560 | 0.50 | 1:36 pm on 3 July, 2023 | Melanie finds pottery class to be like therapy, letting her express herself and get creative. | Melanie |
| 6 | 0.561 | 0.529 | 0.50 | 1:36 pm on 3 July, 2023 | Melanie made a bowl with a black and white flower design in her pottery class. | Melanie |

---

## [Single-hop] `What did Caroline research?`

- base RRF (Phase A+B): **8** notes (phase_a_hits=60)
- after multi-hop + global re-rank: **13** notes (multi_hop_added=5, max_hops=2)

**Top retrieved (enhanced):**

| # | sim(e_q,e) | intent(e_q,z) | q | session_date | fact | entities |
|--:|--:|--:|--:|---|---|---|
| 1 | 0.718 | 0.739 | 0.50 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Caroline planned to go do some research. | Caroline |
| 2 | 0.663 | 0.685 | 0.50 | 1:14 pm on 25 May, 2023 | Caroline is researching adoption agencies. | Caroline |
| 3 | 0.596 | 0.595 | 0.50 | 7:55 pm on 9 June, 2023 | Caroline's friends, family, and mentors motivate her and give her strength. | Caroline |
| 4 | 0.540 | 0.539 | 0.50 | 1:14 pm on 25 May, 2023 | Caroline is thrilled to make a family for kids who need one. | Caroline |
| 5 | 0.595 | 0.646 | 0.50 | 10:37 am on 27 June, 2023 | Caroline's own journey and the support she received motivated her to pursue counseling. | Caroline |
| 6 | 0.523 | 0.507 | 0.50 | 1:14 pm on 25 May, 2023 | Caroline is grateful for all the support she has got from friends and mentors. | Caroline |

---

## [Multi-hop] `Would Melanie be considered an ally to the transgender community?`

- base RRF (Phase A+B): **8** notes (phase_a_hits=69)
- after multi-hop + global re-rank: **13** notes (multi_hop_added=5, max_hops=2)

**Top retrieved (enhanced):**

| # | sim(e_q,e) | intent(e_q,z) | q | session_date | fact | entities |
|--:|--:|--:|--:|---|---|---|
| 1 | 0.484 | 0.507 | 0.50 | 10:37 am on 27 June, 2023 | Caroline is thinking of working with trans people to help them accept themselves and support their mental health. | Caroline |
| 2 | 0.381 | 0.382 | 0.50 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, the LGBTQ support group has made Caroline feel accepted and given her courage to embrace herself. | Caroline |
| 3 | 0.420 | 0.413 | 0.50 | 1:56 pm on 8 May, 2023 | On 7 May 2023, Caroline found the transgender stories at the LGBTQ support group inspiring. | Caroline |
| 4 | 0.404 | 0.417 | 0.50 | 7:55 pm on 9 June, 2023 | On 2 June 2023, Caroline spoke at her school event about her transgender journey and encouraged students to get involved | Caroline |
| 5 | 0.468 | 0.475 | 0.50 | 4:33 pm on 12 July, 2023 | Caroline feels thankful for the LGBTQ community and believes it is important to fight for trans rights and spread awaren | Caroline |
| 6 | 0.503 | 0.487 | 0.50 | 1:33 pm on 25 August, 2023 | Caroline decided to transition and join the transgender community to find a community where she is accepted, loved and s | Caroline |

---

## [Conversational] `What are Caroline's plans for the summer?`

- base RRF (Phase A+B): **8** notes (phase_a_hits=59)
- after multi-hop + global re-rank: **13** notes (multi_hop_added=5, max_hops=2)

**Top retrieved (enhanced):**

| # | sim(e_q,e) | intent(e_q,z) | q | session_date | fact | entities |
|--:|--:|--:|--:|---|---|---|
| 1 | 0.764 | 0.772 | 0.50 | 1:50 pm on 17 August, 2023 | Caroline proposed planning something special for the summer of 2023, just the two of them, to catch up and explore natur | Caroline, Melanie |
| 2 | 0.727 | 0.723 | 0.50 | 1:50 pm on 17 August, 2023 | Melanie agreed to plan something special with Caroline for the summer of 2023. | Melanie, Caroline |
| 3 | 0.636 | 0.661 | 0.50 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Caroline planned to go do some research. | Caroline |
| 4 | 0.670 | 0.705 | 0.50 | 1:50 pm on 17 August, 2023 | Melanie said she will start thinking about what she and Caroline can do for their summer 2023 trip. | Melanie, Caroline |
| 5 | 0.622 | 0.666 | 0.50 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, Caroline plans to check out career options. | Caroline |
| 6 | 0.619 | 0.618 | 0.50 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, Caroline plans to continue her education. | Caroline |
