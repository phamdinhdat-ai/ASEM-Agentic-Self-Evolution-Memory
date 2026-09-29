# ASEM-THG sample ingestion — demo

- backend: `langchain` / model `deepseek-flash` (config declared `deepseek-v4-flash` — stale, endpoint serves only deepseek-flash/deepseek-v4-pro)
- embedder: `sentence-transformers/all-MiniLM-L6-v2`
- conversation: `conv-26` | 19 sessions | 419 turns total
- ingesting first **3** session(s)

## Ingestion run

| # | session | date | turns | LLM source | facts | bank size |
|---|---|---|---:|---|---:|---:|
| 1 | session_1 | 1:56 pm on 8 May, 2023 | 18 | llm | 32 | 32 |
| 2 | session_2 | 1:14 pm on 25 May, 2023 | 17 | llm | 46 | 78 |
| 3 | session_3 | 7:55 pm on 9 June, 2023 | 23 | llm | 63 | 141 |

## Raw ASEM-THG extraction — session_1 (1:56 pm on 8 May, 2023)

Single LLM call -> atomic facts with `(subject, predicate, object)` triplets:

```json
[
  {
    "fact": "On 8 May 2023, Melanie is swamped with her kids and work.",
    "subject": "Melanie",
    "predicate": "is swamped with",
    "object": "her kids and work",
    "entities": [
      "Melanie"
    ],
    "keywords": [
      "swamped",
      "kids",
      "work"
    ],
    "tags": [
      "personal",
      "professional",
      "fact"
    ],
    "speaker": "Melanie"
  },
  {
    "fact": "On 7 May 2023, Caroline went to an LGBTQ support group.",
    "subject": "Caroline",
    "predicate": "went to",
    "object": "LGBTQ support group",
    "entities": [
      "Caroline",
      "LGBTQ support group"
    ],
    "keywords": [
      "LGBTQ",
      "support group"
    ],
    "tags": [
      "event",
      "personal",
      "fact"
    ],
    "speaker": "Caroline"
  },
  {
    "fact": "On 7 May 2023, Caroline found the LGBTQ support group powerful.",
    "subject": "Caroline",
    "predicate": "found powerful",
    "object": "LGBTQ support group",
    "entities": [
      "Caroline",
      "LGBTQ support group"
    ],
    "keywords": [
      "LGBTQ",
      "support group",
      "powerful"
    ],
    "tags": [
      "personal",
      "event",
      "fact"
    ],
    "speaker": "Caroline"
  },
  {
    "fact": "On 7 May 2023, Caroline found the transgender stories at the LGBTQ support group inspiring.",
    "subject": "Caroline",
    "predicate": "found inspiring",
    "object": "transgender stories at the LGBTQ support group",
    "entities": [
      "Caroline",
      "LGBTQ support group"
    ],
    "keywords": [
      "transgender",
      "stories",
      "inspiring"
    ],
    "tags": [
      "personal",
      "event",
      "fact"
    ],
    "speaker": "Caroline"
  },
  {
    "fact": "On 7 May 2023, Caroline felt happy and thankful for all the support at the LGBTQ support group.",
    "subject": "Caroline",
    "predicate": "felt happy and thankful for",
    "object": "support at the LGBTQ support group",
    "entities": [
      "Caroline",
      "LGBTQ support group"
    ],
    "keywords": [
      "happy",
      "thankful",
      "support"
    ],
    "tags": [
      "personal",
      "event",
      "fact"
    ],
    "speaker": "Caroline"
  },
  {
    "fact": "On 8 May 2023, Melanie loves the painting of a woman shown in Caroline's photo.",
    "subject": "Melanie",
    "predicate": "loves",
    "object": "painting of a woman in Caroline's photo",
    "entities": [
      "Melanie",
      "Caroline"
    ],
    "keywords": [
      "painting",
      "woman",
      "photo"
    ],
    "tags": [
      "preference",
      "fact"
    ],
    "speaker": "Melanie"
  },
  {
    "fact": "As of 8 May 2023, Caroline feels accepted because of her LGBTQ support group.",
    "subject": "Caroline",
    "predicate": "feels accepted because of",
    "object": "her LGBTQ support group",
    "entities": [
      "Caroline",
      "LGBTQ support group"
    ],
    "keywords": [
      "accepted",
      "support group"
    ],
    "tags": [
      "personal",
      "fact"
    ],
    "speaker": "Caroline"
  },
  {
    "fact": "As of 8 May 2023, Caroline has courage to embrace herself because of her LGBTQ support group.",
    "subject": "Caroline",
    "predicate": "has courage to embrace herself because of",
    "object": "her LGBTQ support group",
    "entities": [
      "Caroline",
      "LGBTQ support group"
    ],
    "keywords": [
      "courage",
      "embrace",
      "support group"
    ],
    "tags": [
      "personal",
      "fact"
    ],
    "speaker": "Caroline"
  }
]
```

## Persisted notes (bank: 141)

| # | date | fact | subject | predicate | object | entities | speaker |
|---:|---|---|---|---|---|---|---|
| 1 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, Caroline feels accepted because of her LGBTQ support group. | Caroline | feels accepted because of | her LGBTQ support group | Caroline, LGBTQ support group | Caroline |
| 2 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, Caroline has courage to embrace herself because of her LGBTQ support group. | Caroline | has courage to embrace herself because of | her LGBTQ support group | Caroline, LGBTQ support group | Caroline |
| 3 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, Caroline is a member of an LGBTQ support group. | Caroline | is a member of | LGBTQ support group | Caroline, LGBTQ support group | Caroline |
| 4 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, Caroline is excited about continuing her education and checking out career options. | Caroline | is excited about | continuing her education and checking out career options | Caroline | Caroline |
| 5 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, Caroline is keen on counseling. | Caroline | is keen on | counseling | Caroline | Caroline |
| 6 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, Caroline is keen on working in mental health. | Caroline | is keen on | working in mental health | Caroline | Caroline |
| 7 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, Caroline plans to check out career options. | Caroline | plans to check out | career options | Caroline | Caroline |
| 8 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, Caroline plans to continue her education. | Caroline | plans to continue | her education | Caroline | Caroline |
| 9 | 1:56 pm on 8 May, 2023 | As of 8 May 2023, Caroline would love to support people with similar issues. | Caroline | would love to support | people with similar issues | Caroline | Caroline |
| 10 | 1:56 pm on 8 May, 2023 | In 2022, Melanie painted a painting of a lake sunrise. | Melanie | painted | painting of a lake sunrise | Melanie | Melanie |
| 11 | 1:56 pm on 8 May, 2023 | On 7 May 2023, Caroline felt happy and thankful for all the support at the LGBTQ support group. | Caroline | felt happy and thankful for | support at the LGBTQ support group | Caroline, LGBTQ support group | Caroline |
| 12 | 1:56 pm on 8 May, 2023 | On 7 May 2023, Caroline found the LGBTQ support group powerful. | Caroline | found powerful | LGBTQ support group | Caroline, LGBTQ support group | Caroline |
| 13 | 1:56 pm on 8 May, 2023 | On 7 May 2023, Caroline found the transgender stories at the LGBTQ support group inspiring. | Caroline | found inspiring | transgender stories at the LGBTQ support group | Caroline, LGBTQ support group | Caroline |
| 14 | 1:56 pm on 8 May, 2023 | On 7 May 2023, Caroline went to an LGBTQ support group. | Caroline | went to | LGBTQ support group | Caroline, LGBTQ support group | Caroline |
| 15 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Caroline agrees that relaxing and expressing oneself is key. | Caroline | agrees | relaxing and expressing oneself is key | Caroline | Caroline |
| 16 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Caroline thinks Melanie's sharing of the painting is really sweet. | Caroline | thinks | Melanie's sharing of the painting is really sweet | Caroline, Melanie | Caroline |
| 17 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Caroline thinks painting looks like a great outlet for expressing oneself. | Caroline | thinks | painting looks like a great outlet for expressing oneself | Caroline | Caroline |
| 18 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Caroline thinks the colors in Melanie's painting blend nicely. | Caroline | thinks | colors in Melanie's painting blend nicely | Caroline, Melanie | Caroline |
| 19 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Caroline was about to do some research. | Caroline | was about to do | some research | Caroline | Caroline |
| 20 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie believes taking care of oneself is vital. | Melanie | believes | taking care of oneself is vital | Melanie | Melanie |
| 21 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie finds painting a fun way to express her feelings and get creative. | Melanie | finds | painting a fun way to express her feelings and get creative | Melanie | Melanie |
| 22 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie finds painting a great way to relax after a long day. | Melanie | finds | painting a great way to relax after a long day | Melanie | Melanie |
| 23 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie has children. | Melanie | has | children | Melanie | Melanie |
| 24 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie is swamped with her kids and work. | Melanie | is swamped with | her kids and work | Melanie | Melanie |
| 25 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie loves the painting of a woman shown in Caroline's photo. | Melanie | loves | painting of a woman in Caroline's photo | Melanie, Caroline | Melanie |
| 26 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie plans to talk to Caroline soon. | Melanie | plans to talk to | Caroline soon | Melanie, Caroline | Melanie |
| 27 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie shared a photo of a painting of a sunset over a lake. | Melanie | shared | photo of a painting of a sunset over a lake | Melanie | Melanie |
| 28 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie thinks Caroline has guts. | Melanie | thinks | Caroline has guts | Melanie, Caroline | Melanie |
| 29 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie thinks Caroline would be a great counselor. | Melanie | thinks | Caroline would be a great counselor | Melanie, Caroline | Melanie |
| 30 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie thinks Caroline's empathy and understanding will help people Caroline works with. | Melanie | thinks | Caroline's empathy and understanding will help people Caroline works with | Melanie, Caroline | Melanie |
| 31 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie was about to go swimming with her kids. | Melanie | was about to go swimming with | her kids | Melanie | Melanie |
| 32 | 1:56 pm on 8 May, 2023 | On 8 May 2023, Melanie's painting of a lake sunrise is special to Melanie. | Melanie | is special to | Melanie's painting of a lake sunrise | Melanie | Melanie |
| 33 | 1:14 pm on 25 May, 2023 | After the charity race on 20 May 2023, Melanie thought about taking care of minds. | Melanie | thought about | taking care of minds | Melanie | Melanie |
| 34 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Caroline chose an adoption agency that helps LGBTQ+ folks with adoption. | Caroline | chose | adoption agency that helps LGBTQ+ folks with adoption | Caroline, adoption agency, LGBTQ+ folks | Caroline |
| 35 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Caroline expects being a single parent will be tough. | Caroline | expects | being a single parent will be tough | Caroline | Caroline |
| 36 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Caroline has a dream to give a loving home to kids in need. | Caroline | has a dream | to give a loving home to kids in need | Caroline, kids | Caroline |
| 37 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Caroline has a dream to have a family. | Caroline | has a dream | to have a family | Caroline, family | Caroline |
| 38 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Caroline intends to make sure the kids have a safe and loving home. | Caroline | intends | to make sure the kids have a safe and loving home | Caroline, kids | Caroline |
| 39 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Caroline is excited to make a family for kids in need. | Caroline | is excited to make | a family for kids in need | Caroline, kids | Caroline |
| 40 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Caroline is grateful for support from Caroline's friends and mentors. | Caroline | is grateful for | support from Caroline's friends and mentors | Caroline, friends, mentors | Caroline |
| 41 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Caroline is researching adoption agencies. | Caroline | is researching | adoption agencies | Caroline, adoption agencies | Caroline |
| 42 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Caroline is starting hard work to turn Caroline's adoption dream into reality. | Caroline | is starting hard work | turning Caroline's adoption dream into reality | Caroline, adoption | Caroline |
| 43 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Caroline is up for the challenge of being a single parent. | Caroline | is up for | challenge of being a single parent | Caroline | Caroline |
| 44 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Caroline's goal is to give kids a loving home. | Caroline | has goal | to give kids a loving home | Caroline, kids | Caroline |
| 45 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Melanie carves out me-time each day. | Melanie | carves out | me-time each day | Melanie | Melanie |
| 46 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Melanie finds Melanie's me-time refreshing. | Melanie | finds refreshing | Melanie's me-time | Melanie | Melanie |
| 47 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Melanie plays violin as part of Melanie's daily me-time. | Melanie | plays violin as part of | Melanie's daily me-time | Melanie, violin | Melanie |
| 48 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Melanie reads as part of Melanie's daily me-time. | Melanie | reads as part of | Melanie's daily me-time | Melanie | Melanie |
| 49 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Melanie runs as part of Melanie's daily me-time. | Melanie | runs as part of | Melanie's daily me-time | Melanie | Melanie |
| 50 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Melanie's kids are excited about summer break in 2023. | Melanie's kids | are excited about | summer break in 2023 | Melanie's kids, summer break | Melanie |
| 51 | 1:14 pm on 25 May, 2023 | As of 25 May 2023, Melanie's me-time helps Melanie stay present for Melanie's family. | Melanie's me-time | helps stay present for | Melanie's family | Melanie, family | Melanie |
| 52 | 1:14 pm on 25 May, 2023 | In June 2023, Melanie and Melanie's family are thinking about going camping. | Melanie | is thinking about going camping with family | camping in June 2023 | Melanie, family, June 2023 | Melanie |
| 53 | 1:14 pm on 25 May, 2023 | On 20 May 2023, Melanie found the charity race rewarding. | Melanie | found rewarding | charity race for mental health on 20 May 2023 | Melanie, charity race | Melanie |
| 54 | 1:14 pm on 25 May, 2023 | On 20 May 2023, Melanie found the charity race thought-provoking. | Melanie | found thought-provoking | charity race for mental health on 20 May 2023 | Melanie, charity race | Melanie |
| 55 | 1:14 pm on 25 May, 2023 | On 20 May 2023, Melanie ran a charity race for mental health. | Melanie | ran | charity race for mental health | Melanie, charity race, mental health | Melanie |
| 56 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Caroline agrees that personal self-care is important. | Caroline | agrees | personal self-care is important | Caroline, self-care | Caroline |
| 57 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Caroline believes making a difference and raising awareness for mental health is rewarding. | Caroline | believes | making a difference and raising awareness for mental health is rewarding | Caroline, mental health | Caroline |
| 58 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Caroline is feeling hopeful and optimistic about the adoption process. | Caroline | is feeling | hopeful and optimistic about the adoption process | Caroline, adoption process | Caroline |
| 59 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Caroline is proud of Melanie for taking part in the charity race. | Caroline | is proud of | Melanie for taking part in the charity race | Caroline, Melanie, charity race | Caroline |
| 60 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Caroline says Melanie's kind words mean a lot to Caroline. | Caroline | says | Melanie's kind words mean a lot to Caroline | Caroline, Melanie | Caroline |
| 61 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Caroline shared a photo of an adoption agency she is looking into, showing a sign for a new ar | Caroline | shared photo of | adoption agency she is looking into, showing a sign for a new arrival and an information and domestic building | Caroline, adoption agency | Caroline |
| 62 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Caroline thinks Melanie is doing an awesome job looking after Melanie and Melanie's family. | Caroline | thinks | Melanie is doing an awesome job looking after Melanie and Melanie's family | Caroline, Melanie, family | Caroline |
| 63 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Caroline thinks Melanie is prioritizing self-care. | Caroline | thinks | Melanie is prioritizing self-care | Caroline, Melanie | Caroline |
| 64 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Caroline thinks taking personal time is important. | Caroline | thinks | taking personal time is important | Caroline | Caroline |
| 65 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Caroline values the adoption agency's inclusivity and support. | Caroline | values | adoption agency's inclusivity and support | Caroline, adoption agency, LGBTQ+ folks | Caroline |
| 66 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie appreciates the adoption agency's inclusivity and support. | Melanie | appreciates | adoption agency's inclusivity and support | Melanie, adoption agency | Melanie |
| 67 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie believes Caroline's adopted kids will get all the love and stability the kids need. | Melanie | believes | Caroline's adopted kids will get all the love and stability the kids need | Melanie, Caroline, kids | Melanie |
| 68 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie believes looking after Melanie helps Melanie better look after Melanie's family. | Melanie | believes | looking after Melanie helps Melanie better look after Melanie's family | Melanie, family | Melanie |
| 69 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie considers self-care a journey for Melanie. | Melanie | considers | self-care a journey for Melanie | Melanie, self-care | Melanie |
| 70 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie is doing Melanie's best with self-care. | Melanie | is doing best with | self-care | Melanie, self-care | Melanie |
| 71 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie is excited for Caroline's new chapter. | Melanie | is excited for | Caroline's new chapter | Melanie, Caroline | Melanie |
| 72 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie is starting to realize that self-care is important. | Melanie | is starting to realize | self-care is important | Melanie, self-care | Melanie |
| 73 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie says Melanie's self-care is still a work in progress. | Melanie | says | Melanie's self-care is still a work in progress | Melanie, self-care | Melanie |
| 74 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie thinks Caroline has a caring heart. | Melanie | thinks | Caroline has a caring heart | Melanie, Caroline | Melanie |
| 75 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie thinks Caroline is kind for taking in kids in need. | Melanie | thinks | Caroline is kind for taking in kids in need | Melanie, Caroline, kids | Melanie |
| 76 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie thinks Caroline will be an awesome mom. | Melanie | thinks | Caroline will be an awesome mom | Melanie, Caroline | Melanie |
| 77 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie thinks Caroline's future family will be lucky to have Caroline. | Melanie | thinks | Caroline's future family will be lucky to have Caroline | Melanie, Caroline, future family | Melanie |
| 78 | 1:14 pm on 25 May, 2023 | On 25 May 2023, Melanie thinks the adoption agency Caroline shared looks great. | Melanie | thinks | the adoption agency Caroline shared looks great | Melanie, adoption agency | Melanie |
| 79 | 7:55 pm on 9 June, 2023 | As of 9 June 2023, Caroline has known these friends for 4 years, since she moved from her home country. | Caroline | has known for 4 years | these friends | Caroline, friends | Caroline |
| 80 | 7:55 pm on 9 June, 2023 | As of 9 June 2023, Melanie has been married for 5 years. | Melanie | has been married for | 5 years | Melanie, husband | Melanie |
| 81 | 7:55 pm on 9 June, 2023 | At the school event, Caroline encouraged students to get involved in the LGBTQ community. | Caroline | encouraged | students to get involved in the LGBTQ community | Caroline, students, LGBTQ community, school | Caroline |
| 82 | 7:55 pm on 9 June, 2023 | At the school event, Caroline talked about her transgender journey. | Caroline | talked about | her transgender journey | Caroline, school | Caroline |
| 83 | 7:55 pm on 9 June, 2023 | Before 9 June 2023, Melanie had a fun family day where they played games, ate good food, and hung out together | Melanie | had | a fun family day where they played games, ate good food, and hung out together | Melanie, family | Melanie |
| 84 | 7:55 pm on 9 June, 2023 | Caroline experienced a tough breakup before 9 June 2023. | Caroline | experienced | a tough breakup before 9 June 2023 | Caroline | Caroline |
| 85 | 7:55 pm on 9 June, 2023 | Caroline had a school event in the week before 9 June 2023. | Caroline | had | school event in the week before 9 June 2023 | Caroline, Melanie, school | Caroline |
| 86 | 7:55 pm on 9 June, 2023 | Caroline moved from her home country around June 2019, four years before 9 June 2023. | Caroline | moved from | her home country around June 2019 | Caroline, home country | Caroline |
| 87 | 7:55 pm on 9 June, 2023 | Caroline saw students' reactions at her school event. | Caroline | saw | students' reactions | Caroline, students, school | Caroline |
| 88 | 7:55 pm on 9 June, 2023 | Caroline started transitioning around June 2020, three years before 9 June 2023. | Caroline | started transitioning | around June 2020 | Caroline | Caroline |
| 89 | 7:55 pm on 9 June, 2023 | In the week before 9 June 2023, Caroline met up with her loved ones and took a photo in a yard. | Caroline | met up with | her loved ones and took a photo in a yard | Caroline, family, friends | Caroline |
| 90 | 7:55 pm on 9 June, 2023 | In the week before 9 June 2023, Caroline's talk inspired audience members to become better allies. | Caroline | inspired | audience members to become better allies | Caroline, audience | Caroline |
| 91 | 7:55 pm on 9 June, 2023 | Melanie got married around June 2018, five years before 9 June 2023. | Melanie | got married | around June 2018 | Melanie, husband | Melanie |
| 92 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline congratulated Melanie on her marriage. | Caroline | congratulated | Melanie on her marriage | Caroline, Melanie | Caroline |
| 93 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline expressed intention to pass on the love and support she received to others. | Caroline | expressed intention to | pass on the love and support she received to others | Caroline | Caroline |
| 94 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline felt powerful giving her talk about her transgender journey. | Caroline | felt powerful | giving her talk about her transgender journey | Caroline | Caroline |
| 95 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline felt thankful for being able to give a voice to the trans community. | Caroline | felt thankful for | being able to give a voice to the trans community | Caroline, trans community | Caroline |
| 96 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline reflected on how far she had come since starting her transition. | Caroline | reflected on | how far she had come since starting her transition | Caroline | Caroline |
| 97 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said Melanie and her husband looked great on their wedding day. | Caroline | said | Melanie and her husband looked great on their wedding day | Caroline, Melanie, husband | Caroline |
| 98 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said Melanie is part of her effort to make change. | Caroline | said | Melanie is part of her effort to make change | Caroline, Melanie | Caroline |
| 99 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said Melanie's backing meant a lot to her. | Caroline | said | Melanie's backing meant a lot to her | Caroline, Melanie | Caroline |
| 100 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said conversations about gender identity and inclusion are necessary. | Caroline | said | conversations about gender identity and inclusion are necessary | Caroline | Caroline |
| 101 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said family is everything. | Caroline | said | family is everything | Caroline, family | Caroline |
| 102 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said hanging with loved ones is amazing and brings happiness. | Caroline | said | hanging with loved ones is amazing and brings happiness | Caroline, loved ones | Caroline |
| 103 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said her friends have been there for her through everything. | Caroline | said | her friends have been there for her through everything | Caroline, friends | Caroline |
| 104 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said her friends' love and help were important especially after her tough breakup. | Caroline | said | her friends' love and help were important especially after her tough breakup | Caroline, friends | Caroline |
| 105 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said her friends, family, and mentors are her rocks. | Caroline | said | her friends, family, and mentors are her rocks | Caroline, friends, family, mentors | Caroline |
| 106 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said her friends, family, and mentors motivate her and give her strength. | Caroline | said | her friends, family, and mentors motivate her and give her strength | Caroline, friends, family, mentors | Caroline |
| 107 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said it looked like Melanie had a great day. | Caroline | said | it looked like Melanie had a great day | Caroline, Melanie | Caroline |
| 108 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said sharing experiences is important to promote understanding and acceptance. | Caroline | said | sharing experiences is important to promote understanding and acceptance | Caroline | Caroline |
| 109 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said sharing stories can build a strong, supportive community of hope. | Caroline | said | sharing stories can build a strong, supportive community of hope | Caroline | Caroline |
| 110 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said she and Melanie can spread love and understanding. | Caroline | said | she and Melanie can spread love and understanding | Caroline, Melanie | Caroline |
| 111 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said she and Melanie can tackle life's challenges together. | Caroline | said | she and Melanie can tackle life's challenges together | Caroline, Melanie | Caroline |
| 112 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said she feels lucky to have her support system. | Caroline | said | she feels lucky to have her support system | Caroline, friends, family, mentors | Caroline |
| 113 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said she has been blessed with love and support throughout her journey. | Caroline | said | she has been blessed with love and support throughout her journey | Caroline | Caroline |
| 114 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said she is grateful for the chance to share her story and give others hope. | Caroline | said | she is grateful for the chance to share her story and give others hope | Caroline | Caroline |
| 115 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said she shared her journey, struggles, and growth since coming out. | Caroline | shared | her journey, struggles, and growth since coming out | Caroline | Caroline |
| 116 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said she will keep using her voice to make a change and lift others up. | Caroline | said she will | keep using her voice to make a change and lift others up | Caroline | Caroline |
| 117 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline said those moments make her thankful. | Caroline | said | those moments make her thankful | Caroline | Caroline |
| 118 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline told Melanie to cherish moments. | Caroline | told | Melanie to cherish moments | Caroline, Melanie | Caroline |
| 119 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Caroline wished Melanie many happy years with her husband. | Caroline | wished | Melanie many happy years with her husband | Caroline, Melanie, husband | Caroline |
| 120 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie called Caroline brave for speaking up for the trans community. | Melanie | called | Caroline brave for speaking up for the trans community | Melanie, Caroline, trans community | Melanie |
| 121 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie proposed that she and Caroline keep motivating and helping each other. | Melanie | proposed | she and Caroline keep motivating and helping each other | Melanie, Caroline | Melanie |
| 122 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said Caroline was doing an awesome job inspiring others. | Melanie | said | Caroline was doing an awesome job inspiring others | Melanie, Caroline | Melanie |
| 123 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said Caroline's courage is inspiring. | Melanie | said | Caroline's courage is inspiring | Melanie, Caroline | Melanie |
| 124 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said family and moments make it all worth it. | Melanie | said | family and moments make it all worth it | Melanie, family | Melanie |
| 125 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said family moments make life awesome. | Melanie | said | family moments make life awesome | Melanie, family | Melanie |
| 126 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said her family motivates her and gives her love. | Melanie | said | her family motivates her and gives her love | Melanie, family | Melanie |
| 127 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said her husband and kids keep her motivated. | Melanie | said | her husband and kids keep her motivated | Melanie, husband, kids | Melanie |
| 128 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said sharing different paths shows people they are not alone. | Melanie | said | sharing different paths shows people they are not alone | Melanie | Melanie |
| 129 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said she and Caroline can make a real impact together. | Melanie | said | she and Caroline can make a real impact together | Melanie, Caroline | Melanie |
| 130 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said she cherishes time with family. | Melanie | said | she cherishes time with family | Melanie, family | Melanie |
| 131 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said she feels alive and happy when with family. | Melanie | said | she feels alive and happy when with family | Melanie, family | Melanie |
| 132 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said she is looking forward to more happy years with her husband and family. | Melanie | said she is looking forward to | more happy years with her husband and family | Melanie, husband, family | Melanie |
| 133 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said she is lucky to have her husband and kids. | Melanie | said | she is lucky to have her husband and kids | Melanie, husband, kids | Melanie |
| 134 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said she is proud to be part of the difference Caroline is making. | Melanie | said | she is proud to be part of the difference Caroline is making | Melanie, Caroline | Melanie |
| 135 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said she wants to be courageous for her family. | Melanie | said she wants to be | courageous for her family | Melanie, family | Melanie |
| 136 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said she was proud of Caroline for spreading LGBTQ awareness. | Melanie | was proud of | Caroline for spreading LGBTQ awareness | Melanie, Caroline, LGBTQ community | Melanie |
| 137 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said talking about inclusivity and acceptance is crucial. | Melanie | said | talking about inclusivity and acceptance is crucial | Melanie | Melanie |
| 138 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie said vulnerable moments help people bond and understand each other. | Melanie | said | vulnerable moments help people bond and understand each other | Melanie | Melanie |
| 139 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie shared a photo of a man and a little girl in front of a waterfall. | Melanie | shared | a photo of a man and a little girl in front of a waterfall | Melanie, husband, kids | Melanie |
| 140 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie shared a photo of a man and woman sitting on a blanket eating food. | Melanie | shared | a photo of a man and woman sitting on a blanket eating food | Melanie, husband | Melanie |
| 141 | 7:55 pm on 9 June, 2023 | On 9 June 2023, Melanie shared a photo of herself as a bride in a wedding dress holding a bouquet. | Melanie | shared | a photo of herself as a bride in a wedding dress holding a bouquet | Melanie, husband | Melanie |

## Hyper-graph edges

| relation | directed link records | undirected (÷2) |
|---|---:|---:|
| same-entity | 1368 | 684 |
| semantic | 1030 | 515 |
| temporal | 276 | 138 |
| superseded_by | 104 | 52 |
| **total** | **2778** | **1389** |

### Graph-level stats

- fact nodes in hyper-graph: 141
- entity nodes: 198
- networkx graph: 339 nodes, 1341 edges
- subject/predicate versioning index entries: 89

### Example `superseded_by` (temporal versioning, zero LLM)

- `As of 8 May 2023, Caroline is keen on counseling.`
  ← superseded_by → `As of 8 May 2023, Caroline is keen on working in mental health.`
- `On 8 May 2023, Caroline agrees that relaxing and expressing oneself is key.`
  ← superseded_by → `On 25 May 2023, Caroline agrees that personal self-care is important.`
- `On 8 May 2023, Caroline thinks Melanie's sharing of the painting is really sweet`
  ← superseded_by → `On 8 May 2023, Caroline thinks the colors in Melanie's painting blend nicely.`
- `On 8 May 2023, Caroline thinks painting looks like a great outlet for expressing`
  ← superseded_by → `On 8 May 2023, Caroline thinks the colors in Melanie's painting blend nicely.`
- `On 8 May 2023, Caroline thinks painting looks like a great outlet for expressing`
  ← superseded_by → `On 25 May 2023, Caroline thinks Melanie is prioritizing self-care.`

### Entity nodes (sample)

`2022`, `4 years`, `5 years`, `acceptance`, `accepted`, `adoption`, `adoption agencies`, `adoption agency`, `adoption process`, `agreement`, `alive`, `allies`, `audience`, `awareness`, `backing`, `blanket`, `blend`, `bonding`, `brave`, `breakup`, `bride`, `camping`, `career`, `caring heart`, `caroline`, `challenge`, `challenges`, `change`, `charity race`, `cherish`, `children`, `colors`, `coming out`, `community`, `compliment`, `congratulations`, `counseling`, `counselor`, `courage`, `courageous`
