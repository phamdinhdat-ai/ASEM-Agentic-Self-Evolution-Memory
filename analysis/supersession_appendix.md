# Supersession appendix — every `superseded_by` pair

Source: `static/memory_banks/locomo10/thg_v2/ASEM-THG`  |  pairs: **166**

Counting rule: one row per **unique undirected pair**. Each pair is stored
on BOTH endpoint notes, so the stored `superseded_by` link-record count is
2x this number (166 pairs = 332 records across the 10 conversations).

`semantic_verdict` is `unread` for every row by design — deciding whether two
facts are genuine versions of one another needs semantic judgement, not a
script. The script-classifiable columns ARE filled.

- same/cross-session: {'cross': 137, 'same': 29}
- interval class: {'older_closed_newer_open': 137, 'inverted': 18, 'zero_width': 11}
- shared head verb: {True: 56, False: 110}

## locomo_0000 — a9c6eafb → 3683f17f

- **cross-session**; older `s8` (1:51 pm on 15 July, 2023) → newer `s12` (1:50 pm on 17 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`appreciates` newer=`appreciates` shared=True
- OLDER: Melanie appreciates Caroline's friendship and says it is great to have a supporter.
  - K=['friendship', 'support', 'appreciation'] entities=['Melanie', 'Caroline']
- NEWER: Melanie said she appreciates her friendship with Caroline and that Caroline has always been there for her.
  - K=['friendship', 'appreciation', 'support'] entities=['Melanie', 'Caroline']
- semantic verdict: **unread**

## locomo_0000 — 00cb74d3 → 9cfb0e3f

- **cross-session**; older `s1` (1:56 pm on 8 May, 2023) → newer `s4` (10:37 am on 27 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`reacted` newer=`gorgeous` shared=False
- OLDER: Melanie reacted to Caroline's necklace, saying it was gorgeous and that such items can mean so much; Melanie's own grandmother gave her a necklace from Sweden.
  - K=['necklace', 'grandmother', 'Sweden', 'sentiment'] entities=['Melanie', 'Caroline', 'Sweden']
- NEWER: Melanie reacted to Caroline's necklace, saying it is gorgeous and that items can mean a lot to us.
  - K=['necklace', 'gorgeous', 'items', 'meaning'] entities=['Melanie', 'Caroline']
- semantic verdict: **unread**

## locomo_0000 — bfa7f428 → e0e77d59

- **same-session**; older `s13` (3:31 pm on 23 August, 2023) → newer `s13` (3:31 pm on 23 August, 2023)
- interval class: `inverted`
- head verbs: older=`named` newer=`pets` shared=False
- OLDER: Melanie has a pet named Oliver.
  - K=['Oliver', 'pet'] entities=['Melanie', 'Oliver']
- NEWER: Melanie's pets are doing well and she recently got another cat named Bailey.
  - K=['Bailey', 'cat', 'pet'] entities=['Melanie', 'Bailey']
- semantic verdict: **unread**

## locomo_0000 — 63ba2eb3 → e0e77d59

- **cross-session**; older `s7` (4:33 pm on 12 July, 2023) → newer `s13` (3:31 pm on 23 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`pets` newer=`pets` shared=True
- OLDER: Melanie has a dog and a cat, which brighten up her day and always make her smile.
  - K=['dog', 'cat', 'pets'] entities=['Melanie']
- NEWER: Melanie's pets are doing well and she recently got another cat named Bailey.
  - K=['Bailey', 'cat', 'pet'] entities=['Melanie', 'Bailey']
- semantic verdict: **unread**

## locomo_0000 — c47a1be6 → 09f44432

- **cross-session**; older `s10` (8:56 pm on 20 July, 2023) → newer `s18` (6:55 pm on 20 October, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`says` newer=`values` shared=False
- OLDER: Melanie says she is really lucky to have her family, who bring her so much joy and love.
  - K=['family', 'joy', 'love', 'gratitude'] entities=['Melanie']
- NEWER: After the accident, Melanie reflected on how much she needs her family, saying they mean the world to her.
  - K=['family', 'values', 'accident', 'thankful'] entities=['Melanie']
- semantic verdict: **unread**

## locomo_0001 — 689b345b → 79de5cb3

- **same-session**; older `s15` (10:04 am on 19 June, 2023) → newer `s15` (10:04 am on 19 June, 2023)
- interval class: `inverted`
- head verbs: older=`grand` newer=`memories` shared=False
- OLDER: Gina said she cannot wait for the grand opening of Jon's dance studio on 20 June 2023.
  - K=['grand opening', 'dance studio'] entities=['Gina', 'Jon', 'dance studio']
- NEWER: Gina said she will be right by Jon's side and that they will make great memories on 20 June 2023.
  - K=['support', 'grand opening', 'memories'] entities=['Gina', 'Jon', 'dance studio']
- semantic verdict: **unread**

## locomo_0001 — 09029e9e → b0c7ed8c

- **same-session**; older `s15` (10:04 am on 19 June, 2023) → newer `s15` (10:04 am on 19 June, 2023)
- interval class: `inverted`
- head verbs: older=`memories` newer=`congrats` shared=False
- OLDER: Gina said she cannot wait to make more memories at Jon's dance studio.
  - K=['memories', 'dance studio'] entities=['Gina', 'Jon', 'dance studio']
- NEWER: Gina congratulated Jon, saying his dance studio looks amazing and that she is pumped for the launch on 20 June 2023.
  - K=['congrats', 'dance studio', 'launch'] entities=['Gina', 'Jon', 'dance studio']
- semantic verdict: **unread**

## locomo_0001 — 09029e9e → 46a77f28

- **same-session**; older `s15` (10:04 am on 19 June, 2023) → newer `s15` (10:04 am on 19 June, 2023)
- interval class: `zero_width`
- head verbs: older=`memories` newer=`friends` shared=False
- OLDER: Gina said she cannot wait to make more memories at Jon's dance studio.
  - K=['memories', 'dance studio'] entities=['Gina', 'Jon', 'dance studio']
- NEWER: Gina said she loves being around friends and having a great time, and cannot wait to have fun at Jon's dance studio.
  - K=['friends', 'fun', 'dance studio'] entities=['Gina', 'Jon', 'dance studio']
- semantic verdict: **unread**

## locomo_0001 — db7c3a77 → 78405b2f

- **cross-session**; older `s9` (10:33 am on 9 April, 2023) → newer `s13` (8:29 pm on 13 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`thanked` newer=`thanks` shared=False
- OLDER: Jon thanked Gina for her support, saying it means a lot to him.
  - K=['gratitude', 'support'] entities=['Jon', 'Gina']
- NEWER: Jon thanked Gina for being there for him and believing in him, saying it means a lot.
  - K=['thanks', 'support', 'belief'] entities=['Jon', 'Gina']
- semantic verdict: **unread**

## locomo_0001 — 5d5df5b4 → 64061c85

- **cross-session**; older `s9` (10:33 am on 9 April, 2023) → newer `s16` (2:15 pm on 21 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`loss` newer=`loss` shared=True
- OLDER: Jon lost his job, which pushed him to pursue his love of dance.
  - K=['job loss', 'dance', 'motivation'] entities=['Jon']
- NEWER: Jon lost his job, which has made things tough for him.
  - K=['job loss', 'tough', 'business'] entities=['Jon']
- semantic verdict: **unread**

## locomo_0001 — 3788ecc2 → 77e8a3d4

- **cross-session**; older `s9` (10:33 am on 9 April, 2023) → newer `s19` (6:46 pm on 23 July, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`cheered` newer=`encouraged` shared=False
- OLDER: Gina cheered Jon on, telling him he is almost there and to keep it up.
  - K=['encouragement', 'support'] entities=['Gina', 'Jon']
- NEWER: Gina encouraged Jon on 23 July 2023 with a 'Just do it' motivational message attributed to Shia LaBeouf.
  - K=['encouragement', 'Just do it', 'Shia LaBeouf'] entities=['Gina', 'Jon', 'Shia LaBeouf']
- semantic verdict: **unread**

## locomo_0001 — 97fb54b0 → be48b4f9

- **same-session**; older `s4` (10:43 am on 4 February, 2023) → newer `s4` (10:43 am on 4 February, 2023)
- interval class: `inverted`
- head verbs: older=`tells` newer=`encourages` shared=False
- OLDER: Gina tells Jon he will kill it in the competition and that his hard work and talent will pay off, wishing him good luck.
  - K=['competition', 'good luck', 'encouragement'] entities=['Gina', 'Jon']
- NEWER: Gina encourages Jon to hang in there, saying he will find the perfect dance studio soon.
  - K=['dance studio', 'encouragement'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — 877b627b → 97fb54b0

- **same-session**; older `s4` (10:43 am on 4 February, 2023) → newer `s4` (10:43 am on 4 February, 2023)
- interval class: `inverted`
- head verbs: older=`tells` newer=`tells` shared=True
- OLDER: Gina tells Jon he has it, praising his passion and commitment and urging him to keep it up.
  - K=['passion', 'commitment', 'encouragement'] entities=['Gina', 'Jon']
- NEWER: Gina tells Jon he will kill it in the competition and that his hard work and talent will pay off, wishing him good luck.
  - K=['competition', 'good luck', 'encouragement'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — 71b5a1c1 → 86853d5d

- **same-session**; older `s4` (10:43 am on 4 February, 2023) → newer `s4` (10:43 am on 4 February, 2023)
- interval class: `zero_width`
- head verbs: older=`setbacks` newer=`dreams` shared=False
- OLDER: Gina tells Jon that setbacks are opportunities for comebacks and that he has the skills, passion, and drive, plus her full support.
  - K=['setbacks', 'support', 'encouragement'] entities=['Gina', 'Jon']
- NEWER: Gina tells Jon she is always there to cheer him on, urging him to make his dreams happen, keep going, and believe in himself.
  - K=['support', 'dreams', 'encouragement'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — 6a9519a9 → 1dbe9199

- **cross-session**; older `s4` (10:43 am on 4 February, 2023) → newer `s5` (9:32 am on 8 February, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`says` newer=`loves` shared=False
- OLDER: Jon says dancing is his passion and brings him joy and fulfillment.
  - K=['dance', 'passion', 'joy'] entities=['Jon']
- NEWER: Jon loves dancing because it brings him joy and reminds him why he is passionate about it.
  - K=['dancing', 'joy', 'passion'] entities=['Jon']
- semantic verdict: **unread**

## locomo_0001 — 02e10d03 → be48b4f9

- **same-session**; older `s4` (10:43 am on 4 February, 2023) → newer `s4` (10:43 am on 4 February, 2023)
- interval class: `zero_width`
- head verbs: older=`encourages` newer=`encourages` shared=True
- OLDER: Gina encourages Jon, telling him his drive will help him make a splash.
  - K=['encouragement', 'drive', 'support'] entities=['Gina', 'Jon']
- NEWER: Gina encourages Jon to hang in there, saying he will find the perfect dance studio soon.
  - K=['dance studio', 'encouragement'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — 67f6bed2 → 5eeb4702

- **cross-session**; older `s4` (10:43 am on 4 February, 2023) → newer `s15` (10:04 am on 19 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`business` newer=`business` shared=True
- OLDER: Jon is working hard on his business despite obstacles and is determined to make it succeed.
  - K=['business', 'work', 'determination'] entities=['Jon']
- NEWER: Jon is still working on his business.
  - K=['business', 'work'] entities=['Jon']
- semantic verdict: **unread**

## locomo_0001 — 9a4c23d8 → ac8e6046

- **same-session**; older `s4` (10:43 am on 4 February, 2023) → newer `s4` (10:43 am on 4 February, 2023)
- interval class: `inverted`
- head verbs: older=`thanks` newer=`dreams` shared=False
- OLDER: Jon thanks Gina, saying her kind words mean a lot and that he won't give up.
  - K=['gratitude', 'persistence'] entities=['Jon', 'Gina']
- NEWER: Jon thanks Gina, saying her help means a lot to him and that he will reach his dreams by pushing on.
  - K=['gratitude', 'support', 'dreams'] entities=['Jon', 'Gina']
- semantic verdict: **unread**

## locomo_0001 — 9a4c23d8 → 20a51eb9

- **cross-session**; older `s4` (10:43 am on 4 February, 2023) → newer `s5` (9:32 am on 8 February, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`thanks` newer=`thanks` shared=True
- OLDER: Jon thanks Gina, saying her kind words mean a lot and that he won't give up.
  - K=['gratitude', 'persistence'] entities=['Jon', 'Gina']
- NEWER: Jon thanked Gina, saying her help means a lot and he will keep plugging away and stay optimistic.
  - K=['thanks', 'optimism', 'support'] entities=['Jon', 'Gina']
- semantic verdict: **unread**

## locomo_0001 — 86853d5d → 06968753

- **cross-session**; older `s4` (10:43 am on 4 February, 2023) → newer `s19` (6:46 pm on 23 July, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`dreams` newer=`support` shared=False
- OLDER: Gina tells Jon she is always there to cheer him on, urging him to make his dreams happen, keep going, and believe in himself.
  - K=['support', 'dreams', 'encouragement'] entities=['Gina', 'Jon']
- NEWER: On 23 July 2023, Gina told Jon she is there to support him and that every step is getting him closer to his dream.
  - K=['support', 'dream', 'encouragement'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — 877b627b → 91475068

- **cross-session**; older `s4` (10:43 am on 4 February, 2023) → newer `s5` (9:32 am on 8 February, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`tells` newer=`encouraged` shared=False
- OLDER: Gina tells Jon he has it, praising his passion and commitment and urging him to keep it up.
  - K=['passion', 'commitment', 'encouragement'] entities=['Gina', 'Jon']
- NEWER: Gina encouraged Jon, saying it's tough starting a business but he can make his studio work, and that she is always there for him.
  - K=['encouragement', 'studio', 'support'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — d65332ea → b55dc1d3

- **cross-session**; older `s3` (12:48 am on 1 February, 2023) → newer `s8` (1:26 pm on 3 April, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`always` newer=`always` shared=True
- OLDER: Jon told Gina he is always there to support her and encouraged her to keep up her store work.
  - K=['support', 'encouragement', 'store'] entities=['Jon', 'Gina']
- NEWER: Jon told Gina he is always there for her and wished her good luck with her store.
  - K=['support', 'encouragement', 'store'] entities=['Jon', 'Gina']
- semantic verdict: **unread**

## locomo_0001 — 014f61db → 02e10d03

- **cross-session**; older `s3` (12:48 am on 1 February, 2023) → newer `s4` (10:43 am on 4 February, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`encouraged` newer=`encourages` shared=False
- OLDER: Gina encouraged Jon to keep going with his dance studio and stay positive, saying they'll get through this.
  - K=['encouragement', 'dance studio', 'positivity'] entities=['Gina', 'Jon']
- NEWER: Gina encourages Jon, telling him his drive will help him make a splash.
  - K=['encouragement', 'drive', 'support'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — f2b275dd → 3788ecc2

- **cross-session**; older `s8` (1:26 pm on 3 April, 2023) → newer `s9` (10:33 am on 9 April, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`goals` newer=`cheered` shared=False
- OLDER: Gina said Jon's support means a lot and encouraged him to keep pursuing his goals.
  - K=['support', 'encouragement', 'goals'] entities=['Gina', 'Jon']
- NEWER: Gina cheered Jon on, telling him he is almost there and to keep it up.
  - K=['encouragement', 'support'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — 64061c85 → 991e577d

- **cross-session**; older `s16` (2:15 pm on 21 June, 2023) → newer `s17` (1:25 pm on 9 July, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`loss` newer=`loss` shared=True
- OLDER: Jon lost his job, which has made things tough for him.
  - K=['job loss', 'tough', 'business'] entities=['Jon']
- NEWER: Jon lost his job.
  - K=['job loss', 'push', 'plunge'] entities=['Jon']
- semantic verdict: **unread**

## locomo_0001 — 93ca19a4 → 2e10090c

- **cross-session**; older `s16` (2:15 pm on 21 June, 2023) → newer `s17` (1:25 pm on 9 July, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`offered` newer=`offered` shared=True
- OLDER: Gina offered Jon support and told him to let her know if he needs anything.
  - K=['support', 'help', 'encouragement'] entities=['Gina', 'Jon']
- NEWER: Gina offered to always be there to cheer Jon on as of 9 July 2023.
  - K=['support', 'cheer', 'friendship'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — af3bf69b → 2fcf1e70

- **cross-session**; older `s2` (2:32 pm on 29 January, 2023) → newer `s9` (10:33 am on 9 April, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`business` newer=`reminds` shared=False
- OLDER: Gina owns a clothing store.
  - K=['clothing store', 'business ownership'] entities=['Gina']
- NEWER: Gina has a trophy from a dance contest that reminds her of the hard work, dedication and joy dance brings.
  - K=['trophy', 'dance contest', 'achievement'] entities=['Gina']
- semantic verdict: **unread**

## locomo_0001 — abdd8c33 → 014f61db

- **cross-session**; older `s2` (2:32 pm on 29 January, 2023) → newer `s3` (12:48 am on 1 February, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`destress` newer=`encouraged` shared=False
- OLDER: Gina encouraged Jon to believe in himself, push through the tough process, and take breaks to destress.
  - K=['encouragement', 'support', 'destress'] entities=['Gina', 'Jon']
- NEWER: Gina encouraged Jon to keep going with his dance studio and stay positive, saying they'll get through this.
  - K=['encouragement', 'dance studio', 'positivity'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — ce002a22 → db7c3a77

- **cross-session**; older `s6` (2:35 pm on 16 March, 2023) → newer `s9` (10:33 am on 9 April, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`thanks` newer=`thanked` shared=False
- OLDER: Jon thanked Gina for having his back and toasted to taking on new heights and all the trials that come with it.
  - K=['thanks', 'support', 'toast'] entities=['Jon', 'Gina']
- NEWER: Jon thanked Gina for her support, saying it means a lot to him.
  - K=['gratitude', 'support'] entities=['Jon', 'Gina']
- semantic verdict: **unread**

## locomo_0001 — 69fe5f69 → 4adab50c

- **cross-session**; older `s6` (2:35 pm on 16 March, 2023) → newer `s13` (8:29 pm on 13 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`paths` newer=`always` shared=False
- OLDER: Gina said that she and Jon are on different paths but it is nice to have someone to root for them.
  - K=['support', 'different paths'] entities=['Gina', 'Jon']
- NEWER: Gina told Jon she is always around for him.
  - K=['support', 'availability'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — 0afb02fa → 93ca19a4

- **cross-session**; older `s11` (3:14 pm on 11 May, 2023) → newer `s16` (2:15 pm on 21 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`tells` newer=`offered` shared=False
- OLDER: Gina tells Jon she is always here to help and cannot wait to see the dance routine video.
  - K=['support', 'dance routine video'] entities=['Gina', 'Jon']
- NEWER: Gina offered Jon support and told him to let her know if he needs anything.
  - K=['support', 'help', 'encouragement'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — 98dde767 → b0c7ed8c

- **cross-session**; older `s11` (3:14 pm on 11 May, 2023) → newer `s15` (10:04 am on 19 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`says` newer=`congrats` shared=False
- OLDER: Gina says Jon is living the dream and inspiring others, and his studio will change things for many people.
  - K=['dance studio', 'inspiration'] entities=['Gina', 'Jon']
- NEWER: Gina congratulated Jon, saying his dance studio looks amazing and that she is pumped for the launch on 20 June 2023.
  - K=['congrats', 'dance studio', 'launch'] entities=['Gina', 'Jon', 'dance studio']
- semantic verdict: **unread**

## locomo_0001 — f06ec7a4 → ce414166

- **cross-session**; older `s1` (4:04 pm on 20 January, 2023) → newer `s6` (2:35 pm on 16 March, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`loss` newer=`loss` shared=True
- OLDER: Gina lost her job at Door Dash in January 2023.
  - K=['job loss', 'Door Dash', 'employment'] entities=['Gina', 'Door Dash']
- NEWER: Gina lost her job at Door Dash.
  - K=['job loss', 'Door Dash', 'unemployment'] entities=['Gina', 'Door Dash']
- semantic verdict: **unread**

## locomo_0001 — 29436df1 → 01070b10

- **cross-session**; older `s12` (7:18 pm on 27 May, 2023) → newer `s13` (8:29 pm on 13 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`milestones` newer=`goals` shared=False
- OLDER: Jon uses a whiteboard with a list of dates to keep track of ideas and milestones and to visualize his progress.
  - K=['whiteboard', 'milestones', 'organization'] entities=['Jon']
- NEWER: Jon said he uses a notepad to stay organized and motivated, setting goals, tracking achievements, and identifying areas to improve.
  - K=['notepad', 'organization', 'motivation', 'goals'] entities=['Jon']
- semantic verdict: **unread**

## locomo_0001 — f9e1c216 → f43877db

- **cross-session**; older `s7` (7:28 pm on 23 March, 2023) → newer `s16` (2:15 pm on 21 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`encourages` newer=`belief` shared=False
- OLDER: Gina encourages Jon to keep going, never give up, and believe in himself even when it's tough.
  - K=['encouragement', 'believe in yourself', 'never give up'] entities=['Gina', 'Jon']
- NEWER: Gina told Jon to believe in himself and keep going.
  - K=['belief', 'encouragement', 'support'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0001 — 01070b10 → ec4719c6

- **same-session**; older `s13` (8:29 pm on 13 June, 2023) → newer `s13` (8:29 pm on 13 June, 2023)
- interval class: `zero_width`
- head verbs: older=`goals` newer=`goals` shared=True
- OLDER: Jon said he uses a notepad to stay organized and motivated, setting goals, tracking achievements, and identifying areas to improve.
  - K=['notepad', 'organization', 'motivation', 'goals'] entities=['Jon']
- NEWER: Jon said he also uses a whiteboard to stay on track, visualize goals, and tokenize successes.
  - K=['whiteboard', 'goals', 'motivation'] entities=['Jon']
- semantic verdict: **unread**

## locomo_0001 — 78405b2f → be6010ee

- **cross-session**; older `s13` (8:29 pm on 13 June, 2023) → newer `s19` (6:46 pm on 23 July, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`thanks` newer=`thanks` shared=True
- OLDER: Jon thanked Gina for being there for him and believing in him, saying it means a lot.
  - K=['thanks', 'support', 'belief'] entities=['Jon', 'Gina']
- NEWER: On 23 July 2023, Jon thanked Gina, saying her help and encouragement mean a lot and will help him make his studio happen.
  - K=['thanks', 'encouragement', 'support'] entities=['Jon', 'Gina']
- semantic verdict: **unread**

## locomo_0001 — 20a51eb9 → ec7e8198

- **same-session**; older `s5` (9:32 am on 8 February, 2023) → newer `s5` (9:32 am on 8 February, 2023)
- interval class: `zero_width`
- head verbs: older=`thanks` newer=`dreams` shared=False
- OLDER: Jon thanked Gina, saying her help means a lot and he will keep plugging away and stay optimistic.
  - K=['thanks', 'optimism', 'support'] entities=['Jon', 'Gina']
- NEWER: Jon thanked Gina, saying her pep-talk meant a lot and he will not give up on his dreams of his dance studio and business ventures.
  - K=['pep-talk', 'dreams', 'dance studio'] entities=['Jon', 'Gina']
- semantic verdict: **unread**

## locomo_0001 — ec7e8198 → ce002a22

- **cross-session**; older `s5` (9:32 am on 8 February, 2023) → newer `s6` (2:35 pm on 16 March, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`dreams` newer=`thanks` shared=False
- OLDER: Jon thanked Gina, saying her pep-talk meant a lot and he will not give up on his dreams of his dance studio and business ventures.
  - K=['pep-talk', 'dreams', 'dance studio'] entities=['Jon', 'Gina']
- NEWER: Jon thanked Gina for having his back and toasted to taking on new heights and all the trials that come with it.
  - K=['thanks', 'support', 'toast'] entities=['Jon', 'Gina']
- semantic verdict: **unread**

## locomo_0001 — 1dbe9199 → d304ad60

- **cross-session**; older `s5` (9:32 am on 8 February, 2023) → newer `s9` (10:33 am on 9 April, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`loves` newer=`happiness` shared=False
- OLDER: Jon loves dancing because it brings him joy and reminds him why he is passionate about it.
  - K=['dancing', 'joy', 'passion'] entities=['Jon']
- NEWER: Jon says dancing makes him happy and allows him to express his true self.
  - K=['dance', 'happiness', 'self-expression'] entities=['Jon']
- semantic verdict: **unread**

## locomo_0001 — 91475068 → f2b275dd

- **cross-session**; older `s5` (9:32 am on 8 February, 2023) → newer `s8` (1:26 pm on 3 April, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`encouraged` newer=`goals` shared=False
- OLDER: Gina encouraged Jon, saying it's tough starting a business but he can make his studio work, and that she is always there for him.
  - K=['encouragement', 'studio', 'support'] entities=['Gina', 'Jon']
- NEWER: Gina said Jon's support means a lot and encouraged him to keep pursuing his goals.
  - K=['support', 'encouragement', 'goals'] entities=['Gina', 'Jon']
- semantic verdict: **unread**

## locomo_0002 — f49dbd1c → 0cf7a3d1

- **cross-session**; older `s1` (11:01 am on 17 December, 2022) → newer `s2` (6:10 pm on 22 December, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`homeless` newer=`homeless` shared=True
- OLDER: Maria has been volunteering at a homeless shelter.
  - K=['volunteering', 'homeless shelter', 'community'] entities=['Maria']
- NEWER: Maria volunteers at a homeless shelter.
  - K=['volunteering', 'homeless shelter'] entities=['Maria']
- semantic verdict: **unread**

## locomo_0002 — 94d25e29 → 42f16c78

- **cross-session**; older `s17` (11:51 am on 3 June, 2023) → newer `s30` (12:10 am on 11 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`thanked` newer=`thanks` shared=False
- OLDER: John thanked Maria again for her support, saying it means a lot to him.
  - K=['support', 'gratitude', 'appreciation'] entities=['John', 'Maria']
- NEWER: John thanks Maria for her kind words and says he will let her know if he needs anything else.
  - K=['thanks', 'kind words'] entities=['John', 'Maria']
- semantic verdict: **unread**

## locomo_0002 — d746144b → b5462a19

- **cross-session**; older `s17` (11:51 am on 3 June, 2023) → newer `s23` (6:29 pm on 7 July, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`reiterated` newer=`expressed` shared=False
- OLDER: Maria reiterated to John that he should let her know if he needs any help.
  - K=['help', 'support', 'friendship'] entities=['Maria', 'John']
- NEWER: Maria expressed sympathy to John about the flood in his area and offered to help.
  - K=['sympathy', 'help', 'flood'] entities=['Maria', 'John']
- semantic verdict: **unread**

## locomo_0002 — 94d25e29 → 9bc64150

- **same-session**; older `s17` (11:51 am on 3 June, 2023) → newer `s17` (11:51 am on 3 June, 2023)
- interval class: `inverted`
- head verbs: older=`thanked` newer=`thanked` shared=True
- OLDER: John thanked Maria again for her support, saying it means a lot to him.
  - K=['support', 'gratitude', 'appreciation'] entities=['John', 'Maria']
- NEWER: John thanked Maria for her offer of help with adopting a rescue dog and said he would let her know if he needs help.
  - K=['help', 'rescue dog', 'appreciation', 'friendship'] entities=['John', 'Maria']
- semantic verdict: **unread**

## locomo_0002 — 52188d9d → 79d330e9

- **cross-session**; older `s17` (11:51 am on 3 June, 2023) → newer `s18` (2:47 pm on 12 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`loss` newer=`loss` shared=True
- OLDER: John's dog Max died; Max had been an important part of John's family for 10 years.
  - K=['Max', 'dog', 'pet loss', 'grief', 'family'] entities=['John', 'Max']
- NEWER: Max, John's pet, died before 12 June 2023.
  - K=['Max', 'pet', 'loss', 'grief'] entities=['John', 'Max']
- semantic verdict: **unread**

## locomo_0002 — e5b4b877 → fd5ffe1a

- **cross-session**; older `s20` (12:21 am on 27 June, 2023) → newer `s26` (1:59 pm on 31 July, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`kindness` newer=`kindness` shared=True
- OLDER: Maria intends to continue volunteering as her way of spreading kindness and positivity.
  - K=['volunteering', 'kindness', 'positivity', 'plan'] entities=['Maria']
- NEWER: Maria agreed with John that kindness matters and said she will keep up her volunteering.
  - K=['volunteering', 'kindness', 'commitment'] entities=['Maria']
- semantic verdict: **unread**

## locomo_0002 — 4dbd2135 → 9bc64150

- **cross-session**; older `s16` (1:24 pm on 25 May, 2023) → newer `s17` (11:51 am on 3 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`thanks` newer=`thanked` shared=False
- OLDER: John thanked Maria for letting him help with her fundraiser.
  - K=['thanks', 'help', 'fundraiser'] entities=['John', 'Maria']
- NEWER: John thanked Maria for her offer of help with adopting a rescue dog and said he would let her know if he needs help.
  - K=['help', 'rescue dog', 'appreciation', 'friendship'] entities=['John', 'Maria']
- semantic verdict: **unread**

## locomo_0002 — eae9160a → 724679eb

- **cross-session**; older `s26` (1:59 pm on 31 July, 2023) → newer `s32` (11:08 am on 16 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`joined` newer=`fire` shared=False
- OLDER: John joined a fire-fighting brigade because Maria's homeless shelter work inspired him to give back to his community.
  - K=['fire brigade', 'volunteering', 'community'] entities=['John', 'fire-fighting brigade']
- NEWER: On 16 August 2023, John said he is now part of the fire-fighting brigade.
  - K=['fire-fighting brigade', 'community', 'volunteering'] entities=['John']
- semantic verdict: **unread**

## locomo_0002 — ab39fc0c → 7dcae2f2

- **cross-session**; older `s28` (5:19 pm on 5 August, 2023) → newer `s30` (12:10 am on 11 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`times` newer=`steps` shared=False
- OLDER: Maria told John that staying positive makes a big difference even in tough times.
  - K=['positivity', 'encouragement', 'tough times'] entities=['Maria', 'John']
- NEWER: Maria suggests to John that taking small steps can help.
  - K=['small steps', 'advice'] entities=['Maria', 'John']
- semantic verdict: **unread**

## locomo_0002 — 703a6d13 → d5d2e507

- **cross-session**; older `s8` (6:03 pm on 6 March, 2023) → newer `s11` (6:13 pm on 10 April, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`homeless` newer=`homeless` shared=True
- OLDER: Maria volunteers at a homeless shelter.
  - K=['homeless shelter', 'volunteering'] entities=['Maria']
- NEWER: Maria volunteers at a homeless shelter.
  - K=['volunteering', 'homeless shelter'] entities=['Maria', 'homeless shelter']
- semantic verdict: **unread**

## locomo_0002 — 0cf7a3d1 → 637912f7

- **cross-session**; older `s2` (6:10 pm on 22 December, 2022) → newer `s7` (8:55 pm on 25 February, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`homeless` newer=`homeless` shared=True
- OLDER: Maria volunteers at a homeless shelter.
  - K=['volunteering', 'homeless shelter'] entities=['Maria']
- NEWER: Maria has been volunteering at a homeless shelter and finds it really rewarding.
  - K=['volunteering', 'homeless shelter', 'rewarding'] entities=['Maria']
- semantic verdict: **unread**

## locomo_0002 — 5cfdb633 → 4dbd2135

- **cross-session**; older `s11` (6:13 pm on 10 April, 2023) → newer `s16` (1:24 pm on 25 May, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`thanks` newer=`thanks` shared=True
- OLDER: John thanked Maria for being a positive force and said it gives hope and reminds people they are not alone.
  - K=['thanks', 'positive force', 'hope', 'community'] entities=['John', 'Maria']
- NEWER: John thanked Maria for letting him help with her fundraiser.
  - K=['thanks', 'help', 'fundraiser'] entities=['John', 'Maria']
- semantic verdict: **unread**

## locomo_0002 — 1f65f0de → 5cfdb633

- **same-session**; older `s11` (6:13 pm on 10 April, 2023) → newer `s11` (6:13 pm on 10 April, 2023)
- interval class: `zero_width`
- head verbs: older=`thanks` newer=`thanks` shared=True
- OLDER: John thanked Maria and said her kind words meant a lot to him.
  - K=['thanks', 'kind words', 'support'] entities=['John', 'Maria']
- NEWER: John thanked Maria for being a positive force and said it gives hope and reminds people they are not alone.
  - K=['thanks', 'positive force', 'hope', 'community'] entities=['John', 'Maria']
- semantic verdict: **unread**

## locomo_0002 — b5462a19 → 685117f2

- **cross-session**; older `s23` (6:29 pm on 7 July, 2023) → newer `s30` (12:10 am on 11 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`expressed` newer=`tells` shared=False
- OLDER: Maria expressed sympathy to John about the flood in his area and offered to help.
  - K=['sympathy', 'help', 'flood'] entities=['Maria', 'John']
- NEWER: Maria tells John she is happy to help and to let her know if he needs anything else, encouraging him.
  - K=['help', 'encouragement'] entities=['Maria', 'John']
- semantic verdict: **unread**

## locomo_0002 — 1729492a → 95ad0df4

- **cross-session**; older `s22` (6:59 pm on 5 July, 2023) → newer `s31` (3:14 pm on 13 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`reacted` newer=`activities` shared=False
- OLDER: Maria reacted to John's family photo, saying his family looks stoked and that their trip must have been incredible.
  - K=['family', 'photo', 'trip'] entities=['Maria', 'John']
- NEWER: Maria reacted to John's family photo, calling it cute and asking what activities his family enjoys together.
  - K=['family photo', 'reaction', 'activities'] entities=['Maria', 'John']
- semantic verdict: **unread**

## locomo_0002 — d01dc590 → 2d682152

- **cross-session**; older `s22` (6:59 pm on 5 July, 2023) → newer `s32` (11:08 am on 16 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`offers` newer=`supports` shared=False
- OLDER: Maria offers to lend an ear or help John however she can and tells him he is making a difference.
  - K=['support', 'encouragement'] entities=['Maria', 'John']
- NEWER: Maria told John that she is here for him and supports him.
  - K=['support', 'friendship'] entities=['Maria', 'John']
- semantic verdict: **unread**

## locomo_0002 — 637912f7 → 703a6d13

- **cross-session**; older `s7` (8:55 pm on 25 February, 2023) → newer `s8` (6:03 pm on 6 March, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`homeless` newer=`homeless` shared=True
- OLDER: Maria has been volunteering at a homeless shelter and finds it really rewarding.
  - K=['volunteering', 'homeless shelter', 'rewarding'] entities=['Maria']
- NEWER: Maria volunteers at a homeless shelter.
  - K=['homeless shelter', 'volunteering'] entities=['Maria']
- semantic verdict: **unread**

## locomo_0002 — 06658b20 → d746144b

- **cross-session**; older `s9` (9:36 am on 2 April, 2023) → newer `s17` (11:51 am on 3 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`offered` newer=`reiterated` shared=False
- OLDER: Maria offered to help John and noted that she and John work well together.
  - K=['help', 'support', 'teamwork'] entities=['Maria', 'John']
- NEWER: Maria reiterated to John that he should let her know if he needs any help.
  - K=['help', 'support', 'friendship'] entities=['Maria', 'John']
- semantic verdict: **unread**

## locomo_0003 — 79c2c5d4 → 6c881565

- **cross-session**; older `s15` (2:12 pm on 5 June, 2022) → newer `s20` (6:03 pm on 5 September, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`makes` newer=`turtles` shared=False
- OLDER: Nate has a toy Iron Man figure in his room that makes him feel invincible.
  - K=['Iron Man', 'figure', 'toy', 'room'] entities=['Nate', 'Iron Man']
- NEWER: Nate has two pet turtles and finds hanging with them a big help, especially recently.
  - K=['turtles', 'pets', 'help'] entities=['Nate']
- semantic verdict: **unread**

## locomo_0003 — b97bc8e1 → 818575f7

- **cross-session**; older `s15` (2:12 pm on 5 June, 2022) → newer `s22` (11:15 am on 6 October, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`congratulations` newer=`congratulations` shared=True
- OLDER: Nate congratulated Joanna on seeing her hard work pay off with her screenplay on the big screen.
  - K=['congratulations', 'screenplay', 'hard work'] entities=['Nate', 'Joanna']
- NEWER: Nate congratulated Joanna on finishing her book.
  - K=['congratulations', 'book', 'writing'] entities=['Nate', 'Joanna']
- semantic verdict: **unread**

## locomo_0003 — 6e871b79 → 019d2cb9

- **cross-session**; older `s8` (6:44 pm on 17 April, 2022) → newer `s15` (2:12 pm on 5 June, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`asked` newer=`quotes` shared=False
- OLDER: Joanna is lactose intolerant and asked for the dairy-free recipe.
  - K=['lactose intolerant', 'dairy-free', 'recipe'] entities=['Joanna']
- NEWER: Joanna has a cork board full of inspiring quotes and pictures for motivation and creativity in her room.
  - K=['cork board', 'quotes', 'pictures', 'motivation', 'creativity'] entities=['Joanna']
- semantic verdict: **unread**

## locomo_0003 — 29e9533d → 5a1ed340

- **cross-session**; older `s5` (6:59 pm on 18 March, 2022) → newer `s9` (7:44 pm on 21 April, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`encouraged` newer=`encouraged` shared=True
- OLDER: Nate encouraged Joanna not to quit, saying the path forward will show up soon.
  - K=['encouragement', 'support', 'quitting'] entities=['Nate', 'Joanna']
- NEWER: On 21 April 2022, Nate encouraged Joanna to follow her passion for writing and to give acting a shot if it makes her happy.
  - K=['encouragement', 'writing', 'acting'] entities=['Nate', 'Joanna']
- semantic verdict: **unread**

## locomo_0003 — 6851cb20 → 79c2c5d4

- **cross-session**; older `s5` (6:59 pm on 18 March, 2022) → newer `s15` (2:12 pm on 5 June, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`turtles` newer=`makes` shared=False
- OLDER: Nate owns turtles and is drawn to them because they are unique, low-maintenance and calming, and their slow pace is a nice change from the rush of life.
  - K=['turtles', 'pets', 'calming', 'low-maintenance'] entities=['Nate']
- NEWER: Nate has a toy Iron Man figure in his room that makes him feel invincible.
  - K=['Iron Man', 'figure', 'toy', 'room'] entities=['Nate', 'Iron Man']
- semantic verdict: **unread**

## locomo_0003 — 6851cb20 → 77e319d1

- **cross-session**; older `s5` (6:59 pm on 18 March, 2022) → newer `s27` (8:10 pm on 7 November, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`turtles` newer=`turtles` shared=True
- OLDER: Nate owns turtles and is drawn to them because they are unique, low-maintenance and calming, and their slow pace is a nice change from the rush of life.
  - K=['turtles', 'pets', 'calming', 'low-maintenance'] entities=['Nate']
- NEWER: Nate's turtles are doing great and bring joy and calm to his life.
  - K=['turtles', 'pets', 'joy'] entities=['Nate']
- semantic verdict: **unread**

## locomo_0003 — 885c5687 → ea6f39f9

- **cross-session**; older `s5` (6:59 pm on 18 March, 2022) → newer `s25` (8:16 pm on 25 October, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`turtles` newer=`turtles` shared=True
- OLDER: Joanna finds Nate's turtles peaceful and says it is amazing how these creatures bring so much calm and joy.
  - K=['turtles', 'calm', 'joy', 'peaceful'] entities=['Joanna', 'Nate']
- NEWER: Joanna said she never knew turtles could be so interesting until she met Nate's turtles.
  - K=['turtles', 'interesting', 'reaction'] entities=['Joanna', 'Nate']
- semantic verdict: **unread**

## locomo_0003 — ea6f39f9 → eb137f41

- **cross-session**; older `s25` (8:16 pm on 25 October, 2022) → newer `s29` (12:06 am on 11 November, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`turtles` newer=`turtles` shared=True
- OLDER: Joanna said she never knew turtles could be so interesting until she met Nate's turtles.
  - K=['turtles', 'interesting', 'reaction'] entities=['Joanna', 'Nate']
- NEWER: Joanna reacted to Nate's turtles, saying they look serene and happy and that it's great to have something like that.
  - K=['turtles', 'reaction', 'serene'] entities=['Joanna', 'Nate']
- semantic verdict: **unread**

## locomo_0003 — ac5b173e → 2638ad98

- **cross-session**; older `s3` (9:27 am on 7 February, 2022) → newer `s22` (11:15 am on 6 October, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`recommended` newer=`recommended` shared=True
- OLDER: Joanna recommended the movie "Little Women" to Nate, calling it a must-see.
  - K=['Little Women', 'recommendation', 'must-see'] entities=['Joanna', 'Nate']
- NEWER: Joanna recommended "Little Women" to Nate a while back before 6 October 2022.
  - K=['Little Women', 'recommendation', 'movie'] entities=['Joanna', 'Nate']
- semantic verdict: **unread**

## locomo_0003 — e6f3edbd → 4f662818

- **cross-session**; older `s3` (9:27 am on 7 February, 2022) → newer `s22` (11:15 am on 6 October, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`dreams` newer=`dreams` shared=True
- OLDER: Joanna watched the movie "Little Women" and found it amazing, describing it as a great story about sisterhood, love, and reaching for dreams.
  - K=['Little Women', 'movie', 'sisterhood', 'dreams'] entities=['Joanna']
- NEWER: Joanna watched "Little Women" and it spoke to her, exploring themes of sisterhood, love, and chasing dreams.
  - K=['Little Women', 'sisterhood', 'love', 'dreams'] entities=['Joanna']
- semantic verdict: **unread**

## locomo_0004 — 04a618f9 → 1b479faf

- **cross-session**; older `s6` (1:08 pm on 11 August, 2023) → newer `s8` (4:29 pm on 21 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`shoes` newer=`loves` shared=False
- OLDER: John shared his lucky basketball shoes, saying every mark on them has a story.
  - K=['basketball shoes', 'lucky', 'stories'] entities=['John']
- NEWER: John said he loves the first Harry Potter movie and has the whole collection.
  - K=['Harry Potter', 'movie collection'] entities=['John']
- semantic verdict: **unread**

## locomo_0004 — dde29555 → 445cae0e

- **cross-session**; older `s6` (1:08 pm on 11 August, 2023) → newer `s13` (1:50 pm on 13 October, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`encouraged` newer=`encouraged` shared=True
- OLDER: Tim encouraged John, saying he has tons of determination and passion and will make a difference with his charity work.
  - K=['support', 'encouragement', 'charity', 'determination'] entities=['Tim', 'John']
- NEWER: Tim told John he is proud of him and encouraged him to keep it up.
  - K=['encouragement', 'support', 'pride'] entities=['Tim', 'John']
- semantic verdict: **unread**

## locomo_0004 — 445cae0e → 64b1d153

- **cross-session**; older `s13` (1:50 pm on 13 October, 2023) → newer `s17` (3:36 pm on 11 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`encouraged` newer=`encouraged` shared=True
- OLDER: Tim told John he is proud of him and encouraged him to keep it up.
  - K=['encouragement', 'support', 'pride'] entities=['Tim', 'John']
- NEWER: Tim encouraged John to take care of himself mentally and physically and told him he's got this.
  - K=['support', 'mental health', 'physical health', 'encouragement'] entities=['Tim', 'John']
- semantic verdict: **unread**

## locomo_0004 — 3c7e0deb → 85139f98

- **cross-session**; older `s13` (1:50 pm on 13 October, 2023) → newer `s21` (5:34 pm on 6 December, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`trophy` newer=`described` shared=False
- OLDER: John's team won a trophy and he said all the hard work and effort was totally worth it.
  - K=['trophy', 'basketball', 'win'] entities=['John']
- NEWER: Before 6 December 2023, John won a trophy, which he described as awesome.
  - K=['trophy', 'winning', 'achievement'] entities=['John']
- semantic verdict: **unread**

## locomo_0004 — 64b1d153 → 95de3d7f

- **cross-session**; older `s17` (3:36 pm on 11 November, 2023) → newer `s24` (3:37 pm on 16 December, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`encouraged` newer=`rehab` shared=False
- OLDER: Tim encouraged John to take care of himself mentally and physically and told him he's got this.
  - K=['support', 'mental health', 'physical health', 'encouragement'] entities=['Tim', 'John']
- NEWER: Tim told John that rehab can be tough but is key to keep up.
  - K=['rehab', 'encouragement', 'recovery'] entities=['Tim', 'John']
- semantic verdict: **unread**

## locomo_0004 — 95de3d7f → 5aef7530

- **cross-session**; older `s24` (3:37 pm on 16 December, 2023) → newer `s25` (10:04 am on 19 December, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`rehab` newer=`encouraged` shared=False
- OLDER: Tim told John that rehab can be tough but is key to keep up.
  - K=['rehab', 'encouragement', 'recovery'] entities=['Tim', 'John']
- NEWER: Tim encouraged John to keep believing in himself and said he is there for him.
  - K=['encouragement', 'support', 'believing'] entities=['Tim', 'John']
- semantic verdict: **unread**

## locomo_0004 — b7d45368 → 62e603a1

- **cross-session**; older `s18` (3:59 pm on 16 November, 2023) → newer `s27` (5:26 pm on 2 January, 2024)
- interval class: `older_closed_newer_open`
- head verbs: older=`injury` newer=`painting` shared=False
- OLDER: As of 16 November 2023, John is dealing with an injury and has a bandage on his leg.
  - K=['injury', 'bandage', 'leg'] entities=['John']
- NEWER: John has a painting in his room of Aragorn to remind him to stay true and be a leader.
  - K=['painting', 'room', 'Aragorn', 'reminder'] entities=['John']
- semantic verdict: **unread**

## locomo_0004 — a18b40cd → d6bf1e36

- **cross-session**; older `s3` (4:21 pm on 16 July, 2023) → newer `s8` (4:29 pm on 21 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`books` newer=`books` shared=True
- OLDER: Tim does not surf, but reading a great fantasy book helps Tim escape and feel free.
  - K=['fantasy', 'books', 'reading', 'escape'] entities=['Tim']
- NEWER: Tim has been reading a bunch of fantasy books as a break from stress.
  - K=['fantasy books', 'reading', 'stress relief'] entities=['Tim']
- semantic verdict: **unread**

## locomo_0004 — c9ac5c04 → 272e6f17

- **cross-session**; older `s2` (5:08 pm on 15 June, 2023) → newer `s16` (11:41 am on 6 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`props` newer=`signed` shared=False
- OLDER: As of 15 June 2023, Tim owns a picture from MinaLima, who created all the props for the Harry Potter films.
  - K=['MinaLima', 'Harry Potter', 'props', 'picture'] entities=['Tim', 'MinaLima']
- NEWER: Tim owns a prized possession: a basketball signed by his favorite player.
  - K=['basketball', 'signed', 'prized possession', 'keepsake'] entities=['Tim']
- semantic verdict: **unread**

## locomo_0004 — d21f8d25 → dde29555

- **cross-session**; older `s2` (5:08 pm on 15 June, 2023) → newer `s6` (1:08 pm on 11 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`dreams` newer=`encouraged` shared=False
- OLDER: On 15 June 2023, Tim encouraged John to keep going and not give up on his dreams.
  - K=['encouragement', 'dreams', 'support'] entities=['Tim', 'John']
- NEWER: Tim encouraged John, saying he has tons of determination and passion and will make a difference with his charity work.
  - K=['support', 'encouragement', 'charity', 'determination'] entities=['Tim', 'John']
- semantic verdict: **unread**

## locomo_0004 — e5d157dc → 2775c7e1

- **cross-session**; older `s2` (5:08 pm on 15 June, 2023) → newer `s27` (5:26 pm on 2 January, 2024)
- interval class: `older_closed_newer_open`
- head verbs: older=`reminders` newer=`realms` shared=False
- OLDER: As of 15 June 2023, Tim has lots of Harry Potter reminders that help him escape reality.
  - K=['Harry Potter', 'reminders', 'escape reality'] entities=['Tim']
- NEWER: Tim has a map of Middle-earth from Lord of the Rings showing the different realms and regions.
  - K=['map', 'Middle-earth', 'LOTR', 'realms'] entities=['Tim']
- semantic verdict: **unread**

## locomo_0004 — d76adce2 → dcf9adba

- **cross-session**; older `s9` (6:59 pm on 26 August, 2023) → newer `s20` (9:52 am on 1 December, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`means` newer=`says` shared=False
- OLDER: Tim said John's support means a lot to him.
  - K=['support', 'appreciation', 'friendship'] entities=['Tim', 'John']
- NEWER: Tim says John's support and encouragement have truly made Tim's journey better.
  - K=['support', 'encouragement', 'friendship'] entities=['Tim', 'John']
- semantic verdict: **unread**

## locomo_0005 — 0efd3fb5 → affb2fa8

- **cross-session**; older `s7` (11:27 am on 2 June, 2023) → newer `s21` (4:18 pm on 4 October, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`offered` newer=`plants` shared=False
- OLDER: Audrey offered to help Andrew find a pet-friendly place by asking him to email her his criteria.
  - K=['help', 'email', 'criteria', 'pet-friendly'] entities=['Audrey', 'Andrew']
- NEWER: Audrey offered to help Andrew with advice if he needs it for plants.
  - K=['help', 'advice', 'plants'] entities=['Audrey', 'Andrew']
- semantic verdict: **unread**

## locomo_0005 — f25b310f → c3cf3b3a

- **cross-session**; older `s17` (12:24 am on 24 August, 2023) → newer `s18` (7:49 pm on 6 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`reminded` newer=`stress` shared=False
- OLDER: Audrey reminded Andrew that forming a bond takes time and not to rush.
  - K=['bond', 'patience', 'advice'] entities=['Audrey', 'Andrew']
- NEWER: Audrey advised Andrew to take care of himself and find ways to manage stress.
  - K=['advice', 'self-care', 'stress management'] entities=['Audrey', 'Andrew']
- semantic verdict: **unread**

## locomo_0005 — 5d411141 → 8e97a70c

- **cross-session**; older `s17` (12:24 am on 24 August, 2023) → newer `s19` (5:53 pm on 24 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`dogs` newer=`dogs` shared=True
- OLDER: Audrey loves her dogs very much and wants to make them as happy as possible.
  - K=['love', 'dogs', 'bond'] entities=['Audrey']
- NEWER: Audrey loves her dogs and they bring so much joy into her life.
  - K=['love', 'joy', 'dogs'] entities=['Audrey']
- semantic verdict: **unread**

## locomo_0005 — e584390b → 4e9790e7

- **cross-session**; older `s3` (4:19 pm on 16 April, 2023) → newer `s17` (12:24 am on 24 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`dogs` newer=`dogs` shared=True
- OLDER: Audrey has always had a strong bond with her dogs and wants to see their happy faces even when they are not with her.
  - K=['dogs', 'bond', 'happiness'] entities=['Audrey']
- NEWER: Audrey says she has a strong bond with her dogs.
  - K=['bond', 'dogs'] entities=['Audrey']
- semantic verdict: **unread**

## locomo_0005 — d8b72604 → 47f2f4e8

- **cross-session**; older `s19` (5:53 pm on 24 September, 2023) → newer `s27` (7:59 pm on 4 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`garden` newer=`dogs` shared=False
- OLDER: Audrey has a small garden.
  - K=['garden', 'small'] entities=['Audrey']
- NEWER: Audrey has four dogs.
  - K=['dogs', 'pups', 'pet ownership'] entities=['Audrey']
- semantic verdict: **unread**

## locomo_0005 — d8b72604 → fff3f923

- **same-session**; older `s19` (5:53 pm on 24 September, 2023) → newer `s19` (5:53 pm on 24 September, 2023)
- interval class: `inverted`
- head verbs: older=`garden` newer=`dogs` shared=False
- OLDER: Audrey has a small garden.
  - K=['garden', 'small'] entities=['Audrey']
- NEWER: Audrey has four dogs, all of which are mutts.
  - K=['four dogs', 'mutts'] entities=['Audrey']
- semantic verdict: **unread**

## locomo_0005 — 43c0ba1a → 9380e691

- **cross-session**; older `s19` (5:53 pm on 24 September, 2023) → newer `s28` (9:02 am on 22 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`reacted` newer=`reacted` shared=True
- OLDER: Andrew reacted to Audrey's dog photo saying she looks adorable and that's the connection he'd like to have with Toby.
  - K=['reacted', 'adorable', 'connection', 'Toby'] entities=['Andrew', 'Audrey', 'Toby']
- NEWER: Andrew reacted positively to Audrey's photo of her dog at the park.
  - K=['photo', 'dog', 'reaction'] entities=['Andrew', 'Audrey']
- semantic verdict: **unread**

## locomo_0005 — f5b13fd1 → 6fd08332

- **cross-session**; older `s19` (5:53 pm on 24 September, 2023) → newer `s27` (7:59 pm on 4 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`named` newer=`dogs` shared=False
- OLDER: Andrew has a dog named Toby.
  - K=['Toby', 'dog'] entities=['Andrew']
- NEWER: Andrew has two dogs, Toby and Buddy.
  - K=['dogs', 'Toby', 'Buddy', 'pet ownership'] entities=['Andrew', 'Toby', 'Buddy']
- semantic verdict: **unread**

## locomo_0005 — c3cf3b3a → 32a42745

- **cross-session**; older `s18` (7:49 pm on 6 September, 2023) → newer `s27` (7:59 pm on 4 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`stress` newer=`dogs` shared=False
- OLDER: Audrey advised Andrew to take care of himself and find ways to manage stress.
  - K=['advice', 'self-care', 'stress management'] entities=['Audrey', 'Andrew']
- NEWER: Audrey advised Andrew to take care of Toby and Buddy first and ensure they are happy and healthy before getting more dogs.
  - K=['advice', 'dogs', 'pet care'] entities=['Audrey', 'Andrew', 'Toby', 'Buddy']
- semantic verdict: **unread**

## locomo_0005 — 32a42745 → b6211ad0

- **cross-session**; older `s27` (7:59 pm on 4 November, 2023) → newer `s28` (9:02 am on 22 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`dogs` newer=`advises` shared=False
- OLDER: Audrey advised Andrew to take care of Toby and Buddy first and ensure they are happy and healthy before getting more dogs.
  - K=['advice', 'dogs', 'pet care'] entities=['Audrey', 'Andrew', 'Toby', 'Buddy']
- NEWER: Audrey advises Andrew to slowly introduce Scout to Toby and Buddy.
  - K=['advice', 'Scout', 'Toby', 'Buddy', 'introduction'] entities=['Audrey', 'Andrew']
- semantic verdict: **unread**

## locomo_0006 — 8806fcb3 → e42c70f3

- **cross-session**; older `s18` (1:45 pm on 6 August, 2022) → newer `s22` (6:53 pm on 1 September, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`james` newer=`times` shared=False
- OLDER: James offered John financial assistance or advice if John needs it, as of 6 August 2022.
  - K=['financial assistance', 'advice', 'support'] entities=['James', 'John']
- NEWER: James told John he is there if John needs someone to talk or vent to, saying it might help alleviate some of the difficult times John is going through.
  - K=['support', 'listening', 'difficult times', 'friendship'] entities=['James', 'John']
- semantic verdict: **unread**

## locomo_0006 — 53cc2e91 → a27fa8ff

- **cross-session**; older `s4` (2:13 pm on 4 April, 2022) → newer `s22` (6:53 pm on 1 September, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`congratulations` newer=`congratulations` shared=True
- OLDER: John congratulated James on his performance and making it to the semifinals.
  - K=['gaming', 'congratulations', 'semifinals'] entities=['John', 'James']
- NEWER: John congratulated James on finishing his game, saying it looks amazing and that he is proud of all the hard work James put in.
  - K=['congratulations', 'game', 'pride', 'support'] entities=['John', 'James']
- semantic verdict: **unread**

## locomo_0006 — 9f191a8b → 254ef07b

- **cross-session**; older `s8` (2:36 pm on 29 April, 2022) → newer `s9` (7:01 pm on 4 May, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`james` newer=`james` shared=True
- OLDER: James has a dog named Daisy who lay down next to him while he was playing a game.
  - K=['dog', 'Daisy', 'gaming companion'] entities=['James', 'Daisy']
- NEWER: James has a Labrador dog named Daisy who loves to play with her toys and loves to eat.
  - K=['dog', 'Labrador', 'Daisy', 'pet'] entities=['James', 'Daisy']
- semantic verdict: **unread**

## locomo_0006 — d325b0ef → b3f4913b

- **cross-session**; older `s20` (3:57 pm on 21 August, 2022) → newer `s21` (9:18 pm on 26 August, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`james` newer=`james` shared=True
- OLDER: James has a gaming setup with a gaming PC, keyboard, mouse, and a comfy chair.
  - K=['gaming PC', 'keyboard', 'mouse', 'chair'] entities=['James']
- NEWER: Ned, James's dog, came to James on 26 August 2022 while James was playing on a console.
  - K=['dog', 'Ned', 'console', 'pet'] entities=['James', 'Ned']
- semantic verdict: **unread**

## locomo_0006 — a41a9ec4 → 949696a1

- **cross-session**; older `s22` (6:53 pm on 1 September, 2022) → newer `s25` (8:56 pm on 20 September, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`congratulations` newer=`congratulations` shared=True
- OLDER: John said completing the game must have been a challenge, especially making sure it was enjoyable and balanced, and congratulated James again.
  - K=['congratulations', 'game', 'balance', 'challenge'] entities=['John', 'James']
- NEWER: John congratulated James on his successful stream, telling him his hard work is paying off.
  - K=['congratulations', 'support', 'hard work'] entities=['John', 'James']
- semantic verdict: **unread**

## locomo_0006 — a27fa8ff → a41a9ec4

- **same-session**; older `s22` (6:53 pm on 1 September, 2022) → newer `s22` (6:53 pm on 1 September, 2022)
- interval class: `zero_width`
- head verbs: older=`congratulations` newer=`congratulations` shared=True
- OLDER: John congratulated James on finishing his game, saying it looks amazing and that he is proud of all the hard work James put in.
  - K=['congratulations', 'game', 'pride', 'support'] entities=['John', 'James']
- NEWER: John said completing the game must have been a challenge, especially making sure it was enjoyable and balanced, and congratulated James again.
  - K=['congratulations', 'game', 'balance', 'challenge'] entities=['John', 'James']
- semantic verdict: **unread**

## locomo_0006 — fe51380d → 63bac73a

- **cross-session**; older `s22` (6:53 pm on 1 September, 2022) → newer `s28` (7:36 pm on 21 October, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`james` newer=`thanks` shared=False
- OLDER: James thanked John for his support and encouragement, saying it made a real difference and that John believed in him.
  - K=['support', 'encouragement', 'gratitude', 'friendship'] entities=['James', 'John']
- NEWER: James thanked John for the advice and for avoiding spoilers about Cyberpunk 2077.
  - K=['thanks', 'spoilers', 'Cyberpunk 2077'] entities=['James', 'John']
- semantic verdict: **unread**

## locomo_0006 — 254ef07b → 3c4fb073

- **same-session**; older `s9` (7:01 pm on 4 May, 2022) → newer `s9` (7:01 pm on 4 May, 2022)
- interval class: `zero_width`
- head verbs: older=`james` newer=`dogs` shared=False
- OLDER: James has a Labrador dog named Daisy who loves to play with her toys and loves to eat.
  - K=['dog', 'Labrador', 'Daisy', 'pet'] entities=['James', 'Daisy']
- NEWER: James has two dogs that are shepherds, which John identified from the photo.
  - K=['dogs', 'shepherds', 'pets'] entities=['James']
- semantic verdict: **unread**

## locomo_0006 — 3c4fb073 → d325b0ef

- **cross-session**; older `s9` (7:01 pm on 4 May, 2022) → newer `s20` (3:57 pm on 21 August, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`dogs` newer=`james` shared=False
- OLDER: James has two dogs that are shepherds, which John identified from the photo.
  - K=['dogs', 'shepherds', 'pets'] entities=['James']
- NEWER: James has a gaming setup with a gaming PC, keyboard, mouse, and a comfy chair.
  - K=['gaming PC', 'keyboard', 'mouse', 'chair'] entities=['James']
- semantic verdict: **unread**

## locomo_0006 — e9ae4020 → 80fbd6a3

- **cross-session**; older `s12` (7:33 pm on 23 May, 2022) → newer `s27` (2:14 pm on 13 October, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`goals` newer=`offered` shared=False
- OLDER: John offered ongoing help and support to James to reach his goals.
  - K=['support', 'help', 'goals'] entities=['John', 'James']
- NEWER: John offered James his support, telling James he can count on him for anything.
  - K=['support', 'friendship', 'help'] entities=['John', 'James']
- semantic verdict: **unread**

## locomo_0006 — ab07ed9c → 7c43a9fb

- **cross-session**; older `s6` (9:32 pm on 20 April, 2022) → newer `s31` (8:57 pm on 7 November, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`james` newer=`dogs` shared=False
- OLDER: James has a screenshot of the game character he created for the virtual world.
  - K=['screenshot', 'game character', 'virtual world'] entities=['James']
- NEWER: James has three dogs at home as of 7 November 2022.
  - K=['dogs', 'pets'] entities=['James']
- semantic verdict: **unread**

## locomo_0006 — 081aad5b → a8094482

- **cross-session**; older `s5` (9:52 am on 12 April, 2022) → newer `s21` (9:18 pm on 26 August, 2022)
- interval class: `older_closed_newer_open`
- head verbs: older=`james` newer=`james` shared=True
- OLDER: On 5 April 2022, James adopted a puppy named Ned from a shelter in Stamford.
  - K=['adoption', 'puppy', 'shelter', 'Ned'] entities=['James', 'Ned', 'Stamford']
- NEWER: James adopted Ned, and Ned has been an awesome companion to James ever since.
  - K=['dog', 'adoption', 'Ned', 'pet'] entities=['James', 'Ned']
- semantic verdict: **unread**

## locomo_0007 — d6ab19cb → ec02c526

- **cross-session**; older `s9` (11:22 am on 13 March, 2023) → newer `s25` (8:31 pm on 6 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`needed` newer=`needs` shared=False
- OLDER: Deborah told Jolene she is there for her if needed and suggested they catch up soon.
  - K=['support', 'catch up', 'care'] entities=['Deborah', 'Jolene']
- NEWER: Deborah told Jolene she is there if Jolene needs her and advised Jolene to take care of herself and rest up.
  - K=['support', 'care', 'rest'] entities=['Deborah', 'Jolene']
- semantic verdict: **unread**

## locomo_0007 — df377951 → 405654fe

- **cross-session**; older `s18` (2:58 pm on 16 August, 2023) → newer `s27` (2:18 pm on 12 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`this` newer=`mindfulness` shared=False
- OLDER: Deborah told Jolene they are in this together and to give her a shout if she needs anything.
  - K=['support', 'together'] entities=['Deborah', 'Jolene']
- NEWER: Deborah and Jolene support and motivate each other on their mindfulness journey.
  - K=['support', 'community', 'mindfulness'] entities=['Deborah', 'Jolene']
- semantic verdict: **unread**

## locomo_0007 — fd9bb2b8 → b11192fd

- **cross-session**; older `s13` (3:56 pm on 6 June, 2023) → newer `s17` (8:50 pm on 12 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`thanks` newer=`thanks` shared=True
- OLDER: Jolene thanked Deborah for the advice and said she would definitely give it a try.
  - K=['thanks', 'advice', 'gratitude'] entities=['Jolene', 'Deborah']
- NEWER: Jolene thanked Deborah, saying her support means a lot and she will keep pushing to make a difference.
  - K=['thanks', 'support', 'make a difference'] entities=['Jolene', 'Deborah']
- semantic verdict: **unread**

## locomo_0007 — 3b6d3e78 → 61849b23

- **same-session**; older `s1` (4:06 pm on 23 January, 2023) → newer `s1` (4:06 pm on 23 January, 2023)
- interval class: `inverted`
- head verbs: older=`reminds` newer=`keeps` shared=False
- OLDER: Deborah has a pendant that reminds her of her mother.
  - K=['pendant', 'mother', 'reminder'] entities=['Deborah', "Deborah's mother"]
- NEWER: Deborah keeps a last photo together with her mother, who is in a wheelchair.
  - K=['photo', 'mother', 'wheelchair'] entities=['Deborah', "Deborah's mother"]
- semantic verdict: **unread**

## locomo_0007 — 76a9cd7c → b68057ea

- **cross-session**; older `s1` (4:06 pm on 23 January, 2023) → newer `s29` (1:24 pm on 17 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`memories` newer=`memories` shared=True
- OLDER: Deborah visited her mother's old house during the week before 23 January 2023.
  - K=['mother', 'house', 'memories'] entities=['Deborah']
- NEWER: On 10 September 2023, Deborah visited her deceased mother's old house and sat on a bench there.
  - K=['mother', 'old house', 'bench', 'memories'] entities=['Deborah']
- semantic verdict: **unread**

## locomo_0007 — 3b6d3e78 → d41afbbf

- **cross-session**; older `s1` (4:06 pm on 23 January, 2023) → newer `s15` (7:37 pm on 9 July, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`reminds` newer=`photos` shared=False
- OLDER: Deborah has a pendant that reminds her of her mother.
  - K=['pendant', 'mother', 'reminder'] entities=['Deborah', "Deborah's mother"]
- NEWER: Deborah has many photos from her running group activity.
  - K=['photos', 'running group', 'activity'] entities=['Deborah']
- semantic verdict: **unread**

## locomo_0007 — 74deeedf → e51be4ed

- **cross-session**; older `s6` (4:12 pm on 22 February, 2023) → newer `s23` (11:46 am on 30 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`loss` newer=`expressed` shared=False
- OLDER: Jolene expressed sympathy to Deborah for losing her friend and asked how she is holding up.
  - K=['sympathy', 'loss', 'friend'] entities=['Jolene', 'Deborah']
- NEWER: Jolene expressed sympathy to Deborah about the friend who wrote the quote and will never be able to support her.
  - K=['sympathy', 'friend', 'quote'] entities=['Jolene', 'Deborah']
- semantic verdict: **unread**

## locomo_0007 — 25934ca6 → 92fad7b5

- **cross-session**; older `s12` (4:30 pm on 9 April, 2023) → newer `s15` (7:37 pm on 9 July, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`snakes` newer=`reptiles` shared=False
- OLDER: Jolene owns snakes that cannot chat or understand her, but Jolene values her time with them as it teaches her to take time and be in tune with herself.
  - K=['snakes', 'pets', 'self-care'] entities=['Jolene']
- NEWER: Jolene decided to get a snake because she was fascinated by reptiles and it felt like the perfect pet for her.
  - K=['snake', 'pet', 'reptiles', 'nature'] entities=['Jolene']
- semantic verdict: **unread**

## locomo_0007 — a129d699 → 63b0833e

- **cross-session**; older `s7` (4:50 pm on 25 February, 2023) → newer `s10` (5:35 pm on 22 March, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`progress` newer=`encouraged` shared=False
- OLDER: Deborah reacted to Jolene's progress, telling Jolene to keep it up.
  - K=['encouragement', 'progress'] entities=['Deborah', 'Jolene']
- NEWER: Deborah encouraged Jolene to take the first steps toward surfing and to believe in herself and keep going.
  - K=['encouragement', 'surfing', 'believe in yourself'] entities=['Deborah', 'Jolene']
- semantic verdict: **unread**

## locomo_0007 — 75a4acbf → 97a55f53

- **cross-session**; older `s7` (4:50 pm on 25 February, 2023) → newer `s16` (9:26 am on 1 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`sets` newer=`prioritizes` shared=False
- OLDER: Jolene sets aside time for herself doing yoga and meditation.
  - K=['yoga', 'meditation', 'self-care'] entities=['Jolene']
- NEWER: Jolene prioritizes self-care through yoga and meditation to stay balanced and grounded.
  - K=['self-care', 'yoga', 'meditation', 'balance'] entities=['Jolene']
- semantic verdict: **unread**

## locomo_0007 — 475815fa → 874f58b8

- **cross-session**; older `s22` (5:33 pm on 26 August, 2023) → newer `s26` (7:39 pm on 8 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`helps` newer=`balanced` shared=False
- OLDER: Jolene has been into yoga and meditation lately, finding it helps her recharge, relieves tension, and calms her mind.
  - K=['yoga', 'meditation', 'self-care'] entities=['Jolene']
- NEWER: Jolene practices self-care activities like yoga and meditation to stay balanced and relaxed.
  - K=['self-care', 'yoga', 'meditation', 'balanced'] entities=['Jolene']
- semantic verdict: **unread**

## locomo_0007 — 39107296 → 24b35054

- **cross-session**; older `s10` (5:35 pm on 22 March, 2023) → newer `s20` (9:11 am on 21 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`uses` newer=`stress` shared=False
- OLDER: Jolene uses the Pomodoro Technique, working 25 minutes and taking 5-minute breaks, to avoid burnout.
  - K=['Pomodoro', 'time management', 'burnout'] entities=['Jolene']
- NEWER: Jolene uses yoga as a way to escape studying and work stress.
  - K=['yoga', 'studying', 'work stress'] entities=['Jolene']
- semantic verdict: **unread**

## locomo_0007 — 63b0833e → df9ace62

- **cross-session**; older `s10` (5:35 pm on 22 March, 2023) → newer `s17` (8:50 pm on 12 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`encouraged` newer=`encouraged` shared=True
- OLDER: Deborah encouraged Jolene to take the first steps toward surfing and to believe in herself and keep going.
  - K=['encouragement', 'surfing', 'believe in yourself'] entities=['Deborah', 'Jolene']
- NEWER: Deborah encouraged Jolene to keep going on her project.
  - K=['encouragement', 'support'] entities=['Deborah', 'Jolene']
- semantic verdict: **unread**

## locomo_0007 — f4392898 → 1eb32a38

- **cross-session**; older `s15` (7:37 pm on 9 July, 2023) → newer `s22` (5:33 pm on 26 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`cats` newer=`cats` shared=True
- OLDER: Deborah has cats.
  - K=['cats', 'pets'] entities=['Deborah']
- NEWER: Deborah owns two cats, Luna and Max.
  - K=['cats', 'pets', 'Luna', 'Max'] entities=['Deborah', 'Luna', 'Max']
- semantic verdict: **unread**

## locomo_0007 — 92fad7b5 → ca79e773

- **cross-session**; older `s15` (7:37 pm on 9 July, 2023) → newer `s22` (5:33 pm on 26 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`reptiles` newer=`likes` shared=False
- OLDER: Jolene decided to get a snake because she was fascinated by reptiles and it felt like the perfect pet for her.
  - K=['snake', 'pet', 'reptiles', 'nature'] entities=['Jolene']
- NEWER: Jolene's pet snake likes watching Jolene chill and is great company that brings a sense of calm.
  - K=['snake', 'pet', 'calm'] entities=['Jolene']
- semantic verdict: **unread**

## locomo_0007 — d41afbbf → 3ebb4db4

- **cross-session**; older `s15` (7:37 pm on 9 July, 2023) → newer `s29` (1:24 pm on 17 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`photos` newer=`books` shared=False
- OLDER: Deborah has many photos from her running group activity.
  - K=['photos', 'running group', 'activity'] entities=['Deborah']
- NEWER: Deborah has a big bookshelf in her living room.
  - K=['bookshelf', 'living room', 'books'] entities=['Deborah']
- semantic verdict: **unread**

## locomo_0007 — df9ace62 → 0d332982

- **cross-session**; older `s17` (8:50 pm on 12 August, 2023) → newer `s25` (8:31 pm on 6 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`encouraged` newer=`challenges` shared=False
- OLDER: Deborah encouraged Jolene to keep going on her project.
  - K=['encouragement', 'support'] entities=['Deborah', 'Jolene']
- NEWER: Deborah told Jolene she is strong for handling all the challenges and that she's got this.
  - K=['support', 'encouragement', 'challenges'] entities=['Deborah', 'Jolene']
- semantic verdict: **unread**

## locomo_0007 — 24b35054 → 9884d999

- **cross-session**; older `s20` (9:11 am on 21 August, 2023) → newer `s26` (7:39 pm on 8 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`stress` newer=`classes` shared=False
- OLDER: Jolene uses yoga as a way to escape studying and work stress.
  - K=['yoga', 'studying', 'work stress'] entities=['Jolene']
- NEWER: Jolene relies on a routine and a schedule for classes, studying, and personal time to stay on top of everything.
  - K=['routine', 'schedule', 'classes', 'studying', 'personal time'] entities=['Jolene']
- semantic verdict: **unread**

## locomo_0007 — 2bd7da82 → 475815fa

- **cross-session**; older `s20` (9:11 am on 21 August, 2023) → newer `s22` (5:33 pm on 26 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`practices` newer=`helps` shared=False
- OLDER: Jolene practices yoga and meditation to relax and stay focused.
  - K=['yoga', 'meditation', 'relaxation'] entities=['Jolene']
- NEWER: Jolene has been into yoga and meditation lately, finding it helps her recharge, relieves tension, and calms her mind.
  - K=['yoga', 'meditation', 'self-care'] entities=['Jolene']
- semantic verdict: **unread**

## locomo_0007 — 97a55f53 → 2bd7da82

- **cross-session**; older `s16` (9:26 am on 1 August, 2023) → newer `s20` (9:11 am on 21 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`prioritizes` newer=`practices` shared=False
- OLDER: Jolene prioritizes self-care through yoga and meditation to stay balanced and grounded.
  - K=['self-care', 'yoga', 'meditation', 'balance'] entities=['Jolene']
- NEWER: Jolene practices yoga and meditation to relax and stay focused.
  - K=['yoga', 'meditation', 'relaxation'] entities=['Jolene']
- semantic verdict: **unread**

## locomo_0007 — 3591c46a → 490e1cb0

- **cross-session**; older `s21` (9:34 am on 24 August, 2023) → newer `s25` (8:31 pm on 6 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`means` newer=`means` shared=True
- OLDER: Jolene said Deborah's support means a lot to her.
  - K=['support', 'gratitude'] entities=['Jolene', 'Deborah']
- NEWER: Jolene told Deborah that her support means a lot and she will keep pushing forward.
  - K=['support', 'gratitude', 'encouragement'] entities=['Jolene', 'Deborah']
- semantic verdict: **unread**

## locomo_0007 — 69bf444c → a129d699

- **cross-session**; older `s4` (9:48 am on 4 February, 2023) → newer `s7` (4:50 pm on 25 February, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`always` newer=`progress` shared=False
- OLDER: Deborah told Jolene she is always there to cheer Jolene on and encouraged her to stay in touch and take care of herself.
  - K=['encouragement', 'support', 'stay in touch'] entities=['Deborah', 'Jolene']
- NEWER: Deborah reacted to Jolene's progress, telling Jolene to keep it up.
  - K=['encouragement', 'progress'] entities=['Deborah', 'Jolene']
- semantic verdict: **unread**

## locomo_0007 — 4a393021 → 4b8cef1f

- **cross-session**; older `s4` (9:48 am on 4 February, 2023) → newer `s19` (12:52 am on 19 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`spends` newer=`yoga` shared=False
- OLDER: Deborah spends a lot of her time teaching yoga.
  - K=['yoga', 'teaching'] entities=['Deborah']
- NEWER: Deborah has been focusing on teaching yoga and spending time with the community.
  - K=['yoga', 'teaching', 'community'] entities=['Deborah']
- semantic verdict: **unread**

## locomo_0007 — c47fa148 → 25934ca6

- **cross-session**; older `s2` (9:49 am on 27 January, 2023) → newer `s12` (4:30 pm on 9 April, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`owns` newer=`snakes` shared=False
- OLDER: Jolene owns a snake named Susie.
  - K=['snake', 'Susie', 'pet'] entities=['Jolene', 'Susie']
- NEWER: Jolene owns snakes that cannot chat or understand her, but Jolene values her time with them as it teaches her to take time and be in tune with herself.
  - K=['snakes', 'pets', 'self-care'] entities=['Jolene']
- semantic verdict: **unread**

## locomo_0008 — 3e421639 → 24e0d0ad

- **cross-session**; older `s4` (10:52 am on 27 July, 2023) → newer `s15` (2:56 pm on 25 October, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`appreciates` newer=`appreciates` shared=True
- OLDER: On 27 July 2023, Sam told Evan that Sam appreciates Evan's support.
  - K=['support', 'appreciation', 'friendship'] entities=['Sam', 'Evan']
- NEWER: On 25 October 2023, Sam said Sam appreciates Evan's support.
  - K=['appreciation', 'support'] entities=['Sam', 'Evan']
- semantic verdict: **unread**

## locomo_0008 — 3e13a9d6 → 6e2a3f1f

- **cross-session**; older `s4` (10:52 am on 27 July, 2023) → newer `s17` (7:30 pm on 21 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`needs` newer=`support` shared=False
- OLDER: On 27 July 2023, Evan told Sam that Evan is there if Sam needs Evan and encouraged Sam to keep going.
  - K=['support', 'encouragement'] entities=['Evan', 'Sam']
- NEWER: Evan told Sam he is there for Sam and that they should stick together.
  - K=['support', 'stick together'] entities=['Evan', 'Sam']
- semantic verdict: **unread**

## locomo_0008 — 2edbab85 → 88479f57

- **cross-session**; older `s1` (1:47 pm on 18 May, 2023) → newer `s6` (4:09 pm on 13 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`hobbies` newer=`wins` shared=False
- OLDER: Evan encouraged Sam to keep trying new things until something sparks his excitement.
  - K=['encouragement', 'new hobbies'] entities=['Evan', 'Sam']
- NEWER: Evan encouraged Sam to celebrate small wins because every step forward counts.
  - K=['small wins', 'encouragement', 'progress'] entities=['Evan', 'Sam']
- semantic verdict: **unread**

## locomo_0008 — 1f9b7ab5 → 6792dfd9

- **same-session**; older `s12` (3:09 pm on 8 October, 2023) → newer `s12` (3:09 pm on 8 October, 2023)
- interval class: `inverted`
- head verbs: older=`reminded` newer=`progress` shared=False
- OLDER: Evan reminded Sam that it takes time, told Sam to be patient with Sam's self, and said Sam's health matters and Evan believes in Sam.
  - K=['patience', 'health', 'belief', 'encouragement'] entities=['Evan', 'Sam']
- NEWER: Evan told Sam he can't wait to hear about Sam's progress and encouraged Sam to keep up the hard work.
  - K=['encouragement', 'progress', 'support'] entities=['Evan', 'Sam']
- semantic verdict: **unread**

## locomo_0008 — 1f9b7ab5 → b38c3dde

- **same-session**; older `s12` (3:09 pm on 8 October, 2023) → newer `s12` (3:09 pm on 8 October, 2023)
- interval class: `zero_width`
- head verbs: older=`reminded` newer=`progress` shared=False
- OLDER: Evan reminded Sam that it takes time, told Sam to be patient with Sam's self, and said Sam's health matters and Evan believes in Sam.
  - K=['patience', 'health', 'belief', 'encouragement'] entities=['Evan', 'Sam']
- NEWER: Evan told Sam that progress is key and encouraged Sam to keep pushing on and stay positive, and shared an image of a notepad with a note and a pair of scissors.
  - K=['progress', 'positivity', 'encouragement', 'notepad'] entities=['Evan', 'Sam']
- semantic verdict: **unread**

## locomo_0008 — 6a72cbad → b38c3dde

- **same-session**; older `s12` (3:09 pm on 8 October, 2023) → newer `s12` (3:09 pm on 8 October, 2023)
- interval class: `inverted`
- head verbs: older=`encouraged` newer=`progress` shared=False
- OLDER: Evan encouraged Sam to stay motivated and believe in Sam's self, saying Sam has got this.
  - K=['motivation', 'belief', 'encouragement'] entities=['Evan', 'Sam']
- NEWER: Evan told Sam that progress is key and encouraged Sam to keep pushing on and stay positive, and shared an image of a notepad with a note and a pair of scissors.
  - K=['progress', 'positivity', 'encouragement', 'notepad'] entities=['Evan', 'Sam']
- semantic verdict: **unread**

## locomo_0008 — 275bc720 → 6a72cbad

- **same-session**; older `s12` (3:09 pm on 8 October, 2023) → newer `s12` (3:09 pm on 8 October, 2023)
- interval class: `inverted`
- head verbs: older=`urged` newer=`encouraged` shared=False
- OLDER: Evan told Sam that Evan is here for Sam and urged Sam to keep taking one step at a time, and that Sam will get there eventually.
  - K=['one step at a time', 'encouragement', 'support'] entities=['Evan', 'Sam']
- NEWER: Evan encouraged Sam to stay motivated and believe in Sam's self, saying Sam has got this.
  - K=['motivation', 'belief', 'encouragement'] entities=['Evan', 'Sam']
- semantic verdict: **unread**

## locomo_0008 — 275bc720 → 2574d6b9

- **cross-session**; older `s12` (3:09 pm on 8 October, 2023) → newer `s17` (7:30 pm on 21 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`urged` newer=`steps` shared=False
- OLDER: Evan told Sam that Evan is here for Sam and urged Sam to keep taking one step at a time, and that Sam will get there eventually.
  - K=['one step at a time', 'encouragement', 'support'] entities=['Evan', 'Sam']
- NEWER: Evan encouraged Sam by saying he believes in Sam and that taking small steps can help.
  - K=['encouragement', 'small steps'] entities=['Evan', 'Sam']
- semantic verdict: **unread**

## locomo_0008 — 88479f57 → ee6d4b54

- **cross-session**; older `s6` (4:09 pm on 13 August, 2023) → newer `s7` (4:20 pm on 15 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`wins` newer=`steps` shared=False
- OLDER: Evan encouraged Sam to celebrate small wins because every step forward counts.
  - K=['small wins', 'encouragement', 'progress'] entities=['Evan', 'Sam']
- NEWER: Evan encouraged Sam to keep moving forward slowly and taking small steps.
  - K=['encouragement', 'support', 'small steps'] entities=['Evan', 'Sam']
- semantic verdict: **unread**

## locomo_0008 — ee6d4b54 → 6792dfd9

- **cross-session**; older `s7` (4:20 pm on 15 August, 2023) → newer `s12` (3:09 pm on 8 October, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`steps` newer=`progress` shared=False
- OLDER: Evan encouraged Sam to keep moving forward slowly and taking small steps.
  - K=['encouragement', 'support', 'small steps'] entities=['Evan', 'Sam']
- NEWER: Evan told Sam he can't wait to hear about Sam's progress and encouraged Sam to keep up the hard work.
  - K=['encouragement', 'progress', 'support'] entities=['Evan', 'Sam']
- semantic verdict: **unread**

## locomo_0008 — 5ace2b3b → 6f61c31d

- **cross-session**; older `s8` (6:17 pm on 19 August, 2023) → newer `s10` (9:28 am on 11 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`reacted` newer=`complimented` shared=False
- OLDER: On 19 August 2023, Sam reacted to Evan's painting, calling Evan really talented and asking if Evan took other classes.
  - K=['painting', 'talent', 'reaction'] entities=['Sam', 'Evan']
- NEWER: Sam complimented Evan's painting and said they might try painting or find another calming hobby.
  - K=['painting', 'hobby', 'compliment', 'calming'] entities=['Sam', 'Evan']
- semantic verdict: **unread**

## locomo_0008 — 1bf56f61 → 6e2a3f1f

- **same-session**; older `s17` (7:30 pm on 21 November, 2023) → newer `s17` (7:30 pm on 21 November, 2023)
- interval class: `inverted`
- head verbs: older=`thanked` newer=`support` shared=False
- OLDER: Evan thanked Sam and told Sam he is here for Sam if Sam needs to talk.
  - K=['support', 'here for you'] entities=['Evan', 'Sam']
- NEWER: Evan told Sam he is there for Sam and that they should stick together.
  - K=['support', 'stick together'] entities=['Evan', 'Sam']
- semantic verdict: **unread**

## locomo_0008 — 1bf56f61 → 15315f13

- **cross-session**; older `s17` (7:30 pm on 21 November, 2023) → newer `s25` (9:37 pm on 11 January, 2024)
- interval class: `older_closed_newer_open`
- head verbs: older=`thanked` newer=`offered` shared=False
- OLDER: Evan thanked Sam and told Sam he is here for Sam if Sam needs to talk.
  - K=['support', 'here for you'] entities=['Evan', 'Sam']
- NEWER: Evan offered to always be available if Sam needs to chat and told Sam to look after themselves.
  - K=['support', 'chat', 'look after yourself'] entities=['Evan', 'Sam']
- semantic verdict: **unread**

## locomo_0008 — f0086496 → aa5da0b7

- **cross-session**; older `s5` (7:52 pm on 7 August, 2023) → newer `s17` (7:30 pm on 21 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`reacted` newer=`reacted` shared=True
- OLDER: Sam reacted to Evan's photo of an island with a lone boat by saying it was an amazing view and that the key is to find something that keeps you motivated.
  - K=['island', 'boat', 'motivation', 'amazing view'] entities=['Sam', 'Evan']
- NEWER: Sam reacted to Evan's photo of a small island with a lone boat, calling it gorgeous and asking where it is.
  - K=['island', 'boat', 'photo'] entities=['Sam', 'Evan']
- semantic verdict: **unread**

## locomo_0008 — 4395184e → 65266c41

- **cross-session**; older `s18` (8:16 pm on 5 December, 2023) → newer `s19` (1:45 pm on 9 December, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`tips` newer=`tips` shared=True
- OLDER: Sam offered Evan yoga tips or other help whenever Evan needs it.
  - K=['yoga tips', 'support', 'help'] entities=['Sam', 'Evan']
- NEWER: Sam offered Evan tips on organizing a big family event and support on 9 December 2023.
  - K=['support', 'tips', 'family event'] entities=['Sam', 'Evan']
- semantic verdict: **unread**

## locomo_0008 — 4f4ae4da → 4395184e

- **cross-session**; older `s16` (9:13 pm on 9 November, 2023) → newer `s18` (8:16 pm on 5 December, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`loss` newer=`tips` shared=False
- OLDER: Sam offered Evan support and someone to talk to about Evan losing Evan's job.
  - K=['support', 'job loss', 'friendship'] entities=['Sam', 'Evan']
- NEWER: Sam offered Evan yoga tips or other help whenever Evan needs it.
  - K=['yoga tips', 'support', 'help'] entities=['Sam', 'Evan']
- semantic verdict: **unread**

## locomo_0008 — 6f61c31d → 110808af

- **cross-session**; older `s10` (9:28 am on 11 September, 2023) → newer `s20` (6:48 pm on 17 December, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`complimented` newer=`reacted` shared=False
- OLDER: Sam complimented Evan's painting and said they might try painting or find another calming hobby.
  - K=['painting', 'hobby', 'compliment', 'calming'] entities=['Sam', 'Evan']
- NEWER: Sam reacted to Evan's painting, commenting that the woman in the photo looks confident and asking what kind of painting is in the background.
  - K=['painting', 'reaction', 'confidence'] entities=['Sam', 'Evan']
- semantic verdict: **unread**

## locomo_0009 — 585a8ff7 → 2579d427

- **cross-session**; older `s15` (11:06 am on 22 August, 2023) → newer `s24` (10:11 am on 19 October, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`frank` newer=`wanted` shared=False
- OLDER: Calvin met Frank Ocean at a music festival in Tokyo.
  - K=['Frank Ocean', 'music festival', 'Tokyo'] entities=['Calvin', 'Frank Ocean', 'Tokyo']
- NEWER: Calvin met Frank Ocean in August 2022 at a festival, where Frank Ocean said he wanted to collaborate.
  - K=['festival', 'collaboration', 'meeting'] entities=['Calvin', 'Frank Ocean']
- semantic verdict: **unread**

## locomo_0009 — 61b47bfc → d0666a3b

- **cross-session**; older `s1` (11:53 am on 23 March, 2023) → newer `s27` (10:49 am on 29 October, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`life` newer=`collaborations` shared=False
- OLDER: Calvin just had a big life change, showing a new mansion as part of that change.
  - K=['life change', 'mansion', 'new home'] entities=['Calvin']
- NEWER: Calvin said he has had some great collaborations recently.
  - K=['collaborations', 'music'] entities=['Calvin']
- semantic verdict: **unread**

## locomo_0009 — 9a02278c → 0dfe2fe3

- **cross-session**; older `s19` (12:13 am on 15 September, 2023) → newer `s22` (3:13 pm on 8 October, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`talented` newer=`encouragement` shared=False
- OLDER: Calvin told Dave he is really talented and encouraged him to keep making people happy and doing what he loves.
  - K=['encouragement', 'talent', 'making people happy'] entities=['Calvin', 'Dave']
- NEWER: Calvin said doing what you love for a living is the ultimate goal and told Dave to keep going, calling him inspiring.
  - K=['encouragement', 'goal', 'inspiration'] entities=['Calvin', 'Dave']
- semantic verdict: **unread**

## locomo_0009 — da800066 → 9a02278c

- **cross-session**; older `s14` (12:35 am on 14 August, 2023) → newer `s19` (12:13 am on 15 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`encouraged` newer=`talented` shared=False
- OLDER: Calvin encouraged Dave, telling him he is always there for him and wishing him good luck with the car restoration.
  - K=['encouragement', 'support', 'good luck'] entities=['Calvin', 'Dave']
- NEWER: Calvin told Dave he is really talented and encouraged him to keep making people happy and doing what he loves.
  - K=['encouragement', 'talent', 'making people happy'] entities=['Calvin', 'Dave']
- semantic verdict: **unread**

## locomo_0009 — 15ff4f6a → 23af6fd9

- **cross-session**; older `s14` (12:35 am on 14 August, 2023) → newer `s23` (9:39 am on 15 October, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`thanks` newer=`thanked` shared=False
- OLDER: Dave thanked Calvin, saying Calvin's support means a lot to him.
  - K=['support', 'thanks', 'friendship'] entities=['Dave', 'Calvin']
- NEWER: Dave thanked Calvin, saying having a believing friend helps him keep pushing.
  - K=['gratitude', 'friendship', 'support'] entities=['Dave', 'Calvin']
- semantic verdict: **unread**

## locomo_0009 — 1007306f → 8ef2669a

- **same-session**; older `s12` (1:12 pm on 3 August, 2023) → newer `s12` (1:12 pm on 3 August, 2023)
- interval class: `inverted`
- head verbs: older=`dreams` newer=`motivation` shared=False
- OLDER: Dave told Calvin he is eager to see what Calvin does and urged him to keep at it and never forget his dreams.
  - K=['support', 'dreams', 'encouragement'] entities=['Dave', 'Calvin']
- NEWER: Dave told Calvin that having something like his car is motivating and a reminder of what one can achieve.
  - K=['motivation', 'car', 'encouragement'] entities=['Dave', 'Calvin']
- semantic verdict: **unread**

## locomo_0009 — 1111561d → 8fd8727d

- **cross-session**; older `s12` (1:12 pm on 3 August, 2023) → newer `s29` (9:15 pm on 13 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`artists` newer=`encouraged` shared=False
- OLDER: Dave encouraged Calvin to keep working with different artists and crafting great sounds to reach his goals.
  - K=['encouragement', 'artists', 'music'] entities=['Dave', 'Calvin']
- NEWER: Dave encouraged Calvin to keep pushing himself and exploring as an artist.
  - K=['encouragement', 'growth', 'artist', 'exploration'] entities=['Dave', 'Calvin']
- semantic verdict: **unread**

## locomo_0009 — 7d3a02ed → 27ed0323

- **cross-session**; older `s5` (1:16 pm on 3 May, 2023) → newer `s20` (8:57 pm on 22 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`sports` newer=`sports` shared=True
- OLDER: Dave reacted to Calvin's photo of a red sports car driving down a road.
  - K=['red sports car', 'photo', 'react'] entities=['Dave', 'Calvin']
- NEWER: Dave reacted to Calvin's red sports car photo, saying he loves how the sunlight hits it and that it is a great symbol of Calvin's hard work.
  - K=['red sports car', 'sunlight', 'hard work', 'symbol'] entities=['Dave', 'Calvin']
- semantic verdict: **unread**

## locomo_0009 — 0dfe2fe3 → 7fbdf08b

- **same-session**; older `s22` (3:13 pm on 8 October, 2023) → newer `s22` (3:13 pm on 8 October, 2023)
- interval class: `zero_width`
- head verbs: older=`encouragement` newer=`dreams` shared=False
- OLDER: Calvin said doing what you love for a living is the ultimate goal and told Dave to keep going, calling him inspiring.
  - K=['encouragement', 'goal', 'inspiration'] entities=['Calvin', 'Dave']
- NEWER: Calvin told Dave to go for it, that chasing dreams is what life is about, and to keep working hard and living his best life.
  - K=['encouragement', 'dreams', 'hard work'] entities=['Calvin', 'Dave']
- semantic verdict: **unread**

## locomo_0009 — 4a822cb4 → 7fbdf08b

- **same-session**; older `s22` (3:13 pm on 8 October, 2023) → newer `s22` (3:13 pm on 8 October, 2023)
- interval class: `inverted`
- head verbs: older=`focus` newer=`dreams` shared=False
- OLDER: Calvin told Dave to keep up the good work and stay focused, saying Dave has got this.
  - K=['encouragement', 'focus'] entities=['Calvin', 'Dave']
- NEWER: Calvin told Dave to go for it, that chasing dreams is what life is about, and to keep working hard and living his best life.
  - K=['encouragement', 'dreams', 'hard work'] entities=['Calvin', 'Dave']
- semantic verdict: **unread**

## locomo_0009 — 4a822cb4 → 3cedafa2

- **cross-session**; older `s22` (3:13 pm on 8 October, 2023) → newer `s23` (9:39 am on 15 October, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`focus` newer=`encouraged` shared=False
- OLDER: Calvin told Dave to keep up the good work and stay focused, saying Dave has got this.
  - K=['encouragement', 'focus'] entities=['Calvin', 'Dave']
- NEWER: Calvin encouraged Dave to keep pushing and believe in himself, saying the payoff would be worth it.
  - K=['encouragement', 'support', 'belief'] entities=['Calvin', 'Dave']
- semantic verdict: **unread**

## locomo_0009 — 2bca8152 → 51323830

- **same-session**; older `s22` (3:13 pm on 8 October, 2023) → newer `s22` (3:13 pm on 8 October, 2023)
- interval class: `inverted`
- head verbs: older=`dreams` newer=`dreams` shared=True
- OLDER: Dave told Calvin that Calvin's support means everything to him and that he will keep pushing and reaching for his dreams.
  - K=['support', 'appreciation', 'dreams'] entities=['Dave', 'Calvin']
- NEWER: Dave said Calvin's support means a lot and that he will keep chasing his dreams and working hard.
  - K=['support', 'dreams', 'hard work', 'appreciation'] entities=['Dave', 'Calvin']
- semantic verdict: **unread**

## locomo_0009 — f31d67af → 45daf9dc

- **cross-session**; older `s3` (4:15 pm on 20 April, 2023) → newer `s15` (11:06 am on 22 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`attended` newer=`attended` shared=True
- OLDER: Calvin attended an awesome music festival in Tokyo before 20 April 2023.
  - K=['music festival', 'Tokyo', 'travel'] entities=['Calvin', 'Tokyo']
- NEWER: Calvin attended a music festival in Tokyo with a lively crowd and buzzing energy.
  - K=['music festival', 'Tokyo', 'crowd'] entities=['Calvin', 'Tokyo']
- semantic verdict: **unread**

## locomo_0009 — 60899695 → e6354467

- **cross-session**; older `s2` (4:45 pm on 26 March, 2023) → newer `s9` (3:15 pm on 21 June, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`sports` newer=`rims` shared=False
- OLDER: Calvin's new car is a red sports car.
  - K=['red sports car', 'car'] entities=['Calvin']
- NEWER: Calvin owns a red car with black rims.
  - K=['red car', 'black rims', 'car'] entities=['Calvin']
- semantic verdict: **unread**

## locomo_0009 — 60899695 → 1dd97d6e

- **cross-session**; older `s2` (4:45 pm on 26 March, 2023) → newer `s20` (8:57 pm on 22 September, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`sports` newer=`sports` shared=True
- OLDER: Calvin's new car is a red sports car.
  - K=['red sports car', 'car'] entities=['Calvin']
- NEWER: Calvin shared a photo of his red sports car parked in a parking lot, calling it a sign of his hard work and dedication.
  - K=['red sports car', 'parking lot', 'hard work', 'dedication'] entities=['Calvin']
- semantic verdict: **unread**

## locomo_0009 — adc42780 → 39c7a383

- **cross-session**; older `s28` (5:46 pm on 2 November, 2023) → newer `s30` (10:54 am on 17 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`congratulations` newer=`congratulations` shared=True
- OLDER: Dave congratulated Calvin on his album release and the party.
  - K=['congratulations', 'album', 'party'] entities=['Dave', 'Calvin']
- NEWER: Dave congratulated Calvin on his gala attendance and upcoming performance in Boston.
  - K=['congratulations', 'gala', 'performance'] entities=['Dave', 'Calvin', 'Boston']
- semantic verdict: **unread**

## locomo_0009 — 0117ffe3 → d4851d9f

- **same-session**; older `s7` (6:06 pm on 31 May, 2023) → newer `s7` (6:06 pm on 31 May, 2023)
- interval class: `zero_width`
- head verbs: older=`performance` newer=`enjoys` shared=False
- OLDER: Calvin felt intensely alive while performing live.
  - K=['performance', 'emotion', 'life'] entities=['Calvin']
- NEWER: Calvin enjoys performing live and finds the rush and connection with the crowd an absolute high.
  - K=['performance', 'crowd', 'rush'] entities=['Calvin']
- semantic verdict: **unread**

## locomo_0009 — 1c8eb6d4 → 54a3178f

- **cross-session**; older `s7` (6:06 pm on 31 May, 2023) → newer `s14` (12:35 am on 14 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`performed` newer=`performed` shared=True
- OLDER: During the Frank Ocean tour the week before 31 May 2023, Calvin performed in Tokyo to an insane crowd.
  - K=['Tokyo', 'concert', 'crowd', 'performance'] entities=['Calvin', 'Tokyo']
- NEWER: Calvin performed a show in Tokyo and the crowd sang along to one of his songs.
  - K=['Tokyo', 'show', 'concert', 'crowd', 'sing along'] entities=['Calvin', 'Tokyo']
- semantic verdict: **unread**

## locomo_0009 — ae754397 → 1111561d

- **cross-session**; older `s11` (6:38 pm on 21 July, 2023) → newer `s12` (1:12 pm on 3 August, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`goals` newer=`artists` shared=False
- OLDER: Dave told Calvin he is Calvin's biggest fan and encouraged him to keep pushing toward his goals.
  - K=['support', 'encouragement', 'goals'] entities=['Dave', 'Calvin']
- NEWER: Dave encouraged Calvin to keep working with different artists and crafting great sounds to reach his goals.
  - K=['encouragement', 'artists', 'music'] entities=['Dave', 'Calvin']
- semantic verdict: **unread**

## locomo_0009 — 2538aac4 → a6ab31f5

- **same-session**; older `s29` (9:15 pm on 13 November, 2023) → newer `s29` (9:15 pm on 13 November, 2023)
- interval class: `inverted`
- head verbs: older=`ideas` newer=`asked` shared=False
- OLDER: Dave asked Calvin whether he had tried out any new ideas lately.
  - K=['new ideas', 'music', 'question'] entities=['Dave', 'Calvin']
- NEWER: Dave asked Calvin whether having supportive people around him is important for Calvin's artist development.
  - K=['support', 'artist development', 'question'] entities=['Dave', 'Calvin']
- semantic verdict: **unread**

## locomo_0009 — 01f2d6e3 → 3cedafa2

- **same-session**; older `s23` (9:39 am on 15 October, 2023) → newer `s23` (9:39 am on 15 October, 2023)
- interval class: `inverted`
- head verbs: older=`believes` newer=`encouraged` shared=False
- OLDER: Calvin told Dave he believes in him and urged him never to forget how awesome he is.
  - K=['belief', 'encouragement', 'self-worth'] entities=['Calvin', 'Dave']
- NEWER: Calvin encouraged Dave to keep pushing and believe in himself, saying the payoff would be worth it.
  - K=['encouragement', 'support', 'belief'] entities=['Calvin', 'Dave']
- semantic verdict: **unread**

## locomo_0009 — 00860890 → adc42780

- **cross-session**; older `s23` (9:39 am on 15 October, 2023) → newer `s28` (5:46 pm on 2 November, 2023)
- interval class: `older_closed_newer_open`
- head verbs: older=`congratulations` newer=`congratulations` shared=True
- OLDER: Dave congratulated Calvin on his new Ferrari, calling it an incredible ride, and asked him to share a photo when ready.
  - K=['congratulations', 'Ferrari', 'photo'] entities=['Dave', 'Calvin']
- NEWER: Dave congratulated Calvin on his album release and the party.
  - K=['congratulations', 'album', 'party'] entities=['Dave', 'Calvin']
- semantic verdict: **unread**

