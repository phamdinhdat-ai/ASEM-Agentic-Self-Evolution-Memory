# Retrieval-context format — before / after

Cùng một tập note (retrieval thật trên bank `ds_fixed`), hai cách render.

## `#1941` — What advice did Calvin receive from the chef at the music festival?

- gold: `to stay true to himself and sound unique`
- note lấy về: **8**
- context cũ: **9718** ký tự (~2429 token) · context mới: **5912** ký tự (~1478 token) → giảm **39%**

<details><summary>OLD — JSON payload (UUID, 1 dòng)</summary>

```json
[{"id": "7b09e5df-5cd4-441c-b912-8f181b4868fe", "keywords": ["calvin", "advice", "music industry", "professionals", "producer", "unique sound", "music direction", "dave", "motivation", "music future", "dreams", "collaboration", "positive energy", "storytelling", "feedback", "music journey"], "description": "Calvin learned a lot and received great advice from music industry professionals, including a producer who told him to stay true to himself and sound unique, prompting him to think about his music's direction; Dave found Calvin's authenticity motivating and asked where he sees his music taking him. Calvin also said he started making music to follow his dreams and stays motivated through collaboration, learning from others, and surrounding himself with positive energy, and that moments like this remind him why he got into music \u2014 making a difference and sharing his story \u2014 giving him strength to keep going.", "session_date": "2023-04-20T16:15:00Z", "entities": ["Calvin"], "speaker": "Calvin", "relations": [{"relation": "same-topic", "target_id": "d0c39ab9-29b1-4fac-9fee-6df9f968feb2"}, {"relation": "extends", "target_id": "c2a86835-eb0c-4004-9bda-e07ac44b79d9"}], "also_linked": "extends:3, same-topic:3, semantic:1", "content": "[Calvin] I learned a lot and got some great advice from professionals in the music industry. It was inspiring!"}, {"id": "0bc04b6c-c2f6-468e-9dfc-152c5bd84d0f", "keywords": ["dave", "music festival", "concerts", "music", "connection", "memories", "crowd buzz", "repair work", "artist", "september 2023"], "description": "Dave recently returned from an amazing music festival full of energy and crowd excitement that made him feel alive, and he also recalled good times at concerts in September 2023, sharing a crowd photo and noting that music connects people and creates memories; he remarked on the amazing indescribable connection between artist and crowd and is glad Calvin gets to experience it, comparing music's bringing-together of  …
```

</details>

<details open><summary>NEW — blocked context</summary>

```text
[1] 20 April 2023 · said by Calvin · about: Calvin
      • Calvin learned a lot and received great advice from music industry professionals, including a producer
        who told him to stay true to himself and sound unique, prompting him to think about his music's
        direction; Dave found Calvin's authenticity motivating and asked where he sees his music taking him.
      • Calvin also said he started making music to follow his dreams and stays motivated through collaboration,
        learning from …
      • [+1 more fact(s) in this note, not shown]
      turn: "[Calvin] I learned a lot and got some great advice from professionals in the music industry. It was
             inspiring!"
      topics: calvin, advice, music industry, professionals, producer, unique sound, music direction, dave
      links: [5] same-topic, [6] extends  (+7 more not in this list: extends:3, same-topic:3, semantic:1)

[2] 15 October 2023 · said by Dave · about: Dave, Calvin
      • Dave recently returned from an amazing music festival full of energy and crowd excitement that made him
        feel alive, and he also recalled good times at concerts in September 2023, sharing a crowd photo and
        noting that music connects people and creates memories; he remarked on the amazing indescribable
        connection between artist and crowd and is glad Calvin gets to experience it, comparing music's
        bringing-together of …
      • [+1 more fact(s) in this note, not shown]
      turn: "[Dave] That's a great approach, Cal! Reminding yourself of the passion for the goals and getting help
             from others is really important. Taking a break and having fun sounds so refreshing. Oh, I just …"
      topics: dave, music festival, concerts, music, connection, memories, crowd buzz, repair work
      links: [7] extends  (+10 more not in this list: same-topic:6, extends:4)

[3] 15 September 2023 · said by Calvin · about: Calvin, Dave, Disney
      • Calvin expressed disappointment about the missing recording, reflected that some memories can't be
        captured, and shared a Disney poster image.
      turn: "[Calvin] Aww, bummer! I would've loved to hear that music. Oh well, some of the best memories can't be
             captured on video or audio. It's like those special moments that stay in our hearts and minds …"
      topics: calvin, memories, disney poster
      links: [8] causal  (+7 more not in this list: same-topic:4, semantic:3)

[4] 4 October 2023 · said by Dave · about: Dave, Calvin
      • Dave responded enthusiastically to Calvin's artist meeting and asked how the meeting was arranged and
        whether a photo showed musicians Calvin is collaborating with; Calvin explained that a mutual friend
        connected him with the artists, and he now reports great collaborations, an almost-finished album, and
        plans to send previews and arrange a catch-up.
      • Calvin also describes an inspiring conversation with an artist at …
      • [+1 more fact(s) in this note, not shown]
      turn: "[Dave] Awesome, Calvin! Connecting with all those talented artists must have been an inspiring
             experience. Can't wait to hear what you come up with in your collaboration. Let me know how it goes! …"
      topics: dave, calvin, artists, collaboration, meeting arrangement, mutual friend, album, previews
      links: [7] same-topic  (+33 more not in this list: same-topic:16, semantic:11, extends:4, temporal:2)

[5] 23 March 2023 · said by Calvin · about: Dave, Calvin, Japan
      • Calvin said his agent found him the place and expressed gratitude.
      turn: "[Calvin] Wow, my agent found me this awesome place, so thankful!"
      topics: calvin, agent, accommodation, gratitude
      links: [1] same-topic, [6] semantic  (+12 more not in this list: semantic:6, same-topic:4, extends:2)

[6] 20 April 2023 · said by Calvin · about: Calvin
      • Calvin shared that a producer advised him to stay true to himself and sound unique, prompting him to
        think about his music's direction; Dave found this motivating and asked where Calvin sees his music
        taking him.
      • Calvin also said embracing nature and learning about Japanese culture have been calming, and he has been
        experimenting with different genres and adding electronic elements to his songs as an exciting process …
      • [+4 more fact(s) in this note, not shown]
      turn: "[Calvin] The producer gave me some advice to stay true to myself and sound unique. It got me thinking
             about where I want my music to go. It's really motivating!"
      topics: calvin, producer, advice, unique sound, music direction, dave, motivation, creativity block
      links: [1] extends, [5] semantic  (+22 more not in this list: extends:13, same-topic:8, semantic:1)

[7] 23 October 2023 · said by Dave · about: Dave, Calvin
      • Dave remarked on how amazing the indescribable connection between artist and crowd is and said he is
        glad Calvin gets to experience it.
      turn: "[Dave] Wow, it's amazing how that connection between artist and crowd can be indescribable. So glad you
             get to experience that!"
      topics: artist, crowd, connection, experience
      links: [4] same-topic, [2] extends  (+3 more not in this list: extends:2, same-topic:1)

[8] 15 September 2023 · said by Dave · about: Dave, Calvin, Boston
      • Dave said they forgot to record the jam because they were too absorbed in playing; Calvin expressed
        disappointment about the missing recording, reflected that some memories can't be captured, and shared a
        Disney poster image.
      turn: "[Dave] Hey Calvin! I wish we had recorded the jam, but we were way too into it and totally forgot."
      topics: dave, jam, recording, forgot, calvin, memories, disney poster
      links: [3] causal  (+7 more not in this list: same-topic:5, semantic:2)
```

</details>
