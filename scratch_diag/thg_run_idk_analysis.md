# in-progress preds: ds_thg__deepseek_v4_flash__ASEM-THG.jsonl  (165 answered)

| category | n | 'I don't know' | % idk | EM | em_loose |
|---|--:|--:|--:|--:|--:|
| adversarial | 13 | 11 | 85% | 0.0 | 15.4 |
| multi_hop | 32 | 1 | 3% | 3.1 | 12.5 |
| open_domain | 13 | 6 | 46% | 0.0 | 0.0 |
| single_hop | 70 | 7 | 10% | 4.3 | 32.9 |
| temporal | 37 | 1 | 3% | 27.0 | 48.6 |
| **total** | **165** | **26** | **16%** | **8.5** | **28.5** |

## Sample 'I don't know' answers with their gold (non-adversarial first)

- [multi_hop] Q: How many children does Melanie have?
  gold: '3'
- [open_domain] Q: Would Caroline pursue writing as a career option?
  gold: 'LIkely no; though she likes reading, she wants to be a counselor'
- [open_domain] Q: Would Melanie be considered a member of the LGBTQ community?
  gold: 'Likely no, she does not refer to herself as part of it'
- [open_domain] Q: Would Melanie be considered an ally to the transgender community?
  gold: 'Yes, she is supportive'
- [open_domain] Q: Would Caroline be considered religious?
  gold: 'Somewhat, but not extremely religious'
- [open_domain] Q: Would Melanie go on another roadtrip soon?
  gold: 'Likely no; since this one went badly'
- [open_domain] Q: Would Caroline want to move back to her home country soon?
  gold: "No; she's in the process of adopting children."
- [single_hop] Q: What is Melanie's hand-painted bowl a reminder of?
  gold: 'art and self-expression'
- [single_hop] Q: What do sunflowers represent according to Caroline?
  gold: 'warmth and happiness'
- [single_hop] Q: What inspired Caroline's painting for the art show?
  gold: 'visiting an LGBTQ center and wanting to capture unity and strength'
- [single_hop] Q: What advice does Caroline give for getting started with adoption?
  gold: 'Do research, find an adoption agency or lawyer, gather necessary documents, and prepare emotionally.'
- [single_hop] Q: What does Melanie do to keep herself busy during her pottery break?
  gold: 'Read a book and paint.'
