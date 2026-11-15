# Presentation video script (limit 7:00)

Built for an audience new to multi-agent RL. Time plan: slides 1-6 in under 2 minutes, slides 7-11 plus a 60-second demo
(the existing methods' failure, why, our fix, its results), slides 12-17 in about a minute.
Timestamps assume about 145 words a minute; real delivery with pauses lands near 6:30.

Suggested speakers: Manaswin 1-7, Bhuvan 8-11 and the demo, Harshitha 12-14, Harshith 15-17 (adjust as you like).

## Recording checklist

1. Close PowerPoint's other copies; open `submission/AFML_Final_Presentation.pptx` → **Record → From Beginning**,
   then File → Export → Create a Video. Or record a screen share in OBS or Zoom.
2. **Demo (after slide 10, 60 s):** have `docs/credit_viz.html` open in a browser beforehand, already on
   delay 25 · slack · test episode #1. Follow the numbered clicks in slide 10's notes.
3. One dry run with a timer. If over 6:45, shorten the demo to steps 1-3.
4. Upload to YouTube (Unlisted) or Google Drive (anyone with the link) and put the link in the README.

## Slide 1: Title

[0:00 – 0:07]  Hi, we are Team Aporia: Bhuvan, Manaswin, Harshitha and Harshith. Our project: Who Earned the Reward?

## Slide 2: Topic

[0:07 – 0:31]  In reinforcement learning, an agent learns by trial and error: it acts, gets points, and does more of what earned points. A team of agents often shares one reward, like a group project with one grade, though only some members did the work. Working out who earned it is called credit assignment. Do popular methods get it right?

## Slide 3: Key ideas in plain English

[0:31 – 0:53]  Some words we'll use. Idle agents help with nothing, so they should get zero credit. A critic is a neural network that predicts how many points the team will get. The equal split is the baseline to beat, and credit error measures how far a method is from the truth: zero is perfect.

## Slide 4: Uniqueness

[0:53 – 1:04]  Methods are judged only by team score, because real tasks have no answer key. So we built games with one, tested six popular methods, and found a fix.

## Slide 5: Dataset

[1:04 – 1:13]  There's no downloaded dataset: the data comes from simulated games, because finding the truth means re-playing games with agents taken out.

## Slide 6: Overall design or approach in a free hand diagram

[1:13 – 1:37]  Our example: A pulls a lever that opens a gate, later B and C push a crate together, and the points come even later. D and E just wander. So A, B and C each earned a third. True credit comes from re-playing each game with agents taken out: it is how much reward is lost without you.

## Slide 7: Final results so far

[1:37 – 2:04]  So how do the popular methods do? Up means more wrong. Left to right, the reward arrives later. The dashed line is the equal split. With no idle agents it is exactly right, at zero, yet every method is well above it. Overall only 5 of 96 combinations beat just splitting the reward equally, and it gets worse with delay and with more idle agents.

## Slide 8: Why existing methods fail: they ask the wrong question

[2:04 – 2:39]  Why do they fail? They ask the wrong question. Their credit means: what if A paused for one step? A opened the gate early, so it would just open it a step later. Nothing is lost, so A gets almost no credit. The question that matters is: what if A weren't there at all? Then the gate stays shut and all ten points are lost, so A earned a third. They are also blind to teamwork; our first method, a coalition estimator, does find the groups.

## Slide 9: Our fix: credit by intervention

[2:39 – 3:40]  So we make the methods ask the right question, and give them the data to answer it. Like a coach benching a player in practice: while collecting training games, a random agent or group sometimes sits out the rest of the game. A neural network, the critic, learns how many points the team will still get, given who is still playing. Then an agent's credit is simply points with you minus points without you. Why does this fix A? Before A pulls the lever, if A stopped the team would get nothing, so the team needs A completely. The moment A pulls it, stopping A no longer matters. That change is A's credit, whether or not A had slack. Both pieces matter: without the sitting-out games the same critic fails, and an existing method given the same games but asking its old one-step question still fails.

## Slide 10: Our fix gets the credit right

[3:40 – 4:12]  Does it work? On the left, one real test game. The blue line is the credit our method gives A over time. It jumps to exactly a third at the step A pulls the lever, because after that the team no longer needs A. QMIX, in orange, never gets close. On the right, across all twelve settings: every existing method has an error between 0.5 and 2. Ours is below 0.07, and finds the right team every time.
[4:12 – 5:12]  DEMO, 60 s (open docs/credit_viz.html; choose delay 25, enabler slack, test episode #1). 1) Method QMIX, panel 'Who got the credit': say 'grey is the truth; QMIX gives D and E credit and A very little'. 2) Click 'Ours: trained with sitting out': 'our bars sit on the truth: a third each for A, B, C, nothing for D and E'. 3) Per-step heatmap: 'and A's credit lands exactly when it pulls the lever'. 4) Press Play on the episode replay to show the agents moving. 5) Click 'Ours: coalition estimator' and point at the coalition panel: 'it finds the group A, B, C'. 6) Table at the bottom: 'on all 30 test games, ours has the lowest error'.

## Slide 11: Our fix generalises, with limits

[5:12 – 5:31]  It also generalises: near exact when starting positions change every game, and the right team at team sizes it never trained on. In a standard foraging game it clearly beats the equal split with four agents; with six it is better, but far from exact.

## Slide 12: Done vs Remaining to be done?

[5:31 – 5:34]  Everything planned is done; bigger games are next.

## Slide 13: Team Member Contribution

[5:34 – 5:54]  Our contributions. [Each member: one short sentence.]

## Slide 14: Quantity and quality of work

[5:54 – 5:57]  Every part runs end to end, with tests.

## Slide 15: Top learning points

[5:57 – 6:03]  Our main lesson: ask the right question, and the right data beats a fancier model.

## Slide 16: Top unresolved challenges

[6:03 – 6:08]  Open problems: bigger games, and settings where agents can't sit out.

## Slide 17: Reference papers, if any

[6:08 – 6:10]  These are our references. Thank you.
