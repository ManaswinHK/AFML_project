# Recording script: four takes (≈ 6:50 total, limit 7:00)

| Take | Slides | Target |
|---|---|---|
| 1 | 1 | 0:25 |
| 2 | 2–5 | 2:10 |
| 3 | 6–11 + demo | 3:05 |
| 4 | 12–17 | 1:10 |

*[Brackets]* are actions, not words to say.

**Before take 1**
- PowerPoint: Slide Show → Set Up Slide Show → "Browsed by an individual (window)" → F5, docked on the right half.
- Browser on the left half: `docs/credit_viz.html`, bookmarks bar hidden (Ctrl+Shift+B), zoom about 80%, set to **delay 25 · slack · episode #1**.
- Do Not Disturb on. In Clipchamp, record Full screen, one recording per take.

---

## Take 1: Introduction (≈ 25 s)

**Slide 1**

Hi, we are Team Aporia: Daruru Bhuvan Sai, H K Manaswin, Harshitha B T and Harshith R, from PES University. Our project is called "Who Earned the Reward?". It checks whether teams of learning agents give credit to the agents who actually earned a shared reward, and when they don't, it fixes that.

---

## Take 2: Basics (≈ 2:10)

**Slide 2: Topic (≈ 45 s)**

First, the basics. In reinforcement learning, an agent learns by trial and error. It takes an action, gets points, called a reward, and over many tries it does more of what earned points. Now put several agents in one team that shares a single reward. It's like a group project with one grade: often only some members did the work, but everyone gets the same mark. If each agent learns from that shared mark, the ones who did nothing are told they did well. Deciding who actually earned the reward is called credit assignment. Our question is simple: do popular methods give the credit to the right agents?

*[Next slide]*

**Slide 3: Key ideas (≈ 45 s)**

A few words we'll use. An episode is one play-through of the game. A delayed reward arrives some time after the useful actions, which makes it hard to tell who caused it. Idle agents move around but help with nothing, so they should get zero credit. A critic is a neural network that predicts how many points the team will end up with; most methods read each agent's credit off it. The equal split gives everyone the same share. It needs no learning, so any method should beat it. And credit error measures how far a method's split is from the true split: zero is exactly right, two is completely wrong.

*[Next slide]*

**Slide 4: Uniqueness (≈ 20 s)**

What's new? Methods are normally judged only by the team's score, because real tasks have no answer key. But a team can score perfectly while giving credit to the wrong agents. So we built games with an answer key, tested six popular methods against it, and found a fix.

*[Next slide]*

**Slide 5: Dataset (≈ 20 s)**

There's no downloaded dataset. In reinforcement learning the data comes from playing a game in a simulator. We use our own gridworld and a public benchmark, Level-Based Foraging. A fixed dataset wouldn't work, because finding the true credit means re-playing each game with agents taken out.

---

## Take 3: The problem, our fix, and the demo (≈ 3:05)

**Slide 6: Design (≈ 30 s)**

Here's our game. *[Browser: press Play on the episode replay.]* A, in blue, pulls a lever that opens a gate. Later B and C, in orange, must push a crate together, and the ten points arrive even later. D and E, in grey, just wander. So A, B and C each earned a third. To get the true credit, we re-play the same game with each agent taken out. Without A, B or C the team scores nothing; without D or E it still scores ten. True credit is the reward lost without you.

*[Next slide]*

**Slide 7: Results of existing methods (≈ 20 s)**

How do popular methods do? Up means more wrong, and to the right the reward arrives later. The dashed line is the equal split. Only 5 of 96 combinations beat simply splitting the reward equally, and it gets worse with delay and with more idle agents.

*[Next slide]*

**Slide 8: Why they fail (≈ 28 s)**

Why? They ask the wrong question. Their credit means: what if A paused for one step? A opened the gate early, so it would just open it a step later. Nothing is lost, so A gets almost no credit. The real question is: what if A weren't there at all? Then the gate stays shut and all ten points are lost, so A earned a third.

*[Next slide]*

**Slide 9: Our fix (≈ 40 s)**

Our fix gives the critic the right question and the data to answer it. Like a coach benching a player in practice: while collecting training games, a random agent or group sits out for the rest of the game. The critic learns how many points the team will still get, given who is still playing. An agent's credit is points with you minus points without you. Before A pulls the lever, the team without A gets nothing; right after, it no longer needs A. That jump is A's credit. Both pieces matter: without the sitting-out games, or with the old one-step question, it fails.

*(If you're running long, cut the last sentence.)*

*[Next slide]*

**Slide 10: Results of our fix, with the demo (≈ 47 s)**

Does it work? Across all twelve settings, existing methods have an error between 0.5 and 2. Ours is below 0.07 and finds the right team every time. Here is one real game. *[Browser: click QMIX, scroll to "Per-step credit".]* The top row is QMIX, the bottom is the true credit. They don't match, and D and E get credit. *[Click "Ours: trained with sitting out".]* Now the top matches the bottom: A's credit lands at the lever pull, B and C's at the push, and nothing for D and E.

*[Next slide]*

**Slide 11: Generalisation (≈ 20 s)**

It also generalises: near exact when starting positions change every game, and the right team at team sizes it never trained on. In a standard foraging game it clearly beats the equal split with four agents; with six it is better, but far from exact.

---

## Take 4: Wrap-up (≈ 1:10)

**Slide 12: Done vs remaining (≈ 15 s)**

Everything we planned is done. We built a game with a known answer, tested six popular methods against it, and found that almost none beat an equal split. Our fix gets the credit right in all twelve settings and generalises. What's left: bigger games, and settings where agents can't sit out.

*[Next slide]*

**Slide 13: Contributions (≈ 20 s)**

Our contributions, one line each.

- **Bhuvan:** I designed and built our fix, credit by intervention, with the agent-removal experiments and the learned credit estimators.
- **Manaswin:** I set up the problem, built the game and the true-credit calculation, the evaluation measures, and the coalition estimator.
- **Harshitha:** I tested whether the methods find the right groups of agents, including overlapping and partial teams.
- **Harshith:** I ran the existing methods, VDN, QMIX, COMA and MAPPO, and measured how far their credit is from the truth.

*(About 20 s for all four; if one speaker records the take, read the four lines in turn.)*

*[Next slide]*

**Slide 14: Quality of work (≈ 5 s)**

Every part runs end to end, with tests, and all results are in the repository.

*[Next slide]*

**Slide 15: Learning points (≈ 15 s)**

What we learned: a high team score doesn't mean credit is fair. The question a method asks decides what it can learn, and the right data beats a bigger model. And writing predictions down first kept us honest: a review caught claims that didn't hold, and we corrected them.

*[Next slide]*

**Slide 16: Challenges (≈ 12 s)**

Open problems: checking the true credit gets expensive in big games; our fix needs agents that can sit out in training; and when luck decides a single game, no critic can say who was lucky.

*[Next slide]*

**Slide 17: References**

These are our references. Thank you.

---

**After recording:** in Clipchamp, put the four clips on the timeline in order, trim the dead air at each end, check the total is under 7:00, and export at 1080p.
