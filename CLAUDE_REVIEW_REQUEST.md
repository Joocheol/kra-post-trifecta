# Independent scope review: extending Snowberg and Wolfers (2010)

## Status and reviewer stance

This file is a review request, not an accepted research-plan change and not evidence for
the paper's conclusions. Review the proposal as a skeptical referee trying to identify the
strongest reason to reject it. Do not agree with the proposed framing merely because it is
presented here. Separate what follows from the currently locked design in
`RESEARCH_PLAN.md`.

## Decision under review

Should the project be framed as the following extension?

> Extend Snowberg and Wolfers' multi-pool identification strategy to a market in which the
> complete vector of ordered top-three trifecta odds is observed, using that vector to test
> cross-pool compatibility with win, exacta, quinella, and trio prices and to study reduction
> versus sequential evaluation of a three-stage compound lottery.

The intended contribution is not a Korean replication of the favorite-longshot bias. It is
the use of the full ordered top-three outcome space to connect several separately cleared
parimutuel pools through fixed marginalization operators, followed by a behavioral layer
that asks whether one risk-preference model, one probability-weighting model, or a
sequential non-reduction model can rationalize prices across all pools.

## Evidence that must anchor the comparison

### Snowberg and Wolfers (2010)

The paper fits risk-love and probability-misperception explanations to win-bet data and
asks which model better predicts exotic-bet prices. Its data record the winning payoffs in
exacta, quinella, and trifecta races; the prices of nonwinning exotic combinations are not
recorded (paper Appendix, Data). It predicts the observed winning exotic payoff using win
odds and conditional finishing probabilities, first under Harville's conditional-independence
assumption and then using conditional probabilities estimated from outcomes. It also studies
the simultaneous relative pricing of the winning exacta and corresponding quinella. Its
misperception interpretation is tied to a particular sequential failure to reduce compound
lotteries.

### Koivuranta and Korhonen (2019)

This is the closest follow-up and must not be omitted from the novelty assessment:
"Misperception explains favorite-longshot bias: evidence from the Finnish and Swedish
harness horse race markets," *Empirical Economics* 57, 2149-2160,
doi:10.1007/s00181-018-1538-0. The publisher abstract says that the data contain a complete
set of odds for exotic markets; the authors use exotic and win odds to favor misperception
over risk-love and find evidence that bettors evaluate first and second place sequentially
rather than reducing the exotic event to a simple lottery. The reviewer must determine
precisely what remains new after this paper.

### KRA data and the current design

- Separately cleared pools are observed for win, place, quinella, quinella-place, exacta,
  trifecta, and trio bets.
- The candidate date-restricted sample contains 19,301 races from 2016-06-10 through
  2025-12-31, excluding 2020-2021 and 2018-07-01. The final analysis sample is not yet
  established by reproducible sample-flow output.
- For a race with valid horse set \(N_r\), the complete trifecta vector covers every ordered
  state \((i,j,k)\) with distinct horses. The current design normalizes reciprocal total-payout
  odds within each pool and applies deterministic marginalization matrices to reconstruct
  win, exacta, quinella, and trio price vectors.
- KRA odds are total payout multiples \(D\), so the reciprocal-price convention is \(1/D\).
  Snowberg and Wolfers report net odds \(O/1\), whose contingent-claim price is
  \(1/(O+1)\). Treating those conventions as identical would be an error.
- The current primary estimand is internal cross-pool price compatibility, not objective
  probability, causal information aggregation, profitability, or individual bettor choice.
- Off-repository preliminary calculations, not yet auditable from this PR, report median
  race-level auxiliary OLS \(R^2\) values of 0.982 (win), 0.976 (exacta), 0.972 (quinella),
  and 0.988 (trio) when reconstructing from trifecta prices. Same-field-size race
  permutations reduce the corresponding fit to roughly 0.003-0.009. These numbers are
  motivation only; do not validate them or infer a common price measure from them.

## User-provided literature packet

The review package supplied outside the public repository contains the following papers.
Use this list to locate the relevant line of argument in the manuscript and bibliography;
do not assume that citing all of them is necessary.

1. Griffith (1949), "Odds Adjustments by American Horse-Race Bettors."
2. Weitzman (1965), "Utility Analysis and Group Behavior: An Empirical Study."
3. Ali (1977), "Probability and Utility Estimates for Racetrack Bettors."
4. Busche and Hall (1988), "An Exception to the Risk Preference Anomaly."
5. Shin (1991), "Optimal Betting Odds Against Insider Traders."
6. Vaughan Williams and Paton (1997), "Why is There a Favourite-Longshot Bias in
   British Racetrack Betting Markets?"
7. Vaughan Williams and Paton (1998), "Why are some favourite-longshot biases positive
   and others negative?"
8. Jullien and Salanie (2000), "Estimating Preferences under Risk: The Case of Racetrack
   Bettors."
9. Bruce and Johnson (2000), "Investigating the Roots of the Favourite-Longshot Bias:
   An Analysis of Decision Making by Supply- and Demand-Side Agents in Parallel Betting
   Markets."
10. Walls and Busche (2003), "Broken odds and the favourite-longshot bias in parimutuel
    betting: a direct test."
11. Coleman (2004), "New light on the longshot bias."
12. Winter and Kukuk (2006), "Risk Love and the Favorite-Longshot Bias: Evidence from
    German Harness Horse Racing."
13. Sung, Johnson, and Peirson (2012), "Discovering a Profitable Trading Strategy in an
    Apparently Efficient Market: Exploiting the Actions of Less Informed Traders in
    Speculative Markets."
14. Suhonen, Saastamoinen, and Linden (2018), "A dual theory approach to estimating risk
    preferences in the parimutuel betting market."
15. Jeong, Kim, and Ro (2019), "On the efficiency of racetrack betting market: a new test
    for the favourite-longshot bias."
16. Snowberg and Wolfers (2010), "Explaining the Favorite-Long Shot Bias: Is it Risk-Love
    or Misperceptions?"

## Questions the review must answer

1. Relative to both Snowberg-Wolfers and Koivuranta-Korhonen, what is genuinely new?
   Is “ordered top-three states plus five pools” a substantive economic contribution or only
   a larger accounting exercise?
2. Is marginalizing normalized reciprocal trifecta odds across separately cleared pools an
   economically justified cross-market restriction, or merely an arithmetic comparison of
   different pool-specific pricing measures? State the assumptions needed for a common
   measure and the interpretation available without them.
3. Can this design identify risk preferences, probability misperceptions, and compound-
   lottery non-reduction? If not, specify which objects are not identified and what additional
   structural restrictions, outcome data, or variation would be required.
4. What is the correct three-stage analogue of Snowberg and Wolfers' sequential
   non-reduction model? Which comparisons would distinguish (a) reduction to the joint
   top-three event, (b) first-place then conditional second/third evaluation, and (c) a generic
   rank-dependent or probability-weighting model?
5. Could high \(R^2\) arise mechanically from shared popularity rankings, dimensionality,
   normalization, or common public information? Evaluate whether TV, Jensen-Shannon,
   calibration, log-ratio/compositional metrics, Harville/Plackett-Luce, within-race
   permutation, other-race, and uniform benchmarks are sufficient.
6. Assess the threats from odds caps (including 9999.9), rounding/truncation, zero-winner
   payout rules, incomplete support, different takeout, pool liquidity, late betting, and
   different bettor populations. Which are fatal, partially identifiable, or manageable by
   robustness analysis?
7. Give the strongest plausible rejection report and the minimum design that would survive
   it. End with four explicit lists: claims to retain, claims to weaken or drop, analyses to add,
   and recommended framing.

## Required review output

Use the repository's validated JSON review format. In the summary, give an overall verdict
(`reject as framed`, `promising but major redesign`, or `defensible extension`) and directly
answer the seven questions above. Put concrete design deficiencies in findings. Put only
genuine author choices in `author_questions`. Do not edit files or assume that the proposed
scope has already been adopted.
