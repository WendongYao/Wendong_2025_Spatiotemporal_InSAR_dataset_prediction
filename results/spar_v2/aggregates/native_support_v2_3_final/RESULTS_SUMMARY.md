# CAGEO v2.3 Final Result-to-Claim Summary

Generated from `R098_v23_aggregates` and the prelaunch-locked R099 regional
replication. All values below are machine-traceable through `summary.json` and
`manifest.json`.

## Evidence hierarchy

1. **Primary confirmation:** E32N34, 300 history values, target 2022-12-17,
   spatial partitions 47--50. The all-cells sampler, SPAR architecture, TCN
   architecture, optimizer, and stopping rules were locked before these results.
2. **Same-origin regional replication:** independently trained E29N33, E36N31,
   and E37N41 models on partitions 47--50 under the same final SPAR sampler.
3. **Exploratory temporal robustness:** E32N34 partition 47, four target dates
   and a common 240-history window. This differs from the 300-history primary
   task and is not additional confirmation evidence.
4. **Known-truth diagnostic:** ten support/noise realizations of the
   matched-IDW pseudo-target control. It diagnoses evaluation confounding; it
   does not establish realism of the analytic field.
5. Archived seeds 42--46 and raster-backbone results remain descriptive
   development/legacy evidence and must not be pooled with the primary
   confirmation set.

## Primary locked confirmation

| Model | RMSE (mm) | MAE (mm) | Core time (s) | Parameters |
|---|---:|---:|---:|---:|
| Persistence | 1.6318 +/- 0.3037 | 1.0131 +/- 0.2133 | <0.001 | 0 |
| DLinear | 1.3253 +/- 0.2623 | 0.8162 +/- 0.1866 | 46.04 +/- 4.56 | 602 |
| LASSO | 1.4193 +/- 0.2696 | 0.8894 +/- 0.2014 | 7.59 +/- 0.29 | 301 |
| Causal TCN | 1.3899 +/- 0.2835 | 0.8552 +/- 0.1954 | 145.35 +/- 5.30 | 21,953 |
| SPAR | **0.8266 +/- 0.1334** | **0.5623 +/- 0.0981** | 68.17 +/- 5.67 | 33,210 |

SPAR reduces the ratio-of-means RMSE by 37.63% relative to DLinear, 41.76%
relative to LASSO, 40.53% relative to the causal TCN, and 49.34% relative to
persistence, with 4/4 paired wins in every comparison. The Nadeau--Bengio
corrected interval for the DLinear-minus-SPAR difference is 0.1924 to 0.8051 mm
and the corrected diagnostic p-value is 0.0140. At n=4, the exact one-sided
Wilcoxon p-value is 0.0625 and must not be described as conventionally
significant.

SPAR takes 1.48 times the DLinear core time and 8.99 times the LASSO core time,
but 0.47 times the TCN time. The paper may claim a substantially improved
accuracy--cost trade-off relative to the rejected CNN--LSTM pipeline, but not
that SPAR is the cheapest method.

## Sampler ablation

On the disclosed development partition, the legacy capped sampler retains
58,103/89,865 training cells (64.66%) and obtains 0.8331 mm RMSE. Uniform
all-cells training retains 100% and obtains 0.7400 mm; density-balanced
all-cells training also retains 100% and obtains 0.7892 mm. The preregistered
simplicity rule therefore selects uniform all-cells training. These are
development results and support removal of the inherited patch sampler; they
are not confirmatory effect estimates.

## Final-configuration anchor ablation

Across confirmation partitions 47--50, final SPAR obtains 0.8266 +/- 0.1334 mm
RMSE and the otherwise identical zero-anchor network obtains 0.8620 +/- 0.1496
mm. The anchor lowers the ratio-of-means RMSE by 4.10% and wins 4/4 pairs. The
corrected interval for the no-anchor-minus-anchor difference is -0.0174 to
0.0881 mm (corrected p=0.123; exact one-sided Wilcoxon p=0.0625). The anchor
therefore supplies a directionally consistent but modest empirical gain; these
four overlapping partitions do not establish a universal optimization advantage
or statistical confirmation.

## Final-configuration regional replication

| Tile | DLinear RMSE | LASSO RMSE | SPAR RMSE | SPAR vs DLinear | Wins |
|---|---:|---:|---:|---:|---:|
| E29N33 | 1.4224 +/- 0.0593 | 1.5547 +/- 0.0702 | **0.8789 +/- 0.0354** | 38.21% | 4/4 |
| E36N31 | 1.0480 +/- 0.1274 | 1.3454 +/- 0.1602 | **0.5768 +/- 0.0562** | 44.96% | 4/4 |
| E37N41 | 1.5920 +/- 0.1116 | 1.5976 +/- 0.1053 | 1.5704 +/- 0.1096 | 1.36% | 3/4 |

The E37N41 corrected interval against DLinear crosses zero (-0.0343 to
0.0775 mm; corrected p=0.306). It is an explicit operational boundary, not
evidence of a practically important gain. These experiments are independently
trained same-origin tile replications, not cross-region transfer or Europe-wide
population inference.

## Temporal robustness boundary

With a common 240-history window across four target dates, mean RMSE is 1.5111
mm for SPAR, 1.4850 for DLinear, 1.4814 for the causal TCN, and 1.5489 for
LASSO. SPAR beats LASSO at 4/4 origins but DLinear and TCN at only 1/4; its
ratio-of-means RMSE is 1.75% and 2.00% worse than DLinear and TCN, respectively.

Therefore:

- supported: SPAR is competitive under shortened-history temporal shifts and
  retains a small aggregate advantage over LASSO;
- not supported: SPAR is universally best across forecast origins;
- required qualification: the large primary gain is specific to the tested
  300-history, final-origin task and cannot be attributed to time alone because
  the robustness protocol also shortens the history.

## Matched-interpolation known-truth control

Across ten realizations, pseudo-target RMSE is 0.2286 +/- 0.0695 mm while
analytic-truth RMSE is 0.5401 +/- 0.2619 mm. The optimism gap is positive at
10/10 realizations, with mean 0.3115 +/- 0.2691 mm and range 0.0266--0.8565 mm.
This supports the claim that matched input--target interpolation can make a
model look substantially more accurate against its interpolator-generated
supervision than against an independent field. It does not support a universal
ranking of interpolation methods.

## Claim gate

| Proposed claim | Verdict | Required wording |
|---|---|---|
| Forecasting should be evaluated first on delivered native Level-3 support | Supported | Scope to the tested EGMS Level-3 task |
| The final SPAR depends on an inherited patch sampler | Rejected | State that all native training cells are used uniformly |
| The LASSO anchor supplies part of the final gain | Partially supported | Report 4.10% and 4/4 wins together with the interval crossing zero |
| Final SPAR outperforms direct baselines on the primary full-history task | Supported | Use only partitions 47--50 as primary confirmation |
| SPAR has useful same-origin regional replication | Supported with boundary | Report all three tiles and the small E37N41 effect |
| SPAR is generally superior across forecast dates | Not supported | Report the 240-history counter-results explicitly |
| Matched interpolation can produce optimistic pseudo-target scores | Supported | Use ten-realization distribution, not only seed 42 |
| SPAR is universally best or Europe-wide validated | Not supported | Prohibit this language |

## Paper storyline

The defensible story is a **support-preserving forecasting method paper**, not a
benchmark paper: define the endpoint at native EGMS Level-3 support, use a
LASSO-anchored full-history neural residual with all-cell training, demonstrate
locked confirmation and heterogeneous regional replication, and treat dense
reconstruction as an interpolation-conditional secondary product. The temporal
and analytic controls delimit when the method advantage does and does not hold.
