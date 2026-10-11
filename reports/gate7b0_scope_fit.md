# Gate 7B0 Scope-Fit Audit

The audit uses the venue-neutral manuscript facts frozen at Gate 7A. No manuscript text or formal evidence was changed.

## Applied Intelligence

- Official scope: https://link.springer.com/journal/10489/aims-and-scope
- Retrieved: 2026-07-22
- Scope summary: Publishes new and innovative intelligent-systems methodologies and applications to complex real-life problems, including neural networks, uncertain-information processing, streaming data, and risk analysis. The scope explicitly distinguishes innovative development from applying established technology to another dataset.
- Methodology fit: PARTIAL_TO_GOOD: the neural probabilistic head and fixed missing-history protocol fit intelligent-systems methodology, but the head is deliberately lightweight and is not a new distribution or backbone.
- Forecasting fit: GOOD: recent journal content includes deep probabilistic time-series forecasting, although forecasting is one application area rather than the journal's central identity.
- Probabilistic forecasting fit: GOOD: recent official articles use predictive distributions, CRPS, calibration, and interval metrics.
- Missing-data fit: GOOD: uncertain and incomplete information are compatible with the scope; the manuscript studies missing historical observations rather than imputation as an end in itself.
- Applied-AI fit: GOOD: the work is empirical and application-oriented, but the three public forecasting benchmarks offer less application depth than a named real-life deployment.
- Negative-results fit: GOOD: fixed-mask audits, preregistered tolerances, and retained negative results strengthen reproducibility, though this is not an explicitly named article category.
- Unresolved scope blocker: False

Recent official-publication ecology:

- 2025, [VAEneu: a new avenue for VAE application on probabilistic forecasting](https://link.springer.com/article/10.1007/s10489-024-06203-5): Confirms direct probabilistic-forecasting interest while showing a recent paper with a new model, 12 baselines, and 12 datasets.
- 2024, [Generating probabilistic forecasts from arbitrary point forecasts using a conditional invertible neural network](https://link.springer.com/article/10.1007/s10489-024-05346-9): Demonstrates acceptance of distributional post-processing but also raises the novelty bar through an invertible-network method.

## Neural Computing and Applications

- Official scope: https://link.springer.com/journal/521/aims-and-scope
- Retrieved: 2026-07-22
- Scope summary: Covers practical neural computing and related techniques and explicitly lists benchmarks, intelligent forecasting, machine learning, performance measures, and applied statistics. Original Articles should present new, significant research, development, or applications with practical value.
- Methodology fit: GOOD: both TimePerceiver and PatchTST are neural forecasters, and the fixed Student-t head is evaluated as a practical neural-computing extension.
- Forecasting fit: VERY_GOOD: intelligent forecasting is explicitly named in the official scope.
- Probabilistic forecasting fit: GOOD: uncertainty and probabilistic evaluation fit neural performance assessment, though recent sampled forecasting papers are often application-specific.
- Missing-data fit: GOOD: robustness to missing historical observations is a practical neural-computing problem.
- Applied-AI fit: VERY_GOOD: the empirical benchmark and practical-value framing align directly.
- Negative-results fit: GOOD: reproducible benchmarks and performance measures are scope-compatible, even though negative-result publication is not explicitly promised.
- Unresolved scope blocker: False

Recent official-publication ecology:

- 2025, [Stock price forecasting through symbolic dynamics and state transition graphs with a convolutional recurrent neural network architecture](https://link.springer.com/article/10.1007/s00521-025-11325-z): Confirms recent publication of neural time-series forecasting with practical application framing.
- 2025, [Explainable artificial intelligence for wind power forecasting model based on long short-term memory](https://link.springer.com/article/10.1007/s00521-025-11230-5): Shows that an applied neural-forecasting contribution can fit when practical value and neural integration are explicit.

## Journal of Forecasting

- Official scope: https://onlinelibrary.wiley.com/journal/1099131x
- Retrieved: 2026-07-22
- Scope summary: A multidisciplinary forecasting journal covering theoretical, practical, computational, and methodological forecasting work across statistics, economics, psychology, systems engineering, and the social sciences.
- Methodology fit: VERY_GOOD if framed as a distributional-forecasting robustness study rather than a neural architecture paper.
- Forecasting fit: EXCELLENT: forecasting is the journal's central subject.
- Probabilistic forecasting fit: EXCELLENT: recent official articles cover data-driven predictive distributions, heavy tails, Student-t variants, proper scores, and robust prediction intervals.
- Missing-data fit: GOOD: missing histories are a forecasting-information problem, although the manuscript must explain forecasting consequences rather than focus on generic AI robustness.
- Applied-AI fit: GOOD: computational deep forecasting is in scope, but statistical interpretation and forecasting relevance should lead the presentation.
- Negative-results fit: VERY_GOOD: distributional diagnostics, robust interval evaluation, preregistered tolerances, and honest limitations support a forecasting-methodology narrative.
- Unresolved scope blocker: False

Recent official-publication ecology:

- 2025, [Taming Data-Driven Probability Distributions](https://onlinelibrary.wiley.com/doi/10.1002/for.3208): Directly supports a heavy-tail and predictive-distribution framing for forecasting readers.
- 2026, [Robust Prediction Intervals for Time Series Forecasting: A Bootstrap and Bayesian Approach](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.70126): Shows a current editorial home for robust interval and Student-t-related forecasting methodology.

## Engineering Applications of Artificial Intelligence

- Official scope: https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/about/aims-and-scope
- Retrieved: 2026-07-22
- Scope summary: Publishes practical AI in engineering and requires papers to report novel AI aspects used for a real-world engineering application, with validation on public datasets. The journal warns that abstracts must state both the AI contribution and the engineering application.
- Methodology fit: PARTIAL: the probabilistic neural head is AI methodology, but its novelty is moderate and the paper is organized as a generic forecasting benchmark rather than an engineering solution.
- Forecasting fit: GOOD only when attached to a concrete engineering problem such as load, wind, infrastructure, or industrial sensing.
- Probabilistic forecasting fit: GOOD: recent journal papers include deep probabilistic load forecasting.
- Missing-data fit: GOOD in engineering sensor settings; recent content includes missing-data imputation for sensor and infrastructure systems.
- Applied-AI fit: BLOCKED: the current manuscript has no single named real-world engineering application or deployment/decision context.
- Negative-results fit: PARTIAL: rigorous negative findings are useful, but they do not replace the required engineering application.
- Unresolved scope blocker: True

Recent official-publication ecology:

- 2025, [Transformer-based deep probabilistic network for load forecasting](https://www.sciencedirect.com/science/article/pii/S095219762500781X): Confirms probabilistic forecasting fit when the work addresses a concrete engineering-domain load application.
- 2025, [Time-frequency informed stacked long short-term memory-based generative adversarial network for missing data imputation in sensor networks](https://www.sciencedirect.com/science/article/pii/S095219762500973X): Shows that missing-data research is welcome when embedded in a concrete engineering system and validated accordingly.
