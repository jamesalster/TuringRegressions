#!/usr/bin/env Rscript
# brms reference fits for the TuringRegressions test suite.
#
#   Rscript benchmarks/brms.R            # all models
#   Rscript benchmarks/brms.R normal_iris ranef_corr_sleep   # named subset
#   Rscript benchmarks/brms.R --data-only                    # rebuild data/, no fits
#
# Writes, relative to this directory:
#   data/*.csv                the exact datasets fitted, so Julia reads the same bytes
#   reference/<model>/        one directory per model — COMMITTED, this is the oracle
#       params.csv            posterior summary, one row per parameter
#       predictions.csv       epred/linpred/pp summaries for a sample of rows
#       priors.csv            the prior brms actually used
#       model.csv             formula, family, n, budget, brms version, divergences
#   fits/*.rds                cached Stan fits (gitignored); delete to force a refit
# data/ and fits/ are gitignored — regenerate data with --data-only.
#
# WHY THE PRIORS LOOK ODD: TuringRegressions standardises X (and y, for
# identity-link families) inside `fit!`, and its priors are declared on that
# standardised scale. brms is fitted on RAW data here, so every prior is
# translated to the raw scale by the algebra in src/unstandardise.jl:
#   beta_raw  = beta_std * sd_y / sd_x_j     -> normal(0, 2 * sd_y / sd_x_j)
#   alpha_raw = mean_y + sd_y * alpha_std    -> normal(mean_y, 5 * sd_y) on class Intercept
#                                               (brms's Intercept is at centred X, as ours is)
#   sigma_raw = sigma_std * sd_y             -> exponential(1 / sd_y)
#   sd_raw    = sd_std * sd_y / sd_x_col     -> exponential(sd_x_col / sd_y)
#   shape     = 1 / phi, phi ~ Exponential(1) -> inv_gamma(1, 1)
# For a CORRELATED group-level term the map is not diagonal: the intercept row of
# A picks up -mean_x * slope_scale, so both the correlation and the intercept SD get
# mixed on the way back. Neither the lkj() nor the sd() translation above is then
# exact. Both are exact when the group-level predictor is mean-zero, which is why
# `Days_c` exists; the raw-`Days` model is kept as a deliberate mismatch.

suppressPackageStartupMessages({
  library(brms)
  library(lme4)
  library(posterior)
})

here <- function(...) file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])), ...)
dir.create(here("data"), showWarnings = FALSE, recursive = TRUE)
dir.create(here("fits"), showWarnings = FALSE, recursive = TRUE)

CHAINS <- 4
ITER <- 3000
WARMUP <- 1000  # 2000 kept draws per chain, 8000 total: the reference must be
                # tighter than the fits it is used to check
SEED <- 20260814
N_PRED_ROWS <- 20

# ---- data ------------------------------------------------------------------
# Every factor is releveled alphabetically: StatsModels sorts the levels of a
# String column, R keeps the order it was given (Titanic's Sex is Male-first).
# Without this the two sides pick different reference levels and every dummy
# coefficient silently disagrees.
normalise_factors <- function(df) {
  for (nm in names(df)) {
    if (is.factor(df[[nm]]) || is.character(df[[nm]])) {
      df[[nm]] <- factor(as.character(df[[nm]]), levels = sort(unique(as.character(df[[nm]]))))
    }
  }
  df
}

write_data <- function(df, name) {
  df <- normalise_factors(df)
  write.csv(df, here("data", paste0(name, ".csv")), row.names = FALSE)
  attr(df, "dataname") <- name  # so model.csv can name the file Julia must load
  df
}

set.seed(SEED)

mtcars_df <- mtcars
mtcars_df$Cyl <- mtcars$cyl; mtcars_df$Disp <- mtcars$disp
mtcars_df$MPG <- mtcars$mpg; mtcars_df$HP <- mtcars$hp
mtcars_df$Binom <- as.integer(mtcars$mpg > 20)
mtcars_df <- mtcars_df[, c("MPG", "Cyl", "Disp", "HP", "Binom")]
mtcars_df <- write_data(mtcars_df, "mtcars")

# n=12: small enough that the standardised-scale prior visibly dominates the
# likelihood. The point of this fixture is that both tools shrink the SAME way.
mtcars_lown <- write_data(mtcars_df[1:12, ], "mtcars_lown")

iris_df <- iris
names(iris_df) <- c("SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species")
iris_df <- write_data(iris_df, "iris")

titanic_tbl <- as.data.frame(Titanic)
titanic_df <- titanic_tbl[rep(seq_len(nrow(titanic_tbl)), titanic_tbl$Freq), c("Class", "Sex", "Age", "Survived")]
titanic_df$Survived <- as.integer(titanic_df$Survived == "Yes")
rownames(titanic_df) <- NULL
titanic_df <- write_data(titanic_df, "titanic")

sleep_df <- lme4::sleepstudy
names(sleep_df) <- c("Reaction", "Days", "Subject")
sleep_df$Days_c <- sleep_df$Days - mean(sleep_df$Days)  # mean-zero: keeps lkj() exact
sleep_df$Batch <- rep(c("p", "q"), each = 90)
sleep_df$Subject <- as.character(sleep_df$Subject)
sleep_df <- write_data(sleep_df, "sleepstudy")

cbpp_df <- lme4::cbpp
names(cbpp_df) <- c("Herd", "Incidence", "Size", "Period")
cbpp_df$Herd <- as.character(cbpp_df$Herd)
cbpp_df <- write_data(cbpp_df, "cbpp")

# Aggregated binomial expanded to one Bernoulli row per animal: exercises a
# logit-link ranef model at n=842 rather than cbpp's 56 aggregated rows.
cbpp_bern <- do.call(rbind, lapply(seq_len(nrow(cbpp_df)), function(i) {
  r <- cbpp_df[i, ]
  data.frame(Herd = r$Herd, Period = as.character(r$Period),
             Y = c(rep(1L, r$Incidence), rep(0L, r$Size - r$Incidence)))
}))
cbpp_bern <- write_data(cbpp_bern, "cbpp_bernoulli")

# Genuine count data (10-70 range) for the Poisson fixture, unlike mtcars$HP
# which is continuous and forces brms itself into non-convergence.
warpbreaks_df <- datasets::warpbreaks
names(warpbreaks_df) <- c("Breaks", "Wool", "Tension")
warpbreaks_df <- write_data(warpbreaks_df, "warpbreaks")

# Weighted fit reference. Integer weights so brms's `weights()` (a log-lik
# multiplier, same as ours) has an unambiguous meaning.
mtcars_w <- mtcars_df
mtcars_w$w <- rep(c(1, 3, 2, 1), length.out = nrow(mtcars_w))
mtcars_w <- write_data(mtcars_w, "mtcars_weighted")

# ---- simulated data --------------------------------------------------------
# Simulated so the harder shapes exist at all: no stock dataset gives a 3x3
# random-effect covariance, crossed groups, or a NegBin with random intercepts.

sim_negbin <- local({
  set.seed(SEED + 1)
  n_g <- 20; per <- 15
  g <- rep(sprintf("g%02d", seq_len(n_g)), each = per)
  x <- round(rnorm(n_g * per), 6)
  u <- rnorm(n_g, 0, 0.5)
  mu <- exp(0.5 + 0.4 * x + u[match(g, unique(g))])
  data.frame(G = g, X = x, Y = rnbinom(length(mu), mu = mu, size = 2))
})
sim_negbin <- write_data(sim_negbin, "sim_negbin_re")

sim_crossed <- local({
  set.seed(SEED + 2)
  n <- 240
  g1 <- sprintf("a%d", sample.int(10, n, replace = TRUE))
  g2 <- sprintf("b%d", sample.int(8, n, replace = TRUE))
  x <- round(rnorm(n), 6)
  u1 <- rnorm(10, 0, 1.5); u2 <- rnorm(8, 0, 0.8)
  y <- 1 + 0.5 * x + u1[as.integer(substring(g1, 2))] + u2[as.integer(substring(g2, 2))] + rnorm(n, 0, 1)
  data.frame(G1 = g1, G2 = g2, X = round(x, 6), Y = round(y, 6))
})
sim_crossed <- write_data(sim_crossed, "sim_crossed")

# 3x3 group-level covariance: A in coef_map is 3x3 and triangular, so this is the
# only fixture where the full Sigma -> A Sigma A' back-transform can fail visibly.
sim_three <- local({
  set.seed(SEED + 3)
  n_g <- 25; per <- 12
  g <- rep(sprintf("g%02d", seq_len(n_g)), each = per)
  n <- n_g * per
  x1 <- rnorm(n); x2 <- rnorm(n)
  x1 <- x1 - mean(x1); x2 <- x2 - mean(x2)  # mean-zero, so lkj() transfers exactly
  sds <- c(2.0, 1.0, 0.6)
  R <- matrix(c(1, 0.5, -0.3, 0.5, 1, 0.2, -0.3, 0.2, 1), 3, 3)
  U <- chol(diag(sds) %*% R %*% diag(sds))
  u <- matrix(rnorm(n_g * 3), n_g, 3) %*% U
  idx <- match(g, unique(g))
  y <- 3 + 0.8 * x1 - 0.4 * x2 + u[idx, 1] + u[idx, 2] * x1 + u[idx, 3] * x2 + rnorm(n, 0, 1)
  data.frame(G = g, X1 = round(x1, 6), X2 = round(x2, 6), Y = round(y, 6))
})
sim_three <- write_data(sim_three, "sim_three_effects")

# Wildly unequal group sizes: partial pooling shrinks the singleton groups hard,
# so any disagreement in how the ranef SD is recovered shows up here first.
sim_unbal <- local({
  set.seed(SEED + 4)
  sizes <- c(1, 1, 1, 2, 2, 3, 5, 8, 13, 21, 34, 55)
  g <- rep(sprintf("g%02d", seq_along(sizes)), times = sizes)
  n <- length(g)
  u <- rnorm(length(sizes), 0, 2)
  x <- round(rnorm(n), 6)
  y <- 2 + 0.7 * x + u[match(g, unique(g))] + rnorm(n, 0, 1)
  data.frame(G = g, X = x, Y = round(y, 6))
})
sim_unbal <- write_data(sim_unbal, "sim_unbalanced")

# 5 groups only: too few to identify the group SD from data, so the
# Exponential(1) prior on it does most of the work. Pure prior-agreement test.
sim_fewgroups <- local({
  set.seed(SEED + 5)
  sizes <- rep(8, 5)
  g <- rep(sprintf("g%d", seq_along(sizes)), times = sizes)
  n <- length(g)
  u <- rnorm(5, 0, 1.5)
  x <- round(rnorm(n), 6)
  y <- 1 + 0.5 * x + u[match(g, unique(g))] + rnorm(n, 0, 1)
  data.frame(G = g, X = x, Y = round(y, 6))
})
sim_fewgroups <- write_data(sim_fewgroups, "sim_few_groups")

# ---- prior translation -----------------------------------------------------

# Column SDs of the fixed-effect design matrix, keyed by brms coefficient name.
# `sd()` is the n-1 estimator, which is what StatsBase's ZScoreTransform uses —
# including with center=false, where it still divides by the SD about the mean.
fixef_scales <- function(formula, data) {
  fixed <- reformulas::nobars(formula)
  X <- model.matrix(fixed, data)
  X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
  apply(X, 2, sd)
}

# reformulas::findbars names a nested group "Subject:Batch" (inner:outer);
# brms names the same group "Batch:Subject" (outer:inner, declaration order).
# Reverse the colon-separated parts so both sides agree on the group name.
normalise_group_name <- function(grp) paste(rev(strsplit(grp, ":", fixed = TRUE)[[1]]), collapse = ":")

# One entry per (group, coefficient) pair, matching how TuringRegressions builds
# a Predictors block per random-effect term.
ranef_scales <- function(formula, data) {
  out <- list()
  for (bar in reformulas::findbars(formula)) {
    grp <- normalise_group_name(deparse(bar[[3]]))
    Z <- model.matrix(as.formula(paste("~", deparse(bar[[2]]))), data)
    for (cn in colnames(Z)) {
      coef_name <- if (cn == "(Intercept)") "Intercept" else cn
      # An intercept column is constant; its "scale" is 1 by construction.
      out[[length(out) + 1]] <- list(
        group = grp, coef = coef_name,
        scale = if (cn == "(Intercept)") 1 else sd(Z[, cn])
      )
    }
  }
  out
}

# The full raw-scale prior for one model. `b_sd` / `int_sd` / `re_rate` /
# `lkj_eta` are the standardised-scale hyperparameters — the defaults mirror
# `default_prior`, and overriding one here is how a prior-sensitivity fixture is
# built on both sides at once.
build_priors <- function(formula, data, family,
                         int_sd = 5, b_sd = 2, re_rate = 1, lkj_eta = 1) {
  fixed <- reformulas::nobars(formula)
  resp <- all.vars(fixed)[1]
  has_int <- attr(terms(fixed), "intercept") == 1
  scales_y <- family$family %in% c("gaussian", "student")

  y_scale <- if (scales_y) sd(data[[resp]]) else 1
  y_mean <- if (scales_y && has_int) mean(data[[resp]]) else 0

  p <- brms::empty_prior()  # brmsprior objects combine with +, not c()
  if (has_int) {
    p <- p + set_prior(sprintf("normal(%.10g, %.10g)", y_mean, int_sd * y_scale), class = "Intercept")
  }
  for (nm in names(fixef_scales(formula, data))) {
    sx <- fixef_scales(formula, data)[[nm]]
    p <- p + set_prior(sprintf("normal(0, %.10g)", b_sd * y_scale / sx), class = "b", coef = nm)
  }
  if (scales_y) {
    p <- p + set_prior(sprintf("exponential(%.10g)", 1 / y_scale), class = "sigma")
  }
  if (family$family == "student") {
    # brms's nu is bounded below at 1, and so is ours: default_prior returns
    # truncated(Gamma(2, 10); lower=1), which is this same distribution
    # (Distributions' second Gamma argument is a scale, Stan's is a rate).
    # nu is dimensionless, so no scale translation.
    p <- p + set_prior("gamma(2, 0.1)", class = "nu")
  }
  if (family$family == "negbinomial") {
    # phi ~ Exponential(1) with shape = 1/phi is exactly inv_gamma(1, 1) on shape.
    p <- p + set_prior("inv_gamma(1, 1)", class = "shape")
  }
  for (rs in ranef_scales(formula, data)) {
    p <- p + set_prior(sprintf("exponential(%.10g)", re_rate * rs$scale / y_scale),
                        class = "sd", group = rs$group, coef = rs$coef)
  }
  # brms only creates a correlation matrix (and needs a `cor` prior) for a bar
  # with more than one term — a single-coefficient bar like `(1 | G)`, or two
  # separate bars sharing a group, has no L to put a prior on.
  corr_groups <- unique(vapply(reformulas::findbars(formula), function(bar) {
    Z <- model.matrix(as.formula(paste("~", deparse(bar[[2]]))), data)
    if (ncol(Z) > 1) normalise_group_name(deparse(bar[[3]])) else NA_character_
  }, character(1)))
  corr_groups <- corr_groups[!is.na(corr_groups)]
  for (grp in corr_groups) {
    p <- p + set_prior(sprintf("lkj(%.10g)", lkj_eta), class = "cor", group = grp)
  }
  p
}

# ---- name translation ------------------------------------------------------
# brms and StatsModels name the same coefficient differently. Translating here
# means the Julia side joins on a name it already produces, instead of doing
# string surgery on every row.

factor_level_map <- function(formula, data) {
  m <- list()
  for (v in all.vars(reformulas::nobars(formula))) {
    if (!is.null(data[[v]]) && is.factor(data[[v]])) {
      for (lv in levels(data[[v]])[-1]) m[[paste0(v, lv)]] <- paste0(v, ": ", lv)
    }
  }
  m
}

julia_coef_name <- function(name, lvl_map) {
  if (name == "Intercept") return("α")
  parts <- strsplit(name, ":", fixed = TRUE)[[1]]  # R's interaction separator
  parts <- vapply(parts, function(p) if (!is.null(lvl_map[[p]])) lvl_map[[p]] else p, character(1))
  paste(parts, collapse = " & ")                    # StatsModels' separator
}

# Same, but for the effect dim of a group-level layer, where the intercept keeps
# the name `Intercept` — only the population intercept is called α.
julia_ranef_name <- function(name, lvl_map) {
  if (name == "Intercept") "Intercept" else julia_coef_name(name, lvl_map)
}

julia_group_name <- function(g) gsub(":", "__", g, fixed = TRUE)

julia_aux_name <- function(a) switch(a, sigma = "σ", nu = "ν", phi = "ϕ", a)

# ---- model registry --------------------------------------------------------
# `formula` is the plain (unweighted) formula: prior translation and name mapping
# both read it, and the weights term is spliced back in at fit time.

spec <- function(name, data, formula, family, weights_col = NULL, notes = "", ...) {
  list(name = name, data = data, formula = formula, family = family,
       weights_col = weights_col, notes = notes, prior_args = list(...))
}

MODELS <- list(
  # -- fixed effects, one per family ----------------------------------------
  spec("normal_iris", iris_df, SepalLength ~ SepalWidth + PetalLength, gaussian(),
       notes = "n=150, well identified: the tightest tolerance in the suite"),
  spec("normal_mtcars", mtcars_df, MPG ~ Cyl + Disp, gaussian(),
       notes = "n=32 with collinear predictors: prior shrinkage is visible"),
  spec("normal_mtcars_lown", mtcars_lown, MPG ~ Cyl + Disp, gaussian(),
       notes = "n=12, prior-dominated"),
  spec("normal_mtcars_lown_wide", mtcars_lown, MPG ~ Cyl + Disp, gaussian(), b_sd = 10,
       notes = "same data as normal_mtcars_lown with fixed_effects=Normal(0,10): the pair isolates the prior's effect"),
  spec("normal_mtcars_lown_tight", mtcars_lown, MPG ~ Cyl + Disp, gaussian(), b_sd = 0.25,
       notes = "same data, fixed_effects=Normal(0,0.25): heavy shrinkage toward zero"),
  spec("normal_noint", mtcars_df, MPG ~ 0 + Cyl + Disp, gaussian(),
       notes = "no intercept: X and y are scaled but never centred"),
  spec("normal_interaction", iris_df, SepalLength ~ SepalWidth * PetalLength, gaussian(),
       notes = "product column is standardised as its own predictor, not derived from its parents"),
  spec("normal_categorical", iris_df, SepalLength ~ Species + PetalLength, gaussian(),
       notes = "3-level factor: dummy columns have unequal SDs, so each gets its own translated prior"),
  spec("student_iris", iris_df, SepalLength ~ SepalWidth + PetalLength, student(),
       notes = "nu is weakly identified on clean data; expect a wide interval and check nu loosely"),
  spec("student_mtcars", mtcars_df, MPG ~ Cyl + Disp, student(),
       notes = "n=32 heavy-tailed fit: nu is where the two samplers are most likely to drift apart"),
  spec("bernoulli_titanic", titanic_df, Survived ~ Class + Sex + Age, bernoulli(),
       notes = "n=2201, all-factor predictors"),
  spec("bernoulli_mtcars", mtcars_df, Binom ~ Cyl + Disp, bernoulli(),
       notes = "n=32 binary, near-separated: the prior is what keeps it finite"),
  spec("poisson_warpbreaks", warpbreaks_df, Breaks ~ Wool + Tension, poisson()),
  spec("negbin_mtcars", mtcars_df, HP ~ Cyl + Disp, negbinomial(),
       notes = "shape prior is inv_gamma(1,1); brms shape = 1/phi, and phi is emitted as a derived draw"),

  # -- random effects, term shapes ------------------------------------------
  spec("ranef_int_sleep", sleep_df, Reaction ~ Days_c + (1 | Subject), gaussian()),
  spec("ranef_slope_sleep", sleep_df, Reaction ~ Days_c + (0 + Days_c | Subject), gaussian(),
       notes = "through-origin group slopes, no group intercept"),
  spec("ranef_corr_sleep", sleep_df, Reaction ~ Days_c + (1 + Days_c | Subject), gaussian(),
       notes = "flagship correlated term; Days_c is mean-zero so lkj(1) matches exactly"),
  spec("ranef_corr_sleep_raw", sleep_df, Reaction ~ Days + (1 + Days | Subject), gaussian(),
       notes = "DELIBERATE MISMATCH (V26): raw Days is not mean-zero, so our lkj is flat on the centred correlation, not this one"),
  spec("ranef_uncorr_sleep", sleep_df, Reaction ~ Days_c + (1 | Subject) + (0 + Days_c | Subject), gaussian(),
       notes = "two terms on one grouping variable — the case that used to drop a layer"),
  spec("ranef_nested_sleep", sleep_df, Reaction ~ Days_c + (1 | Batch / Subject), gaussian(),
       notes = "brms group names Batch and Batch:Subject map to :Batch and :Batch__Subject"),
  spec("ranef_crossed", sim_crossed, Y ~ X + (1 | G1) + (1 | G2), gaussian(),
       notes = "crossed, not nested: two independent grouping variables"),
  spec("ranef_three_effects", sim_three, Y ~ X1 + X2 + (1 + X1 + X2 | G), gaussian(),
       notes = "3x3 group covariance: the only fixture exercising a full triangular back-transform"),
  spec("ranef_unbalanced", sim_unbal, Y ~ X + (1 | G), gaussian(),
       notes = "group sizes 1..55: partial pooling is the whole point"),
  spec("ranef_few_groups", sim_fewgroups, Y ~ X + (1 | G), gaussian(),
       notes = "5 groups: the Exponential(1) prior on the group SD dominates"),

  # -- random effects x non-Normal families ----------------------------------
  spec("ranef_poisson_cbpp", cbpp_df, Incidence ~ Period + (1 | Herd), poisson()),
  spec("ranef_bernoulli_cbpp", cbpp_bern, Y ~ Period + (1 | Herd), bernoulli(),
       notes = "logit link with random intercepts, n=842"),
  spec("ranef_negbin_sim", sim_negbin, Y ~ X + (1 | G), negbinomial(),
       notes = "overdispersed counts with random intercepts — untested combination before this"),

  # -- weights ---------------------------------------------------------------
  spec("weighted_normal", mtcars_w, MPG ~ Disp, gaussian(), weights_col = "w",
       notes = "non-uniform integer weights; brms weights() multiplies the log-likelihood, as ours does")
)

# ---- fitting ---------------------------------------------------------------

brms_formula <- function(sp) {
  if (is.null(sp$weights_col)) return(sp$formula)
  lhs <- all.vars(reformulas::nobars(sp$formula))[1]
  rhs <- deparse(sp$formula[[3]], width.cutoff = 500)
  as.formula(sprintf("%s | weights(%s) ~ %s", lhs, sp$weights_col, paste(rhs, collapse = "")))
}

fit_one <- function(sp) {
  pr <- do.call(build_priors, c(list(sp$formula, sp$data, sp$family), sp$prior_args))
  brm(brms_formula(sp), data = sp$data, family = sp$family, prior = pr,
      chains = CHAINS, iter = ITER, warmup = WARMUP, seed = SEED,
      refresh = 0, silent = 2,
      file = here("fits", sp$name), file_refit = "on_change")
}

# One tidy row per parameter. `kind` is what the Julia side switches on;
# `julia_param` is the label our own output uses for the same quantity.
summarise_params <- function(fit, sp) {
  d <- as_draws_df(fit)
  vars <- variables(d)
  keep <- grepl("^(b_|sd_|cor_|r_)", vars) | vars %in% c("sigma", "nu", "shape")
  d <- subset_draws(d, variable = vars[keep])
  # brms reports the NegBin `shape`; we sample phi = 1/shape. Transform the draws
  # rather than the summary — E[1/shape] != 1/E[shape].
  if ("shape" %in% variables(d)) {
    d <- mutate_variables(d, phi = 1 / shape)
  }
  s <- summarise_draws(d, mean, sd, ~quantile2(.x, probs = c(0.025, 0.975)),
                       rhat = rhat, ess_bulk = ess_bulk)
  lvl_map <- factor_level_map(sp$formula, sp$data)

  parsed <- lapply(s$variable, function(v) {
    if (startsWith(v, "b_")) {
      nm <- sub("^b_", "", v)
      list(kind = "fixef", group = "", effect = nm, effect2 = "", level = "",
           julia_param = julia_coef_name(nm, lvl_map), julia_param2 = "")
    } else if (startsWith(v, "sd_")) {
      pieces <- strsplit(sub("^sd_", "", v), "__", fixed = TRUE)[[1]]
      list(kind = "ranef_sd", group = pieces[1], effect = pieces[2], effect2 = "", level = "",
           julia_param = julia_ranef_name(pieces[2], lvl_map), julia_param2 = "")
    } else if (startsWith(v, "cor_")) {
      pieces <- strsplit(sub("^cor_", "", v), "__", fixed = TRUE)[[1]]
      list(kind = "ranef_cor", group = pieces[1], effect = pieces[2], effect2 = pieces[3], level = "",
           julia_param = julia_ranef_name(pieces[2], lvl_map),
           julia_param2 = julia_ranef_name(pieces[3], lvl_map))
    } else if (startsWith(v, "r_")) {
      grp <- sub("\\[.*$", "", sub("^r_", "", v))
      inner <- sub("^.*\\[", "", sub("\\]$", "", v))
      pieces <- strsplit(inner, ",", fixed = TRUE)[[1]]
      list(kind = "ranef_coef", group = grp, effect = pieces[2], effect2 = "", level = pieces[1],
           julia_param = julia_ranef_name(pieces[2], lvl_map), julia_param2 = "")
    } else {
      list(kind = "aux", group = "", effect = v, effect2 = "", level = "",
           julia_param = julia_aux_name(v), julia_param2 = "")
    }
  })

  data.frame(
    model = sp$name,
    variable = s$variable,
    kind = vapply(parsed, `[[`, character(1), "kind"),
    group = julia_group_name(vapply(parsed, `[[`, character(1), "group")),
    effect = vapply(parsed, `[[`, character(1), "effect"),
    effect2 = vapply(parsed, `[[`, character(1), "effect2"),
    level = vapply(parsed, `[[`, character(1), "level"),
    julia_param = vapply(parsed, `[[`, character(1), "julia_param"),
    julia_param2 = vapply(parsed, `[[`, character(1), "julia_param2"),
    mean = s$mean, sd = s$sd, q2.5 = s$q2.5, q97.5 = s$q97.5,
    rhat = s$rhat, ess_bulk = s$ess_bulk,
    stringsAsFactors = FALSE
  )
}

# A sample of rows, not all of them: the Julia side only needs enough rows to
# catch a systematic offset, and a 2201-row model would otherwise dominate the file.
summarise_predictions <- function(fit, sp) {
  n <- nrow(sp$data)
  set.seed(sum(utf8ToInt(sp$name)))
  rows <- sort(sample.int(n, min(n, N_PRED_ROWS)))
  ep <- posterior_epred(fit)[, rows, drop = FALSE]
  lp <- posterior_linpred(fit)[, rows, drop = FALSE]
  pp <- posterior_predict(fit)[, rows, drop = FALSE]
  data.frame(
    model = sp$name, row = rows,
    epred_mean = colMeans(ep), epred_sd = apply(ep, 2, sd),
    epred_q2.5 = apply(ep, 2, quantile, 0.025), epred_q97.5 = apply(ep, 2, quantile, 0.975),
    linpred_mean = colMeans(lp),
    pp_mean = colMeans(pp), pp_sd = apply(pp, 2, sd),
    stringsAsFactors = FALSE
  )
}

# ---- run -------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
# data/ is gitignored, so the Julia tests need a way to rebuild it without paying
# for ~30 Stan compiles. Everything above this point has already written it.
if ("--data-only" %in% args) {
  message("wrote data/ only; no fits")
  quit(save = "no", status = 0)
}
selected <- if (length(args) == 0) MODELS else Filter(function(s) s$name %in% args, MODELS)
if (length(selected) == 0) stop("no model matched: ", paste(args, collapse = ", "))

total <- length(selected)
t_start <- Sys.time()

for (i in seq_along(selected)) {
  sp <- selected[[i]]
  message(sprintf("[%d/%d] %s  (%s, n=%d) ...", i, total, sp$name, sp$family$family, nrow(sp$data)))
  t0 <- Sys.time()
  fit <- fit_one(sp)
  message(sprintf("[%d/%d] %s  done in %.0fs  (elapsed %.0fs)", i, total, sp$name,
                  as.numeric(difftime(Sys.time(), t0, units = "secs")),
                  as.numeric(difftime(Sys.time(), t_start, units = "secs"))))

  div <- rstan::get_num_divergent(fit$fit)
  if (div > 0) warning(sprintf("[%s] %d divergent transitions", sp$name, div))

  # One directory per model rather than four accumulating files: a partial re-run
  # then rewrites only what it refitted, with no merge logic and a readable diff.
  out <- here("reference", sp$name)
  dir.create(out, showWarnings = FALSE, recursive = TRUE)

  write.csv(summarise_params(fit, sp), file.path(out, "params.csv"), row.names = FALSE)
  write.csv(summarise_predictions(fit, sp), file.path(out, "predictions.csv"), row.names = FALSE)

  ps <- prior_summary(fit)
  write.csv(data.frame(
    model = sp$name, prior = ps$prior, class = ps$class, coef = ps$coef, group = ps$group,
    stringsAsFactors = FALSE
  ), file.path(out, "priors.csv"), row.names = FALSE)

  write.csv(data.frame(
    model = sp$name,
    dataset = attr(sp$data, "dataname"),
    formula = paste(deparse(sp$formula, width.cutoff = 500), collapse = ""),
    family = sp$family$family,
    weights_col = if (is.null(sp$weights_col)) "" else sp$weights_col,
    n = nrow(sp$data),
    chains = CHAINS, draws_per_chain = ITER - WARMUP, seed = SEED,
    # Stan's own recorded sampling time, not wall clock: it survives the fit cache,
    # so the Julia report can compare against it on runs that refitted nothing.
    # Sum over chains (rstan samples them sequentially here), warmup included.
    stan_seconds = round(sum(rstan::get_elapsed_time(fit$fit)), 1),
    brms_version = as.character(utils::packageVersion("brms")),
    notes = sp$notes,
    divergences = div,
    stringsAsFactors = FALSE
  ), file.path(out, "model.csv"), row.names = FALSE)

  message(sprintf("[%d/%d] %s  wrote reference/%s/", i, total, sp$name, sp$name))
}

message(sprintf("done: %d model(s) in %.0fs", total,
                as.numeric(difftime(Sys.time(), t_start, units = "secs"))))
