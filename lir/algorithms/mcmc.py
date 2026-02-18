from typing import Any, Self

import numpy as np
from scipy.stats import betabinom, binom, norm

from lir.algorithms.bayeserror import ELUBBounder
from lir.bounding import LLRBounder, check_type
from lir.data.models import FeatureData, InstanceData, LLRData
from lir.transform import Transformer


elub_bounder = ELUBBounder()


class McmcLLRModel(Transformer):
    """
    Use Markov Chain Monte Carlo simulations to fit a statistical distribution for each of the two hypotheses.

    Using samples from the posterior distributions of the model parameters, a posterior distribution of the LR is
    obtained. The median of this distribution is used as best estimate for the LR; a credible interval is also
    determined.
    """

    def __init__(
        self,
        distribution_h1: str,
        parameters_h1: dict[str, dict[str, int]] | None,
        distribution_h2: str,
        parameters_h2: dict[str, dict[str, int]] | None,
        bounding: LLRBounder | None = elub_bounder,
        interval: tuple[float, float] = (0.05, 0.95),
        **mcmc_kwargs: Any,
    ):
        """
        Initialise the MCMC model, based on distributions and parameters.

        :param distribution_h1: statistical distribution used to model H1, for example 'normal' or 'binomial'
        :param parameters_h1: definition of the parameters of distribution_h1, and their prior distributions
        :param distribution_h2: statistical distribution used to model H2, for example 'normal' or 'binomial'
        :param parameters_h2: definition of the parameters of distribution_h2, and their prior distributions
        :param bounder: bounding method to apply to the unbound llrs, to prevent overextrapolation
        :param interval: lower and upper bounds of the credible interval in range 0..1; default: (0.05, 0.95)
        :param mcmc_kwargs: mcmc simulation settings, see `McmcModel.__init__` for more details.
        """
        self.model_h1 = McmcModel(distribution_h1, parameters_h1, **mcmc_kwargs)
        self.model_h2 = McmcModel(distribution_h2, parameters_h2, **mcmc_kwargs)
        self.bounding = bounding
        self.bounders: list[LLRBounder] | None = None
        self.interval = interval

    def fit(self, instances: InstanceData) -> Self:
        """Fit the defined model to the supplied instances."""
        instances = check_type(FeatureData, instances)

        self.model_h1.fit(instances.features[instances.require_labels == 1])
        self.model_h2.fit(instances.features[instances.require_labels == 0])
        if self.bounding is not None:
            # determine the bounds based on the LLRs of the training data, each sample results into an LR-system
            logp_h1 = self.model_h1.transform(instances.features)
            logp_h2 = self.model_h2.transform(instances.features)
            llrs = logp_h1 - logp_h2

            # determine the bounds for each LR-system individually
            self.bounders = [self.bounding.__class__() for _ in range(llrs.shape[1])]
            for i_system in range(llrs.shape[1]):
                feature_data = FeatureData(features=instances.features, labels=instances.require_labels)
                self.bounders[i_system] = self.bounders[i_system].fit(feature_data)
        return self

    def transform(self, instances: FeatureData) -> LLRData:
        """Apply the fitted model to the supplied instances."""
        logp_h1 = self.model_h1.transform(instances.features)
        logp_h2 = self.model_h2.transform(instances.features)
        llrs = logp_h1 - logp_h2
        if (self.bounding is not None) and (self.bounders is not None):
            # apply the bounders one by one
            for i_system in range(llrs.shape[1]):
                feature_data = FeatureData(features=instances.features, labels=instances.require_labels)
                llrs[:, i_system] = self.bounders[i_system].apply(feature_data)
        quantiles = np.quantile(llrs, [0.5] + list(self.interval), axis=1, method='midpoint')
        return instances.replace_as(LLRData, features=quantiles.transpose(1, 0))


class McmcModel:
    """Use Markov Chain Monte Carlo simulations to fit a statistical distribution."""

    def __init__(
        self,
        distribution: str,
        parameters: dict[str, dict[str, int]] | None,
        chain_count: int = 4,
        tune_count: int = 1000,
        draw_count: int = 1000,
        random_seed: int | None = None,
    ):
        """
        Define the MCMC model and settings to be used.

        :param distribution: statistical distribution used, for example 'normal' or 'binomial'
        :param parameters: definition of the parameters of the distribution, and their prior distributions; see below.
        :param chain_count: number of parallel mcmc chains
        :param tune_count: number of tune/warm-up/burn-in samples per chain
        :param draw_count: number of samples to draw from each chain
        :param random_seed: random seed

        Currently supported distributions are: 'betabinomial', 'binomial', 'normal'.
        Names of the parameters are based on the nomenclature used in pymc for distribution parameters:
        https://www.pymc.io/projects/docs/en/stable/api/distributions.html. The parameters should be provided as a
        dictionary where the keys are the names of the parameter used for the selected statistical distribution, and the
        values are dictionaries with a key 'prior', defining the prior distribution used for that parameter (currently
        supported values are 'beta', 'normal' and 'uniform'), and with additional keys corresponding to the names of the
        parameters used for that prior distribution (the dict values are the values of these parameters).
        For example, for a binomial distribution: parameters = {'p': {'prior': 'beta', 'alpha': 0.5, 'beta': 0.5}}.
        Or for a betabinomial distribution: parameters = {'alpha': {'prior': 'uniform', 'lower': 0.01, 'upper': 100},
        'beta': {'prior': 'uniform', 'lower': 0.01, 'upper': 100}}.
        """
        self.distribution = distribution
        self.parameters = parameters
        self.chain_count = chain_count
        self.tune_count = tune_count
        self.draw_count = draw_count
        self.random_seed = random_seed
        self.parameter_samples: dict[str, np.ndarray] = {}

    def fit(self, features: np.ndarray) -> Self:
        """
        Draw samples from the posterior distributions of the parameters of a specified statistical distribution.

        The posteriors are based on the specified prior distributions of these parameters and observed feature values.

        :param features: observed feature values, used to update the prior distributions of the parameters with
        """
        try:
            import pymc as pm
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('pymc is required to use McmcModel.fit(). Install lir with pymc support.') from e

        if self.parameters is None:
            raise ValueError('Distribution parameters not specified.')
        # It looks like all pymc stuff needs to be in a single model block
        with pm.Model():
            # Define the prior distributions of the model parameters based on their definitions
            priors = {}
            for parameter, parameter_input in self.parameters.items():
                if parameter_input['prior'] == 'beta':
                    prior = pm.Beta(parameter, alpha=parameter_input['alpha'], beta=parameter_input['beta'])
                elif parameter_input['prior'] == 'normal':
                    prior = pm.Normal(parameter, mu=parameter_input['mu'], sigma=parameter_input['sigma'])
                elif parameter_input['prior'] == 'uniform':
                    prior = pm.Uniform(parameter, lower=parameter_input['lower'], upper=parameter_input['upper'])
                else:
                    raise ValueError('Unrecognized prior')
                priors.update({parameter: prior})
            # Define the model: priors and the observed data
            if self.distribution == 'betabinomial':
                pm.BetaBinomial(
                    'k', alpha=priors['alpha'], beta=priors['beta'], n=features[:, 1], observed=features[:, 0]
                )
            elif self.distribution == 'binomial':
                pm.Binomial('k', p=priors['p'], n=np.sum(features[:, 1]), observed=np.sum(features[:, 0]))
            elif self.distribution == 'normal':
                pm.Normal('x', mu=priors['mu'], sigma=priors['sigma'], observed=features[:, 0])
            else:
                raise ValueError('Unrecognized distribution')
            # Do simulations and sample from the posterior distributions
            trace = pm.sample(
                draws=self.draw_count,
                chains=self.chain_count,
                tune=self.tune_count,
                cores=1,
                random_seed=self.random_seed,
                progressbar=False,
            )
        # Get the posterior samples of the model parameters and convergence statistics
        self.parameter_samples = {}
        for parameter in list(self.parameters.keys()):
            # Combine the samples from all chains
            samples = np.concatenate(np.array(trace.posterior[parameter]))  # type: ignore [unresolved-attribute]
            self.parameter_samples.update({parameter: samples})
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Get samples of the posterior distribution of the (log10) probability.

        Use the samples of the posterior distributions of the parameters, in combination with the selected statistical
        distribution, to get samples of the posterior distribution of the (log10) probability, evaluated for specified
        feature values.

        :param features: feature values for which the probabilities are to be calculated
        """
        # Prepare features and parameters for 2d-evaluations (number of samples * number of requested feature values)
        sample_count = len(self.parameter_samples[list(self.parameter_samples.keys())[0]])
        features_2d = {}
        for feature_id in range(features.shape[1]):
            feature_2d = np.tile(np.expand_dims(features[:, feature_id], 1), (1, sample_count))
            features_2d.update({feature_id: feature_2d})
        parameters_2d = {}
        for parameter in list(self.parameter_samples.keys()):
            parameter_2d = np.tile(np.expand_dims(self.parameter_samples[parameter], 0), (len(features), 1))
            parameters_2d.update({parameter: parameter_2d})
        # Calculate e-base log probabilities at specified feature values
        if self.distribution == 'betabinomial':
            logp = betabinom.logpmf(features_2d[0], features_2d[1], parameters_2d['alpha'], parameters_2d['beta'])
        elif self.distribution == 'binomial':
            logp = binom.logpmf(features_2d[0], features_2d[1], parameters_2d['p'])
        elif self.distribution == 'norm':
            logp = norm.logpdf(features_2d[0], parameters_2d['mu'], parameters_2d['sigma'])
        else:
            raise ValueError('Unrecognized distribution')
        # Return 10-base log probabilities
        return logp / np.log(10)
