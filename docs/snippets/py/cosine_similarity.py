from pathlib import Path

import numpy as np
from numpy.linalg import norm

from lir import FeatureData, InstanceData, PairedFeatureData, Transformer
from lir.config import ConfigValue, check_is_empty, config_parser, pop_field
from lir.util import check_type


class CosineSimilarity(Transformer):
    def __init__(self, square: bool = False):
        self.square = square

    def apply(self, pairs: InstanceData) -> FeatureData:
        # make sure that we have pairs of feature vectors
        pairs = check_type(PairedFeatureData, pairs)

        # get the feature vectors for the trace and reference data
        features_trace = pairs.features_trace[:, 0, :]
        features_ref = pairs.features_ref[:, 0, :]

        # calculate the similarities between the trace and the reference feature vectors
        dot_products = np.sum(features_trace * features_ref, axis=1)
        norms = norm(features_trace, axis=1) * norm(features_ref, axis=1)
        cosine_similarities = dot_products / norms

        if self.square:
            cosine_similarities = np.square(cosine_similarities)

        # return a FeatureData object with the same attributes as the input pairs, but with the calculated similarities
        # as the features
        return pairs.replace_as(FeatureData, features=cosine_similarities)


@config_parser
def parse_cosine_similarity_config(config: ConfigValue, output_dir: Path) -> CosineSimilarity:
    print('CONTEXT:', config.context)
    print('CONFIG:', config.unwrap())

    # obtain the value of the "square" parameter from the configuration dictionary, and remove it from the dictionary.
    square = pop_field(config, 'square', default=False)

    # check that there are no other parameters left unparsed, and raise an error if there are.
    check_is_empty(config)

    # instantiate the component with the parsed parameters and return it.
    return CosineSimilarity(square=square)
