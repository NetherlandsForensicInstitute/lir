from lir import Transformer


class MyModel(Transformer):
    """A cool ML model."""

    def fit(self, instances):
        """Fit your ML-model."""
        return self 
    

    def apply(self, instances):
        """Apply your ML-model."""
        return instances