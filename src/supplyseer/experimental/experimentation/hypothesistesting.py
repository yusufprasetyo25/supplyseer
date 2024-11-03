import numpy as np
from typing import Callable, List
from pydantic import BaseModel, Field, conlist

class EValue(BaseModel):
    """
    A class representing an E-value for hypothesis testing.
    """
    data: conlist(float, min_items=1) = Field(..., description="The observed data.")
    null_hypothesis: Callable[[np.ndarray], float] = Field(..., description="A function representing the null hypothesis, returning a value under H0.")
    e_value: float = Field(None, description="The computed E-value.", exclude=True)

    def compute_e_value(self, alternative: Callable[[np.ndarray], float]) -> float:
        """
        Compute the E-value for the given alternative hypothesis.

        :param alternative: A function representing the alternative hypothesis, returning a value under H1.
        :return: The computed E-value.
        """
        null_val = self.null_hypothesis(np.array(self.data))
        alt_val = alternative(np.array(self.data))
        if null_val == 0:
            raise ValueError("Null hypothesis value should not be zero.")
        self.e_value = alt_val / null_val
        return self.e_value

    def __str__(self):
        return f"E-Value: {self.e_value}"


class HypothesisTest(BaseModel):
    """
    A class to perform hypothesis testing using E-values.
    """
    data: conlist(float, min_items=1) = Field(..., description="The observed data for testing.")
    e_values: List[EValue] = Field(default_factory=list, description="List of E-value objects.")

    def add_e_value(self, e_value: EValue):
        """
        Add an E-value object to the test.

        :param e_value: An instance of EValue.
        """
        self.e_values.append(e_value)

    def evaluate(self, threshold: float = 1.0) -> bool:
        """
        Evaluate the hypothesis based on E-values and a significance threshold.

        :param threshold: The threshold for rejection (typically 1.0).
        :return: True if we reject the null hypothesis, False otherwise.
        """
        for e_value in self.e_values:
            if e_value.e_value > threshold:
                return True
        return False

    def summary(self):
        """
        Print a summary of the hypothesis test.
        """
        for e_value in self.e_values:
            print(e_value)
