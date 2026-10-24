# region maths
from dataclasses import dataclass


def casimir_energy(F, I, J, A, B):
    """Calculates hyperfine energy shift in MHz using Casimir's formula."""
    K = F * (F + 1) - I * (I + 1) - J * (J + 1)
    denom = 4 * I * (2 * I - 1) * J * (2 * J - 1)
    quad = B * (1.5 * K * (K + 1) - 2 * I * (I + 1) * J * (J + 1)) / denom
    return 0.5 * A * K + quad


# endregion
# region predefined quantities

@dataclass
class ManifoldData:
    name: str
    I: float
    J: float
    Fp_levels: list[float]
    ref_index: int
    ABhfs_guess: tuple[float, float]

    @property
    def names(self):
        return [f"{self.name}->{Fp}" for Fp in self.Fp_levels]


MANIFOLDS_CONFIG = [
    ManifoldData('87Rb F=2', 1.5, 1.5, [1, 2, 3], 2,  # Reference peak is F'=3 (cycling transition)
                 (84.7, 12.5), ),
    ManifoldData('85Rb F=3', 2.5, 1.5, [2, 3, 4], 2,  # Reference peak is F'=4 (cycling transition)
                 (25.0, 25.8)),
    ManifoldData('85Rb F=2', 2.5, 1.5, [1, 2, 3], 0,  # Reference peak is F'=1
                 (25.0, 25.8)),
    ManifoldData('87Rb F=1', 1.5, 1.5, [0, 1, 2], 0,  # Reference peak is F'=0
                 (84.7, 12.5))
]
# endregion
