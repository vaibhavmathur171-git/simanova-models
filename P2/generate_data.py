import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

SELLMEIER = {
    "N-BK7": {
        "B": [1.03961212, 0.231792344, 1.01046945],
        "C": [0.00600069867, 0.0200179144, 103.560653],
    },
    "N-SF11": {
        "B": [1.73759695, 0.313747346, 1.89878101],
        "C": [0.013188707, 0.0623068142, 155.23629],
    },
}

WAVELENGTHS = {"R": 635.0, "G": 532.0, "B": 450.0}
WEIGHTS = {"R": 0.2, "G": 0.6, "B": 0.2}
MATERIALS = ["N-BK7", "N-SF11"]


def refractive_index(wavelength_nm: float, material: str) -> float:
    lambda_um = wavelength_nm / 1000.0
    lambda_sq = lambda_um ** 2
    B = SELLMEIER[material]["B"]
    C = SELLMEIER[material]["C"]
    n_sq = 1.0 + sum(B[i] * lambda_sq / (lambda_sq - C[i]) for i in range(3))
    return np.sqrt(n_sq)


def diffraction_angle(pitch_nm: float, wavelength_nm: float) -> float:
    sin_out = wavelength_nm / pitch_nm
    sin_out = np.clip(sin_out, -1.0, 1.0)
    return np.degrees(np.arcsin(sin_out))


def find_optimal_pitch(target_angle: float, material: str) -> float:
    def loss(pitch_nm: float) -> float:
        error = 0.0
        for color, wavelength in WAVELENGTHS.items():
            angle = diffraction_angle(pitch_nm, wavelength)
            deviation = angle - target_angle
            error += WEIGHTS[color] * deviation ** 2
        return error

    result = minimize_scalar(loss, bounds=(200.0, 2000.0), method='bounded')
    return result.x


def generate_dataset(n_samples: int = 50000) -> pd.DataFrame:
    print(f"Generating {n_samples} ground truth samples...")

    data = []
    for i in range(n_samples):
        angle = np.random.uniform(-40.0, 40.0)
        material_id = np.random.randint(0, 2)
        material = MATERIALS[material_id]

        ideal_pitch_nm = find_optimal_pitch(angle, material)

        data.append({
            "angle": angle,
            "material_id": material_id,
            "ideal_pitch_nm": ideal_pitch_nm,
        })

        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{n_samples} complete")

    df = pd.DataFrame(data)
    df.to_csv("P2/spectral_data.csv", index=False)
    print(f"\nSaved to P2/spectral_data.csv")
    print(f"Shape: {df.shape}")
    print(df.head())
    return df


if __name__ == "__main__":
    generate_dataset(50000)
