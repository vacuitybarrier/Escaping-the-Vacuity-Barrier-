
# Escaping the Vacuity Barrier? A Pre-Registered Gate and an Exploratory Investigation of Control-Sufficient Latent World Models

[![arXiv CS.LG](https://img.shields.io/badge/arXiv-cs.LG-B31B31.svg)](https://arxiv.org/)
[![OpenReview DMLR](https://img.shields.io/badge/OpenReview-DMLR-orange.svg)](https://openreview.net/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for **"Escaping the Vacuity Barrier? A Pre-Registered Gate and an Exploratory Investigation of Control-Sufficient Latent World Models"**. This project investigates joint-embedding predictive architectures (LeJEPA), probabilistic layer normalization, ARCP-Mondrian conformal uncertainty guarantees, and exploitability confounds in closed-loop control tasks (Push-T).

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/vacuitybarrier/Escaping-the-Vacuity-Barrier-.git](https://github.com/vacuitybarrier/Escaping-the-Vacuity-Barrier-.git)
   cd escaping-vacuity-barrier


Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Experimental Phases

Phase 0 (src/phase0_pilot.py): Pilot corpus calibration, feasibility checks, and dry-runs.

Phase 1 (src/phase1_planner.py): Exact dynamics evaluation & Step 0 planner repair gate.

Phase 2 (src/phase2_encoder.py): DINOv2 feature extraction, projector training ($P_\phi$), and hashing.

Phase 3 (src/phase3_decoder.py): Decoder transfer evaluation and P1b gate diagnostics.

Phase 4 (src/phase4_calibration.py): ARCP-Mondrian calibration and conformal prediction quantiles.

Phase 5 (src/phase5_closed_loop.py): Closed-loop evaluation, gate abstention verification, and exploitability diagnostics.

## 📜 License
Distributed under the MIT License. See LICENSE for details.

---

### 2. `requirements.txt`
```text
gym-pusht==0.1.8
pymunk==6.11.1
gymnasium>=0.29.1
torch>=2.1.0
timm>=0.9.16
numpy>=1.21.0
scipy>=1.10.0
scikit-learn>=1.3.0
opencv-python>=4.9.0
pygame>=2.5.2
shapely>=2.0.3
psutil>=5.9.0
```
