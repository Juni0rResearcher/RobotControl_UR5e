# Robot Control Exam

## Notes

- Report in Iuzhanin_Andrei_M25RO1.pdf
- Videos are saved in `logs/videos/`
  - 06_inverse_dynamics.mp4 (task 1)
  - 07_inverse_dynamics_uncertain.mp4 (task 1)
  - 08_smc.mp4 (task 2)
  - 08_smc_lyapunov_tuned.mp4 (task 2)
  - 09_smc_boundary_phi_0.05...0.50.mp4 (task 3)
- Plots are saved in `logs/plots/`
- The simulator supports both real-time visualization and headless operation
- All examples use the UR5e robot model

The structure of the repository is as follows:
```bash
├── logs/
│ ├── videos/ # Simulation recordings
│ └── plots/ # Generated plots
├── robots/ # Robot models
└── simulator/ # Simulator class
```
# Assignment Tasks

- [40 points] Inverse Dynamics Controller
   Implement Inverse Dynamics controller
- [40 points] Sliding Mode Controller
   Modify the UR5 robot model to include:
   Additional end-effector mass
   Joint damping coefficients
   Coulomb friction
   Implement sliding mode controller
   Compare ID and Sliding Mode
- [20 points] Boundary Layer Implementation
   Analyze the chattering phenomenon:
   Causes and practical implications
   Boundary layer modification for smoothing
   Evaluate performance with varying boundary layer thicknesses 
   Analyze the robustness-chattering trade-off
