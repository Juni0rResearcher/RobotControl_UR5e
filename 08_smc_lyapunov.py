# sliding_mode_lyapunov_tuned.py

import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import Simulator
from pathlib import Path
import os

KP = 100.0
KD = 20.0

print("\n" + "="*70)
print("ЛЯПУНОВСКИЙ АНАЛИЗ ДЛЯ ПОДБОРА ПАРАМЕТРОВ SMC")
print("="*70)

# Оценка верхней границы возмущений Δ_max от неопределённостей
# Лишняя масса: 4 кг на конце эффектора
m_payload = 4.0
g = 9.81
l_arm = 0.9  

# Максимальный гравитационный момент
tau_grav_max = m_payload * g * l_arm  

# Кулоновское трение из задания
tau_friction_max = np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1])

# Вязкое трение 
damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
v_typical = 2.0
tau_viscous_max = damping * v_typical

# Суммарная верхняя граница возмущений на сустав
delta_max = tau_grav_max * np.array([0.1, 0.8, 0.6, 0.1, 0.1, 0.05]) + tau_friction_max + tau_viscous_max

print("Оценка верхней границы возмущений |Δ_max| на сустав (Nm):")
for i in range(6):
    print(f"Joint {i+1}: {delta_max[i]:.2f} Nm")

eta = 10.0  # запас для конечного времени сходимости
K_robust_min = delta_max + eta

print(f"\nМинимальный K_robust_vec (с запасом η = {eta} Nm):")
for i in range(6):
    print(f"Joint {i+1}: {K_robust_min[i]:.2f} Nm")

K_robust_vec = 2.0 * K_robust_min
print(f"\nРекомендуемый K_robust_vec (с удвоенным запасом):")
for i in range(6):
    print(f"Joint {i+1}: {K_robust_vec[i]:.2f} Nm")

Lambda_opt = np.sqrt(KP)
print(f"\nОптимальное Lambda по Ляпунову (sqrt(KP) = sqrt({KP})): {Lambda_opt:.2f}")

# Финальные параметры
Lambda = round(Lambda_opt, 1)
K_robust_vec = np.round(K_robust_vec, 1)

print("\n" + "="*70)
print("Подобранные параметры по анализу Ляпунова:")
print(f"Lambda = {Lambda}")
print(f"K_robust_vec = {K_robust_vec}")
print("="*70 + "\n")

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
pin_model = pin.buildModelFromMJCF(xml_path)
pin_data = pin_model.createData()
print(f"Pinocchio модель успешно загружена: {pin_model.nq} DOF")

simulation_data = {
    'time': [],
    'positions': [],
    'velocities': [],
    'torques': [],
    'desired_positions': [],
    'desired_velocities': [],
    'sliding_surface': []
}

def sliding_mode_trajectory_controller(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
    """
    Sliding Mode Controller для отслеживания траектории.
    Параметры подобраны по Ляпунову.
    """
    q_offset = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    amplitude = np.array([0.15, 0.20, 0.15, 0.10, 0.10, 0.08])
    frequency = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
    phase = np.array([0.0, np.pi/4, np.pi/2, np.pi, 3*np.pi/4, np.pi/3])

    warm_up_time = 0.5

    if t < warm_up_time:
        # Держим текущую позу — плавный старт
        q_des = q.copy()
        dq_des = np.zeros(6)
        ddq_des = np.zeros(6)
    else:
        # Траектория начинается с t = 0 в момент t = warm_up_time
        t_eff = t - warm_up_time

        omega = 2 * np.pi * frequency
        q_sin = amplitude * np.sin(omega * t_eff + phase)
        q_des = q_offset + q_sin
        dq_des = amplitude * omega * np.cos(omega * t_eff + phase)
        ddq_des = -amplitude * omega**2 * np.sin(omega * t_eff + phase)

    # Ошибки
    e = q_des - q
    de = dq_des - dq

    # Поверхность скольжения
    s = de + Lambda * e

    # Номинальная часть (эквивалентный контроль)
    pin.computeAllTerms(pin_model, pin_data, q, dq)
    M_hat = pin_data.M
    nle_hat = pin_data.nle

    # Эквивалентное управление: компенсирует номинальную динамику
    tau_nominal = M_hat @ (ddq_des + Lambda * de) + nle_hat
    tau_robust = K_robust_vec * np.sign(s)
    tau = tau_nominal + tau_robust

    # Ограничение по моменту (важно для реализма)
    tau = np.clip(tau, -150, 150)

    # Логирование
    simulation_data['time'].append(t)
    simulation_data['positions'].append(q.copy())
    simulation_data['velocities'].append(dq.copy())
    simulation_data['torques'].append(tau.copy())
    simulation_data['desired_positions'].append(q_des.copy())
    simulation_data['sliding_surface'].append(s.copy())
    simulation_data['desired_velocities'].append(dq_des.copy())  # добавь в словарь!

    return tau

def plot_results():
    """Построение и сохранение графиков результатов."""
    Path("logs/plots/SMC_lyapunov").mkdir(parents=True, exist_ok=True)

    times = np.array(simulation_data['time'])
    positions = np.array(simulation_data['positions'])
    velocities = np.array(simulation_data['velocities'])
    torques = np.array(simulation_data['torques'])
    q_des_all = np.array(simulation_data['desired_positions'])
    dq_des_all = np.array(simulation_data['desired_velocities'])

    pos_errors = q_des_all - positions
    vel_errors = dq_des_all - velocities

    # Ошибки положения
    plt.figure(figsize=(14, 9))
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(times, pos_errors[:, i], 'g-', linewidth=2)
        plt.axhline(0, color='r', linestyle='--', alpha=0.6)
        plt.xlabel('Time [s]')
        plt.ylabel('Pos Error [rad]')
        plt.title(f'Joint {i+1} Position Error')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.suptitle('SMC with Lyapunov: tracking errors', fontsize=16, y=0.98)
    plt.savefig('logs/plots/SMC_lyapunov/01_position_errors.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Ошибки скорости
    plt.figure(figsize=(14, 9))
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(times, vel_errors[:, i], 'b-', linewidth=2)
        plt.axhline(0, color='r', linestyle='--', alpha=0.6)
        plt.xlabel('Time [s]')
        plt.ylabel('Vel Error [rad/s]')
        plt.title(f'Joint {i+1} Velocity Error')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.suptitle('SMC with Lyapunov: Velocity Tracking Errors', fontsize=16, y=0.98)
    plt.savefig('logs/plots/SMC_lyapunov/02_velocity_errors.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Сравнение желаемой и реальной траектории
    plt.figure(figsize=(14, 10))
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(times, positions[:, i], 'b-', linewidth=2, label='Actual')
        plt.plot(times, q_des_all[:, i], 'r--', linewidth=2, label='Desired')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [rad]')
        plt.title(f'Joint {i+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.suptitle('Desired vs Actual Joint Positions', fontsize=16, y=0.98)
    plt.savefig('logs/plots/SMC_lyapunov/03_tracking.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Управляющие моменты
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.plot(times, torques[:, i], linewidth=2, label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.title('Control Torques')
    plt.legend(ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('logs/plots/SMC_lyapunov/04_torques.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Статистика
    
    rmse_pos = np.sqrt(np.mean(pos_errors**2, axis=0))
    rmse_vel = np.sqrt(np.mean(vel_errors**2, axis=0))
    max_pos_err = np.max(np.abs(pos_errors), axis=0)
    max_torque = np.max(np.abs(torques), axis=0)

    print("\nRMSE Position Errors [rad]:")
    for i in range(6):
        print(f"  Joint {i+1}: {rmse_pos[i]:.6f}")

    print("\nRMSE Velocity Errors [rad/s]:")
    for i in range(6):
        print(f"  Joint {i+1}: {rmse_vel[i]:.6f}")

    print("\nMax Absolute Position Errors [rad]:")
    for i in range(6):
        print(f"  Joint {i+1}: {max_pos_err[i]:.6f}")

    print("\nMax Control Torques [Nm]:")
    for i in range(6):
        print(f"  Joint {i+1}: {max_torque[i]:.2f}")


def main():
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    print("\nЗапуск SMC с параметрами из Ляпуновского анализа...")

    for key in simulation_data:
        simulation_data[key].clear()

    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        record_video=True,
        video_path="logs/videos/08_smc_lyapunov_tuned.mp4",
        width=1920,
        height=1080
    )

    sim.modify_body_properties("end_effector", mass=4.0)
    sim.set_joint_damping(np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]))
    sim.set_joint_friction(np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1]))

    sim.set_controller(sliding_mode_trajectory_controller)
    sim.run(time_limit=10.0)

    print("Симуляция завершена.")
    plot_results()

if __name__ == "__main__":
    main()