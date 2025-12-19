import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import Simulator
from pathlib import Path
import os

# Параметры контроллера
K_robust_vec = np.array([20.0, 50.0, 30.0, 10.0, 10.0, 8.0])
Lambda = 12.0

# Список толщин пограничного слоя для анализа
phis = [0.05, 0.1, 0.2, 0.3, 0.5]
results = {}

# Загрузка модели Pinocchio
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
pin_model = pin.buildModelFromMJCF(xml_path)
pin_data = pin_model.createData()
print(f"Pinocchio модель успешно загружена: {pin_model.nq} DOF")

# Глобальный словарь для логирования (будет очищаться перед каждым запуском)
simulation_data = {
    'time': [],
    'positions': [],
    'velocities': [],
    'torques': [],
    'desired_positions': [],
    'desired_velocities': [],
    'sliding_surface': []
}

# Траектория (одинаковая для всех запусков)
def generate_trajectory(q: np.ndarray, t):
    q_offset = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    amplitude = np.array([0.15, 0.20, 0.15, 0.10, 0.10, 0.08])
    frequency = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
    phase = np.array([0.0, np.pi/4, np.pi/2, np.pi, 3*np.pi/4, np.pi/3])
    omega = 2 * np.pi * frequency

    warm_up_time = 0.5  # секунды плавного старта

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
    return q_des, dq_des, ddq_des

def create_controller(phi):
    """Создаёт контроллер с фиксированным phi"""
    def controller(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
        q_des, dq_des, ddq_des = generate_trajectory(q, t)

        e = q_des - q
        de = dq_des - dq
        s = de + Lambda * e

        pin.computeAllTerms(pin_model, pin_data, q, dq)
        M_hat = pin_data.M
        nle_hat = pin_data.nle

        tau_nominal = M_hat @ (ddq_des + Lambda * de) + nle_hat
        tau_robust = K_robust_vec * np.clip(s / phi, -1.0, 1.0)
        tau = tau_nominal + tau_robust
        tau = np.clip(tau, -150, 150)

        # Логирование
        simulation_data['time'].append(t)
        simulation_data['positions'].append(q.copy())
        simulation_data['velocities'].append(dq.copy())
        simulation_data['torques'].append(tau.copy())
        simulation_data['desired_positions'].append(q_des.copy())
        simulation_data['desired_velocities'].append(dq_des.copy())
        simulation_data['sliding_surface'].append(s.copy())

        return tau
    return controller

def plot_results(phi):
    """Строит графики для конкретного phi"""
    folder = Path(f"logs/plots/SMC_boundary_layer/phi_{phi:.2f}")
    folder.mkdir(parents=True, exist_ok=True)

    times = np.array(simulation_data['time'])
    positions = np.array(simulation_data['positions'])
    torques = np.array(simulation_data['torques'])
    velocities = np.array(simulation_data['velocities'])
    desired = np.array(simulation_data['desired_positions'])
    q_des_all = np.array(simulation_data['desired_positions'])
    dq_des_all = np.array(simulation_data['desired_velocities'])
    s_values = np.array(simulation_data['sliding_surface'])
    pos_errors = desired - positions
    vel_errors = dq_des_all - velocities

    # Ошибки по суставам
    plt.figure(figsize=(14, 8))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.plot(times, pos_errors[:, i], 'g-', linewidth=2)
        plt.axhline(0, color='r', linestyle='--', alpha=0.6)
        plt.title(f'Joint {i+1} Error')
        plt.xlabel('Time [s]')
        plt.ylabel('Error [rad]')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.suptitle(f'SMC with Boundary Layer phi = {phi:.2f} — Position Errors', fontsize=16, y=1.02)
    plt.savefig(folder / "position_errors.png", dpi=200, bbox_inches='tight')
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
    plt.suptitle(f'SMC with Boundary Layer phi = {phi:.2f} — Velocity Errors', fontsize=16, y=0.98)
    plt.savefig(folder / "velocity_errors.png", dpi=200, bbox_inches='tight')
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
    plt.suptitle(f'Desired vs Actual Joint Positions phi = {phi:.2f}', fontsize=16, y=0.98)
    plt.savefig(folder / "tracking.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Управляющие моменты 
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.plot(times, torques[:, i], label=f'Joint {i}')
        plt.title(f'Control Torques (phi = {phi:.2f})')
        plt.xlabel('Time [s]')
        plt.ylabel('Torque [Nm]')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(folder / "torques_example.png", dpi=200, bbox_inches='tight')
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


def plot_tradeoff():
    """Строит график trade-off после всех запусков"""
    folder = Path("logs/plots/SMC_boundary_layer")
    folder.mkdir(parents=True, exist_ok=True)

    phi_values = []
    rmse_values = []
    torque_std_values = []

    for phi, res in results.items():
        phi_values.append(phi)
        rmse_values.append(res['rmse'])
        torque_std_values.append(res['torque_std'])

    plt.figure(figsize=(10, 6))
    plt.plot(phi_values, rmse_values, 'o-', label='RMSE Position Error', color='blue')
    plt.plot(phi_values, torque_std_values, 's-', label='Std of Torque (chattering measure)', color='red')
    plt.xlabel('Boundary Layer Thickness phi')
    plt.ylabel('Metric Value')
    plt.title('Trade-off: Accuracy vs. Chattering')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(folder / "tradeoff_analysis.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Таблица в консоль
    print("\n" + "="*80)
    print("TRADE-OFF ANALYSIS: Boundary Layer Thickness phi")
    print("="*80)
    print(f"{'phi':>8} | {'RMSE [rad]':>12} | {'Torque Std [Nm]':>16}")
    print("-" * 80)
    for phi in phis:
        r = results[phi]
        print(f"{phi:8.2f} | {r['rmse']:12.6f} | {r['torque_std']:16.3f}")
    print("="*80)

def main():
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    Path("logs/plots/SMC_boundary_layer").mkdir(parents=True, exist_ok=True)

    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        record_video=True,
        width=1920,
        height=1080
    )

    # Модификации модели (один раз)
    sim.modify_body_properties("end_effector", mass=4.0)
    sim.set_joint_damping(np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]))
    sim.set_joint_friction(np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1]))

    for phi in phis:
        print(f"\n=== Запуск SMC с Boundary Layer phi = {phi:.2f} ===")

        # Очистка данных
        for key in simulation_data:
            simulation_data[key].clear()

        # Создаём контроллер с текущим phi
        controller = create_controller(phi)

        # Установка контроллера и видео
        video_path = f"logs/videos/09_smc_boundary_phi_{phi:.2f}.mp4"
        sim.video_path = video_path
        sim.set_controller(controller)

        # Запуск
        sim.run(time_limit=15.0)

        # Вычисление метрик
        pos_err = np.array(simulation_data['desired_positions']) - np.array(simulation_data['positions'])
        rmse = np.sqrt(np.mean(pos_err**2))
        torque_std = np.std(np.array(simulation_data['torques']))

        results[phi] = {
            'rmse': rmse,
            'torque_std': torque_std,
            'data': {k: v.copy() for k, v in simulation_data.items()}
        }

        # Графики для этого phi
        plot_results(phi)

        print(f"phi = {phi:.2f}: RMSE = {rmse:.6f} rad, Torque std = {torque_std:.3f} Nm")

    # Финальный trade-off анализ
    plot_tradeoff()

    print("\nВсе симуляции завершены. Графики и видео сохранены в logs/")

if __name__ == "__main__":
    main()