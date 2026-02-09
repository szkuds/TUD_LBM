import os
import numpy as np
import matplotlib.pyplot as plt
import json


# TODO: Need to make a plotting config which will determine which plots are actually saved


def visualise(sim_instance, title="LBM Simulation Results"):
    """
    Visualizes simulation results by loading and plotting every saved timestep.
    This version includes a vector plot overlay on the velocity magnitude heatmap.

    Args:
        sim_instance: The completed simulation instance from the Run class.
        title (str): The base title for the plots.
    """
    try:
        # Get the directory where data files are stored
        data_dir = sim_instance.io_handler.data_dir
        run_dir = sim_instance.io_handler.run_dir

        # Create a new directory within the run to store the plots
        plot_dir = os.path.join(sim_instance.io_handler.run_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_dir_overview = os.path.join(plot_dir, 'overview')
        os.makedirs(plot_dir_overview, exist_ok=True)
        print(f"Saving plots to: {plot_dir}")

        # Density profile output directory and trackers
        density_profile_dir = os.path.join(plot_dir, "density_profile_analysis")
        os.makedirs(density_profile_dir, exist_ok=True)
        iter_nums = []
        ratio_rhos = []
        y_index = None

        # Get all data files and sort them by timestep
        files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
        if not files:
            print("No data files found to visualize.")
            return

        # Sort files based on the iteration number in the filename
        files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))

        # Loop through each timestep file and generate a plot
        for filename in files:
            file_path = os.path.join(data_dir, filename)
            timestep = int(filename.split("_")[-1].split(".")[0])

            data = np.load(file_path)
            final_rho = data["rho"]
            final_u = data["u"]
            final_force = data.get("force", None)
            if sim_instance.config.get('force_enabled', False):
                final_force_ext = data.get("force_ext")
            else:
                final_force_ext = None

            # Initialize density profile y-index (mid-plane by default or config override)
            if y_index is None:
                ny = final_rho.shape[1]
                y_index = sim_instance.config.get('density_profile_y', ny // 4)

            # load the config .json
            config = json.load(open(run_dir + "/config.json"))

            # Calculate density ratio and determine scaling
            if config['simulation_type'] == 'multiphase':
                density_ratio = config["rho_l"] / config["rho_v"]
                use_log_scale = density_ratio > 100
            else:
                use_log_scale = False

            fig, axes = plt.subplots(
                1,
                2 if final_force is None else (3 if final_force_ext is None else 4),
                figsize=(12 if final_force is None else 18, 5),
            )

            # Plot density with conditional log scaling
            if use_log_scale:
                density_data = final_rho[:, :, 0, 0].T
                density_data = np.maximum(
                    density_data, np.min(density_data[density_data > 0]) * 1e-10
                )
                im1 = axes[0].imshow(
                    density_data,
                    origin="lower",
                    cmap="viridis",
                    norm=plt.matplotlib.colors.LogNorm(),
                )
                axes[0].set_title("Density (Log Scale)")
            else:
                im1 = axes[0].imshow(final_rho[:, :, 0, 0].T, origin="lower", cmap="viridis")
                axes[0].set_title("Density")

            plt.colorbar(im1, ax=axes[0], label="Density")

            # --- Velocity Plotting with Vector Overlay ---

            # 1. Plot velocity magnitude heatmap
            vel_mag = np.sqrt(final_u[:, :, 0, 0] ** 2 + final_u[:, :, 0, 1] ** 2)
            im2 = axes[1].imshow(vel_mag.T, origin="lower", cmap="plasma")
            axes[1].set_title("Velocity")
            plt.colorbar(im2, ax=axes[1], label="Velocity Magnitude")

            # 2. Overlay velocity vector plot (quiver)
            nx, ny = final_u.shape[0], final_u.shape[1]
            x = np.arange(0, nx)
            y = np.arange(0, ny)
            X, Y = np.meshgrid(x, y)

            # Extract velocity components
            U = final_u[:, :, 0, 0]
            V = final_u[:, :, 0, 1]

            # Downsample the vectors to avoid a cluttered plot
            # Plot one vector every `skip` grid points
            skip = min(nx, ny)//10

            # Plotting quiver requires transposing h_i_prev and V to match the meshgrid and imshow orientation
            axes[1].quiver(
                X[::skip, ::skip],
                Y[::skip, ::skip],
                U.T[::skip, ::skip],
                V.T[::skip, ::skip],
                color="white",
                scale=None,
                scale_units="xy",
                angles="xy",
            )

            # --- End of Velocity Plotting Section ---

            if final_force is not None:
                # Plot force magnitude and vectors
                force_mag = np.sqrt(
                    final_force[:, :, 0, 0] ** 2 + final_force[:, :, 0, 1] ** 2
                )
                im3 = axes[2].imshow(force_mag.T, origin="lower", cmap="cividis")
                axes[2].set_title("Total Force")
                plt.colorbar(im3, ax=axes[2], label="Force Magnitude")
                U_force = final_force[:, :, 0, 0]
                V_force = final_force[:, :, 0, 1]
                axes[2].quiver(
                    X[::skip, ::skip],
                    Y[::skip, ::skip],
                    U_force.T[::skip, ::skip],
                    V_force.T[::skip, ::skip],
                    color="white",
                    scale=None,
                    scale_units="xy",
                    angles="xy",
                )

            if final_force_ext is not None:
                # Plot force magnitude and vectors
                force_mag = np.sqrt(
                    final_force_ext[:, :, 0, 0] ** 2 + final_force_ext[:, :, 0, 1] ** 2
                )
                im3 = axes[3].imshow(force_mag.T, origin="lower", cmap="cividis")
                axes[3].set_title("External Force")
                plt.colorbar(im3, ax=axes[3], label="Force Magnitude")
                U_force = final_force_ext[:, :, 0, 0]
                V_force = final_force_ext[:, :, 0, 1]
                axes[3].quiver(
                    X[::skip, ::skip],
                    Y[::skip, ::skip],
                    U_force.T[::skip, ::skip],
                    V_force.T[::skip, ::skip],
                    color="white",
                    scale=None,
                    scale_units="xy",
                    angles="xy",
                )

            for ax in axes:
                ax.set_xlabel("x")
                ax.set_ylabel("y")

            plt.suptitle(f"{title} - Timestep {timestep}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Save the figure to the plots directory
            plot_filename = os.path.join(plot_dir_overview, f"timestep_{timestep}.png")
            plt.savefig(plot_filename)
            plt.close(fig)  # Close the figure to free up memory

            # ---- Density profile analysis (per timestep) ----
            rho_line = final_rho[:, y_index, 0, 0]
            min_rho = float(np.min(rho_line))
            max_rho = float(np.max(rho_line))
            # Avoid division by zero
            safe_min = min_rho if min_rho > 0 else max(min_rho, 1e-16)
            ratio_rho = max_rho / safe_min if safe_min != 0 else np.inf

            iter_nums.append(timestep)
            ratio_rhos.append(ratio_rho)

            fig_dp, ax_dp = plt.subplots(1, 1, figsize=(8, 4))
            ax_dp.scatter(np.arange(rho_line.shape[0]), rho_line, s=6)
            ax_dp.set_xlabel("x")
            ax_dp.set_ylabel(f"density at y={y_index}")
            ax_dp.set_yscale("log")
            ax_dp.set_title(f"Density profile over x at y={y_index} (log), iter {timestep:05d}")
            ax_dp.legend([f"Min: {min_rho:.3g}, Max: {max_rho:.3g}, Ratio: {ratio_rho:.3g}"])
            dp_filename = os.path.join(density_profile_dir, f"density_profile_iter-{timestep:05d}.png")
            fig_dp.savefig(dp_filename)
            plt.close(fig_dp)
            # ---- End density profile analysis (per timestep) ----

        # ---- Combined simulation analysis plot ----
        log_path = os.path.join(run_dir, "simulation.log")
        iters_log = []
        umax_log = []
        avg_rho_log = []

        if os.path.isfile(log_path):
            try:
                with open(log_path, "r") as lf:
                    for line in lf:
                        if line.startswith("Step ") and "max_u=" in line and "avg_rho=" in line:
                            step_part, rest = line.split(":", 1)
                            step_str = step_part.split()[1]
                            iter_cur = int(step_str.split("/")[0])
                            avg_rho_val = None
                            umax_val = None
                            for token in rest.split(","):
                                token = token.strip()
                                if token.startswith("avg_rho="):
                                    avg_rho_val = float(token.split("=")[1])
                                elif token.startswith("max_u="):
                                    umax_val = float(token.split("=")[1])
                            if avg_rho_val is not None and umax_val is not None:
                                iters_log.append(iter_cur)
                                avg_rho_log.append(avg_rho_val)
                                umax_log.append(umax_val)
            except Exception as e:
                print(f"Failed to parse simulation.log: {e}")
        else:
            print("simulation.log not found, skipping log-based plots.")

        # Create combined figure with 3 subplots
        if iters_log and iter_nums:
            fig_analysis, axes_analysis = plt.subplots(3, 1, figsize=(10, 12))

            # Subplot 1: max_u vs iteration
            axes_analysis[0].scatter(iters_log, umax_log, s=10, color='tab:blue')
            axes_analysis[0].set_xlabel("Iteration")
            axes_analysis[0].set_ylabel("max_u")
            axes_analysis[0].set_title("Max velocity vs iteration")
            axes_analysis[0].grid(True, alpha=0.3)

            # Subplot 2: avg_rho vs iteration
            axes_analysis[1].scatter(iters_log, avg_rho_log, s=10, color='tab:green')
            axes_analysis[1].set_xlabel("Iteration")
            axes_analysis[1].set_ylabel("avg_rho")
            axes_analysis[1].set_title("Average density vs iteration")
            axes_analysis[1].grid(True, alpha=0.3)

            # Subplot 3: density_ratio vs iteration
            axes_analysis[2].scatter(iter_nums, ratio_rhos, s=10, color='tab:orange')
            axes_analysis[2].set_xlabel("Iteration")
            axes_analysis[2].set_ylabel("Density ratio (max/min)")
            axes_analysis[2].set_title("Density ratio vs iteration")
            axes_analysis[2].grid(True, alpha=0.3)

            plt.tight_layout()
            analysis_path = os.path.join(plot_dir, "simulation_analysis.png")
            fig_analysis.savefig(analysis_path, dpi=150)
            plt.close(fig_analysis)
            print(f"Combined simulation analysis plot saved to: {analysis_path}")
        # ---- End combined simulation analysis ----

        print(f"Finished generating plots for all {len(files)} timesteps.")
    except ImportError:
        print("Matplotlib not found. Please install it to visualize results.")
    # except Exception as e:
    #     print(f"An error occurred during visualization: {e}")
